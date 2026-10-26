"""MedWM v5: identity corruption x world-model proxy, on Synthea data, with real baselines and a routed trust pipeline.

What this version adds over v4:
  * DATA: real synthetic-EHR trajectories from Synthea (systolic BP), with realistic demographics. A toy fallback remains.
  * BASELINES: Fellegi-Sunter probabilistic linkage (own EM implementation) and demographic-key repair.
  * METHOD: the five-dimension trust score (completeness, temporal validity, identity consistency, provenance,
    cross-record consistency) -> AUTO_LINK / LINK_WITH_FLAG / QUARANTINE routing; flagged records are down-weighted.
  * STATS: 10 seeds, paired contrasts, CIs, data-kept, gate precision/recall, a figure.
  * LEAK-PROOF: gate/linkage/repair only receive the OBSERVABLE columns (no true patient id). Ground truth is used only to
    inject corruption, to score the gate, and to build the clean test set.
Usage:
  python medwm_v5.py --data synthea --csv synthea_sbp_trajectories.csv      (full run)
  python medwm_v5.py --data synthea --csv synthea_sbp_trajectories.csv --quick   (smoke test)
  python medwm_v5.py --data toy                                              (no Synthea needed)
"""
import argparse, difflib, re, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='synthea'); ap.add_argument('--csv', default='synthea_sbp_trajectories.csv')
ap.add_argument('--quick', action='store_true'); ap.add_argument('--seeds', type=int, default=10)
ap.add_argument('--epochs', type=int, default=30)
ARGS, _ = ap.parse_known_args()
SEEDS = [0] if ARGS.quick else list(range(ARGS.seeds))
RATES = [0.0, 0.2] if ARGS.quick else [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
EPOCHS = 3 if ARGS.quick else ARGS.epochs
WIN, BATCH, HIDDEN, LR, FLAG_W = 4, 256, 32, 3e-3, 0.5
TEST_FRAC = 0.25
NAME_VARIANT_P, DOB_ERROR_P = 0.08, 0.01

# ------------------------------------------------------------------------------------------------ data
def load_synthea_csv(path):
    d = pd.read_csv(path)       # columns: pid, visit, date, bio, first, last, dob, sex, fac
    return d

def add_noise(d, rng):
    """Observation-level demographic noise: name variants (typos/transliteration) and DOB keying errors."""
    d = d.copy(); n = len(d)
    d['name'] = (d['first'].astype(str) + ' ' + d['last'].astype(str)).str.strip()
    vm = rng.random(n) < NAME_VARIANT_P
    names = d['name'].to_numpy(object)
    for i in np.where(vm)[0]:
        k = rng.integers(0, len(names[i])); names[i] = names[i][:k] + names[i][k + 1:]
    d['name'] = names
    dm = rng.random(n) < DOB_ERROR_P
    dob = d['dob'].to_numpy().copy(); dob[dm] = (dob[dm] // 365) * 365 + rng.integers(0, 365, dm.sum()); d['dob'] = dob
    d['fkey'] = d['last'].astype(str).str[:3].str.upper()
    return d

def make_toy(rng, n=4000):
    G = ['Ahmed','Mohammed','Fatima','Aisha','Omar','Khalid','Sara','Hamza','Yusuf','Maryam','Ali','Hassan','Layla','Noor','Priya','Rahul','Anita','Imran','Sana','Bilal']
    F = ['Al-Mansoori','Hashimi','Khan','Sharma','Patel','Al-Farsi','Nair','Siddiqui','Qureshi','Hussain','Reddy','Iyer','Malik','Ansari','Rahman','Gupta','Sheikh','Verma','Mirza','Kapoor']
    rows = []
    for i in range(n):
        det = rng.random() < 0.2; slope = rng.normal(0.25, 0.05) if det else rng.normal(0, 0.03); x = rng.normal(0, 1)
        g, f, dob, sx = rng.choice(G), rng.choice(F), int(rng.integers(8000, 30000)), rng.choice(['M', 'F'])
        fac = rng.integers(0, 4); date = dob + 9000
        for v in range(int(rng.integers(8, 13))):
            if v: x += slope + rng.normal(0, 0.2); date += int(rng.integers(30, 200))
            rows.append((i, v, date, x, g, f, dob, sx, f'F{fac if rng.random() < 0.8 else rng.integers(0, 4)}'))
    return pd.DataFrame(rows, columns=['pid', 'visit', 'date', 'bio', 'first', 'last', 'dob', 'sex', 'fac'])

# ------------------------------------------------------------------------------------------- corruption
def corrupt(df, rate, rng):
    d = df.copy(); d['cid'] = d['pid'].astype(str)
    pats = d['pid'].unique(); k = int(len(pats) * rate)
    ch = rng.choice(pats, k, replace=False); coll, frag = ch[:k // 2], ch[k // 2:]
    for p in coll:
        q = rng.choice(pats[pats != p]); d.loc[d['pid'] == p, 'cid'] = str(q)
    for p in frag:
        idx = d.index[d['pid'] == p]; d.loc[idx[len(idx) // 2:], 'cid'] = f'{p}_F'
    pid_of = d.groupby('cid')['pid'].agg(lambda s: set(s)); fs = set(frag)
    bad = {c for c, s in pid_of.items() if len(s) > 1 or (s & fs)}
    return d, bad                                       # truth: used ONLY to score the gate

OBS = ['cid', 'visit', 'date', 'bio', 'name', 'dob', 'sex', 'fkey', 'fac']      # the only columns the methods may see

# ----------------------------------------------------------------------------------- trust score (5 dims)
def trust_table(obs, step_thr):
    """Per-identifier trust: five dimensions in [0,1], composite, and routing decision."""
    g = obs.sort_values(['visit', 'date'], kind='stable').groupby('cid')
    first = g.first(); sig = first['dob'].astype(str) + '|' + first['sex'] + '|' + first['fkey']
    dup = set(sig[sig.duplicated(keep=False)].index); recs = []
    for cid, grp in g:
        n = len(grp)
        comp = min(1.0, n / (WIN + 2))                                                           # completeness
        age_ok = ((grp['date'] - grp['dob']) >= 0).mean() * float((grp['dob'] >= 0).all())
        mono = float((np.diff(grp['date'].to_numpy()) >= 0).all()) if n > 1 else 1.0
        temp = 0.5 * age_ok + 0.5 * mono                                                         # temporal validity
        pen = 0.0
        if grp['dob'].nunique() > 1: pen += 0.5
        if grp['sex'].nunique() > 1: pen += 0.5
        nm = grp['name'].unique()
        if any(difflib.SequenceMatcher(None, a, b).ratio() < 0.8 for i, a in enumerate(nm) for b in nm[i + 1:]): pen += 0.5
        ident = max(0.0, 1.0 - pen)                                                              # identity consistency
        prov = 0.5 + 0.5 * grp['fac'].value_counts(normalize=True).iloc[0]                       # provenance
        jump = float(np.abs(np.diff(grp['bio'].to_numpy())).max() > step_thr) if n > 1 else 0.0
        cross = 1.0 - 0.6 * (cid in dup) - 0.4 * jump                                            # cross-record consistency
        recs.append((cid, comp, temp, ident, prov, cross))
    t = pd.DataFrame(recs, columns=['cid', 'completeness', 'temporal', 'identity', 'provenance', 'cross']).set_index('cid')
    t['score'] = (0.15 * t.completeness + 0.15 * t.temporal + 0.30 * t.identity + 0.10 * t.provenance + 0.30 * t.cross)
    hard = (t.identity <= 0.5) | (t.cross <= 0.4)                                                # same style as the API: min-dimension rule
    t['decision'] = np.where(hard | (t.score < 0.45), 'QUARANTINE', np.where(t.score < 0.75, 'LINK_WITH_FLAG', 'AUTO_LINK'))
    return t

# ------------------------------------------------------------------------------- repair and linkage
def key_repair(obs):
    """Re-link by observable demographic key (DOB, sex, family-name prefix). Ignores the supplied id."""
    return obs['dob'].astype(str) + '|' + obs['sex'] + '|' + obs['fkey']

def fellegi_sunter_repair(obs, post_thr=0.9, iters=40):
    """Split every id into demographic subgroups, then merge subgroups with Fellegi-Sunter (EM) linkage."""
    sg = key_repair(obs).rename('sg'); o = obs.assign(sg=sg)
    prof = o.groupby('sg').agg(name=('name', lambda s: s.mode().iloc[0]), dob=('dob', 'first'), sex=('sex', 'first'), fkey=('fkey', 'first'))
    prof['blk'] = prof['sex'] + (prof['dob'] // 365).astype(str)
    pairs = []
    for _, b in prof.groupby('blk'):
        ids = b.index.to_list()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, c = b.loc[ids[i]], b.loc[ids[j]]
                pairs.append((ids[i], ids[j], float(difflib.SequenceMatcher(None, a['name'], c['name']).ratio() >= 0.8),
                              float(a['dob'] == c['dob']), float(a['fkey'] == c['fkey'])))
    parent = {s: s for s in prof.index}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    if pairs:
        P = pd.DataFrame(pairs, columns=['a', 'b', 'f1', 'f2', 'f3']); F = P[['f1', 'f2', 'f3']].to_numpy()
        m, u, pi = np.array([.9, .8, .9]), np.array([.05, .001, .05]), 0.01             # EM for the Fellegi-Sunter parameters
        for _ in range(iters):
            lm = pi * np.prod(np.where(F == 1, m, 1 - m), 1); lu = (1 - pi) * np.prod(np.where(F == 1, u, 1 - u), 1)
            r = lm / (lm + lu + 1e-300)
            m = np.clip((r[:, None] * F).sum(0) / max(r.sum(), 1e-9), 1e-3, 1 - 1e-3)
            u = np.clip(((1 - r)[:, None] * F).sum(0) / max((1 - r).sum(), 1e-9), 1e-6, 1 - 1e-3); pi = float(np.clip(r.mean(), 1e-4, 0.5))
        for (a, b), pr in zip(P[['a', 'b']].itertuples(index=False), r):
            if pr >= post_thr: parent[find(a)] = find(b)
    return o['sg'].map(find)

# ---------------------------------------------------------------------------------------------- windows
def windows(d, key, weights=None):
    X, y, w = [], [], []
    for cid, grp in d.sort_values('visit', kind='stable').groupby(key):
        v = grp['bio'].to_numpy(); wt = 1.0 if weights is None else float(weights.get(cid, 1.0))
        if wt <= 0: continue
        for t in range(len(v) - WIN): X.append(v[t:t + WIN]); y.append(v[t + WIN]); w.append(wt)
    return np.array(X, np.float32), np.array(y, np.float32), np.array(w, np.float32)

def test_sets(test_df):
    Xw, yw, _ = windows(test_df.assign(cid=test_df['pid']), 'cid')
    seqs = [g.sort_values('visit')['bio'].to_numpy(np.float32) for _, g in test_df.groupby('pid')]
    return (Xw, yw), np.stack([s[:WIN + 3] for s in seqs if len(s) >= WIN + 3])

def train_eval(train, test_w, test_roll, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed); Xtr, ytr, wtr = train
    if len(ytr) < 200: return np.nan, np.nan
    class Net(nn.Module):
        def __init__(s): super().__init__(); s.g = nn.GRU(1, HIDDEN, batch_first=True); s.f = nn.Linear(HIDDEN, 1)
        def forward(s, x): o, _ = s.g(x); return s.f(o[:, -1, :]).squeeze(-1)
    m = Net(); opt = torch.optim.Adam(m.parameters(), lr=LR)
    X, y, w = torch.tensor(Xtr).unsqueeze(-1), torch.tensor(ytr), torch.tensor(wtr); gen = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y), BATCH):
            b = perm[i:i + BATCH]; opt.zero_grad()
            (w[b] * (m(X[b]) - y[b]) ** 2).sum().div(w[b].sum().clamp_min(1e-6)).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Xt, yt = test_w; mae1 = (m(torch.tensor(Xt).unsqueeze(-1)) - torch.tensor(yt)).abs().mean().item()
        h = torch.tensor(test_roll[:, :WIN]).clone(); ps = []
        for _ in range(3): p = m(h.unsqueeze(-1)); ps.append(p); h = torch.cat([h[:, 1:], p.unsqueeze(1)], 1)
        mae3 = (torch.stack(ps, 1) - torch.tensor(test_roll[:, WIN:WIN + 3])).abs().mean().item()
    return mae1, mae3

# -------------------------------------------------------------------------------------------------- main
def run(train_fn=train_eval):
    base = load_synthea_csv(ARGS.csv) if ARGS.data == 'synthea' else None
    rows, t0 = [], time.time()
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        df = add_noise(base if base is not None else make_toy(rng), rng)
        if ARGS.quick and len(df['pid'].unique()) > 800: df = df[df['pid'].isin(set(df['pid'].unique()[:800]))]
        pats = df['pid'].unique(); rng.shuffle(pats); nt = int(len(pats) * TEST_FRAC)
        test_df, train_df = df[df['pid'].isin(set(pats[:nt]))].copy(), df[df['pid'].isin(set(pats[nt:]))].copy()
        mu, sd = train_df['bio'].mean(), train_df['bio'].std()                          # scale with TRAIN statistics only
        for part in (test_df, train_df): part.loc[:, 'bio'] = (part['bio'] - mu) / sd
        tw, tr = test_sets(test_df)
        step_thr = 4 * 1.4826 * np.median(np.abs(np.diff(train_df.sort_values(['pid', 'visit'])['bio'].to_numpy())) + 1e-9)
        oracle = train_fn(windows(train_df.assign(cid=train_df['pid'].astype(str)), 'cid'), tw, tr, seed)
        pers1 = float(np.abs(tw[0][:, -1] - tw[1]).mean()); pers3 = float(np.abs(tr[:, WIN - 1:WIN] - tr[:, WIN:WIN + 3]).mean())   # sanity: 'predict the last value'
        for rate in RATES:
            rrng = np.random.default_rng(1000 * seed + int(rate * 100)); dc, bad = corrupt(train_df, rate, rrng)
            obs = dc[OBS]; assert 'pid' not in obs.columns
            t = trust_table(obs, step_thr); dec = t['decision']
            hard = {c: (0.0 if v == 'QUARANTINE' else 1.0) for c, v in dec.items()}
            soft = {c: (0.0 if v == 'QUARANTINE' else FLAG_W if v == 'LINK_WITH_FLAG' else 1.0) for c, v in dec.items()}
            fs = dc.copy(); fs['cid'] = fellegi_sunter_repair(obs).to_numpy()
            kr = dc.copy(); kr['cid'] = key_repair(obs).to_numpy()
            ft = trust_table(fs[OBS], step_thr); fsoft = {c: (0.0 if v == 'QUARANTINE' else FLAG_W if v == 'LINK_WITH_FLAG' else 1.0) for c, v in ft['decision'].items()}
            specs = {'naive': (windows(dc, 'cid'), len(dc)), 'gated_hard': (windows(dc, 'cid', hard), len(dc)),
                     'routed_flag_downweight': (windows(dc, 'cid', soft), len(dc)),
                     'key_repair': (windows(kr, 'cid'), len(dc)), 'fellegi_sunter': (windows(fs, 'cid'), len(dc)),
                     'fs_plus_routed': (windows(fs, 'cid', fsoft), len(dc))}
            q = {c for c, v in dec.items() if v == 'QUARANTINE'}; tp = len(bad & q)
            prec, rec = tp / max(len(q), 1), tp / max(len(bad), 1)
            base_n = max(len(specs['naive'][0][1]), 1); res = {'oracle': (oracle, 1.0)}
            for name, (tr_xy, _) in specs.items(): res[name] = (train_fn(tr_xy, tw, tr, seed), float(tr_xy[2].sum() / base_n))
            for name, ((m1, m3), kept) in res.items():
                rows.append(dict(seed=seed, rate=rate, cond=name, mae1=m1, mae3=m3, effective_data=kept,
                                 gate_precision=prec if name == 'gated_hard' else np.nan, gate_recall=rec if name == 'gated_hard' else np.nan))
            rows.append(dict(seed=seed, rate=rate, cond='persistence', mae1=pers1, mae3=pers3, effective_data=1.0, gate_precision=np.nan, gate_recall=np.nan))
            print(f'seed {seed} rate {rate:.2f} done ({time.time() - t0:.0f}s)', flush=True)
    raw = pd.DataFrame(rows); raw.to_csv('v5_raw.csv', index=False)
    om, pm = raw[raw.cond == 'oracle']['mae3'].mean(), raw[raw.cond == 'persistence']['mae3'].mean()
    print(f'SANITY: oracle 3-step MAE {om:.4f} vs persistence baseline {pm:.4f}')
    if not om < pm: print('WARNING: the oracle model does not beat the trivial persistence baseline, so the model has not learned; do NOT interpret the comparisons (increase epochs / check scaling).')
    agg = raw.groupby(['rate', 'cond']).agg(mae1=('mae1', 'mean'), mae3=('mae3', 'mean'), mae3_sd=('mae3', 'std'), n=('mae3', 'size'),
          effective_data=('effective_data', 'mean'), gate_precision=('gate_precision', 'mean'), gate_recall=('gate_recall', 'mean')).reset_index()
    agg['mae3_ci95'] = 1.96 * agg['mae3_sd'] / np.sqrt(agg['n']); agg.to_csv('v5_summary.csv', index=False)
    out = []
    for rate in RATES:
        p = raw[raw.rate == rate].pivot(index='seed', columns='cond', values='mae3')
        for c in [c for c in p.columns if c not in ('oracle', 'naive', 'persistence')] + ['naive']:
            for ref in ('naive', 'oracle'):
                if c == ref: continue
                x = p[c] - p[ref]; out.append(dict(rate=rate, contrast=f'{c} - {ref}', mean_diff=x.mean(), ci95=1.96 * x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                                                   seeds_worse=int((x > 0).sum()), n=len(x)))
    pd.DataFrame(out).round(4).to_csv('v5_paired_contrasts.csv', index=False)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for c, g in agg.groupby('cond'):
            ax.errorbar(g['rate'] * 100, g['mae3'], yerr=g['mae3_ci95'].fillna(0), marker='o', capsize=2, label=c, linestyle='--' if c == 'oracle' else '-')
        ax.set_xlabel('Identity corruption rate (%)'); ax.set_ylabel('3-step rollout MAE (lower is better)'); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig('v5_results.png', dpi=300)
    except Exception as e: print('figure skipped:', e)
    print(agg.pivot(index='rate', columns='cond', values='mae3').round(4).to_string()); return raw

if __name__ == '__main__':
    run()
