"""MedWM v4: GRU world-model proxy under identity corruption, with LEAK-FREE gate and repair.

Design rules (what was wrong in the earlier GRU script):
  * The gate and the repair step receive an OBSERVABLE-ONLY table (no true patient id, no label). Ground truth is
    kept in a separate array and is used only to inject corruption, score the gate, and build the clean test set.
  * 'Repair' re-links records from observable demographics; it never copies the true id.
  * Train/test split is by TRUE patient BEFORE corruption; the test set is clean and identical for all conditions.
  * Minibatch training, fixed epochs (no tuning on the test set), 1-step and 3-step rollout MAE, many seeds, paired CIs.
Run:  python medwm_gru_v4.py            (full)   |   python medwm_gru_v4.py --quick   (smoke test)
"""
import argparse, difflib, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); ARGS, _ = ap.parse_known_args()
N_PATIENTS = 1500 if ARGS.quick else 6000
SEEDS = [0] if ARGS.quick else list(range(10))
RATES = [0.0, 0.2] if ARGS.quick else [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
EPOCHS = 3 if ARGS.quick else 30
WIN, BATCH, HIDDEN, LR = 4, 256, 32, 3e-3          # window length, batch size, GRU width, learning rate
N_TEST = 500 if ARGS.quick else 1000

GIVEN = ['Ahmed','Mohammed','Fatima','Aisha','Omar','Khalid','Sara','Hamza','Yusuf','Maryam','Ali','Hassan','Layla','Noor','Zainab','Ibrahim','Rashid','Salim','Mariam','Abdullah','Priya','Rahul','Anita','Vijay','Sunil','Deepa','Imran','Sana','Bilal','Hina','Tariq','Nadia','Faisal','Reem','Arjun','Kavya','Suresh','Meena','Karim','Dina']
FAMILY = ['Al-Mansoori','Hashimi','Khan','Sharma','Patel','Al-Farsi','Nair','Siddiqui','Qureshi','Hussain','Al-Nuaimi','Reddy','Iyer','Malik','Chaudhry','Al-Suwaidi','Menon','Ansari','Rahman','Gupta','Al-Kaabi','Joshi','Bhat','Sheikh','Verma','Al-Hammadi','Pillai','Mirza','Naqvi','Kapoor']

# ----------------------------------------------------------------------------------------------- data
def make_patients(rng, n):
    det = rng.random(n) < 0.2
    slope = np.where(det, rng.normal(0.25, 0.05, n), rng.normal(0.0, 0.03, n))
    nv = rng.integers(6, 11, n); base = rng.normal(0, 1.0, n)
    given, fam = rng.choice(GIVEN, n), rng.choice(FAMILY, n)
    dob, sex = rng.integers(0, 60 * 365, n), rng.choice(['M', 'F'], n)
    rows = []
    for i in range(n):
        x = base[i]
        for v in range(nv[i]):
            if v: x = x + slope[i] + rng.normal(0, 0.2)
            name = f'{given[i]} {fam[i]}'
            if rng.random() < 0.08:                                   # transliteration/typo variant
                k = rng.integers(0, len(name)); name = name[:k] + name[k + 1:]
            d = dob[i]
            if rng.random() < 0.01: d = (d // 365) * 365 + rng.integers(0, 365)   # DOB keying error
            rows.append((i, v, x, name, d, sex[i], fam[i][:3].upper()))
    return pd.DataFrame(rows, columns=['pid', 'visit', 'bio', 'name', 'dob', 'sex', 'fkey'])

def corrupt(df, rate, rng):
    """Returns the corrupted table (with 'cid') and the set of cids that are truly corrupted (for scoring only)."""
    d = df.copy(); d['cid'] = d['pid'].astype(str)
    pats = d['pid'].unique(); k = int(len(pats) * rate)
    ch = rng.choice(pats, k, replace=False); coll, frag = ch[:k // 2], ch[k // 2:]
    for p in coll:                                                    # p's records filed under another patient's id
        q = rng.choice(pats[pats != p]); d.loc[d['pid'] == p, 'cid'] = str(q)
    for p in frag:                                                    # later half of visits filed under a new id
        idx = d.index[d['pid'] == p]; d.loc[idx[len(idx) // 2:], 'cid'] = f'{p}_F'
    pid_of = d.groupby('cid')['pid'].agg(lambda s: set(s))
    bad = {c for c, s in pid_of.items() if len(s) > 1}
    frag_pids = {p for p in frag}
    bad |= {c for c, s in pid_of.items() if s & frag_pids}            # both halves of a fragmented patient
    return d, bad

OBS = ['cid', 'visit', 'bio', 'name', 'dob', 'sex', 'fkey']           # the ONLY columns gate/repair may see

def gate_flags(obs):
    """Observable-only trust signals -> set of cids to quarantine."""
    q, g = set(), obs.groupby('cid')
    for cid, grp in g:                                                # identity consistency within one identifier
        if grp['dob'].nunique() > 1 or grp['sex'].nunique() > 1: q.add(cid); continue
        nm = grp['name'].unique()
        if any(difflib.SequenceMatcher(None, a, b).ratio() < 0.8 for i, a in enumerate(nm) for b in nm[i + 1:]): q.add(cid)
    f = g.first(); sig = f['dob'].astype(str) + '|' + f['sex'] + '|' + f['fkey']
    q.update(sig[sig.duplicated(keep=False)].index)                   # cross-record consistency: same person, several ids
    return q

def repair_cids(obs):
    """Re-link by observable demographic key; the supplied id is ignored."""
    return obs['dob'].astype(str) + '|' + obs['sex'] + '|' + obs['fkey']

# --------------------------------------------------------------------------------------------- windows
def windows(d, key):
    """All consecutive windows of length WIN -> next value, built per identifier group in visit order."""
    X, y = [], []
    for _, grp in d.sort_values('visit', kind='stable').groupby(key):
        v = grp['bio'].to_numpy()
        for t in range(len(v) - WIN):
            X.append(v[t:t + WIN]); y.append(v[t + WIN])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def test_sets(test_df):
    Xw, yw = windows(test_df.assign(cid=test_df['pid']), 'cid')
    seqs = [g.sort_values('visit')['bio'].to_numpy(np.float32) for _, g in test_df.groupby('pid')]
    seqs = [s for s in seqs if len(s) >= WIN + 3]
    return (Xw, yw), np.stack([s[:WIN + 3] for s in seqs])            # (1-step windows), (first WIN+3 values per patient)

# ------------------------------------------------------------------------------------------ GRU (torch)
def train_eval(train_xy, test_w, test_roll, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    Xtr, ytr = train_xy
    if len(ytr) < 200: return np.nan, np.nan
    class GRUNet(nn.Module):
        def __init__(s): super().__init__(); s.gru = nn.GRU(1, HIDDEN, batch_first=True); s.fc = nn.Linear(HIDDEN, 1)
        def forward(s, x): out, _ = s.gru(x); return s.fc(out[:, -1, :]).squeeze(-1)
    m = GRUNet(); opt = torch.optim.Adam(m.parameters(), lr=LR); lossf = nn.MSELoss()
    X = torch.tensor(Xtr).unsqueeze(-1); y = torch.tensor(ytr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), BATCH):
            b = perm[i:i + BATCH]; opt.zero_grad(); lossf(m(X[b]), y[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Xt, yt = test_w
        mae1 = (m(torch.tensor(Xt).unsqueeze(-1)) - torch.tensor(yt)).abs().mean().item()
        h = torch.tensor(test_roll[:, :WIN]).clone(); preds = []
        for _ in range(3):                                            # 3-step rollout from the first WIN true values
            p = m(h.unsqueeze(-1)); preds.append(p); h = torch.cat([h[:, 1:], p.unsqueeze(1)], 1)
        mae3 = (torch.stack(preds, 1) - torch.tensor(test_roll[:, WIN:WIN + 3])).abs().mean().item()
    return mae1, mae3

# ------------------------------------------------------------------------------------------------ main
def run(train_fn=train_eval):
    rows = []; t0 = time.time()
    for seed in SEEDS:
        rng = np.random.default_rng(seed); df = make_patients(rng, N_PATIENTS)
        pats = df['pid'].unique(); rng.shuffle(pats)
        test_df = df[df['pid'].isin(set(pats[:N_TEST]))]; train_df = df[df['pid'].isin(set(pats[N_TEST:]))]
        tw, tr = test_sets(test_df)
        oracle = train_fn(windows(train_df.assign(cid=train_df['pid']), 'cid'), tw, tr, seed)   # oracle does not depend on the rate
        for rate in RATES:
            rrng = np.random.default_rng(1000 * seed + int(rate * 100))                          # independent stream per (seed, rate)
            dc, bad = corrupt(train_df, rate, rrng)
            obs = dc[OBS]                                                                        # <- no pid, no label beyond this point
            flagged = gate_flags(obs)
            rep = dc.copy(); rep['cid'] = repair_cids(obs)
            robs = rep[OBS]; rflag = gate_flags(robs)
            conds = {'naive': dc, 'gated': dc[~dc['cid'].isin(flagged)], 'repair': rep, 'repair+gate': rep[~rep['cid'].isin(rflag)]}
            tp = len(bad & flagged); prec = tp / max(len(flagged), 1); rec = tp / max(len(bad), 1)
            out = {'oracle': oracle}
            for name, dd in conds.items(): out[name] = train_fn(windows(dd, 'cid'), tw, tr, seed)
            for name, (m1, m3) in out.items():
                rows.append(dict(seed=seed, rate=rate, cond=name, mae1=m1, mae3=m3,
                                 data_kept=(len(conds[name]) / len(dc)) if name in conds else 1.0,
                                 gate_precision=prec if name == 'gated' else np.nan, gate_recall=rec if name == 'gated' else np.nan))
            print(f'seed {seed} rate {rate:.2f} done ({time.time() - t0:.0f}s)', flush=True)
    raw = pd.DataFrame(rows); raw.to_csv('v4_gru_raw.csv', index=False)
    agg = raw.groupby(['rate', 'cond']).agg(mae1=('mae1', 'mean'), mae1_sd=('mae1', 'std'), mae3=('mae3', 'mean'), mae3_sd=('mae3', 'std'),
          n=('mae3', 'size'), data_kept=('data_kept', 'mean'), gate_precision=('gate_precision', 'mean'), gate_recall=('gate_recall', 'mean')).reset_index()
    agg['mae3_ci95'] = 1.96 * agg['mae3_sd'] / np.sqrt(agg['n']); agg.to_csv('v4_gru_summary.csv', index=False)
    out = []                                                                                      # paired contrasts (same seed, same test set)
    for rate in RATES:
        p = raw[raw.rate == rate].pivot(index='seed', columns='cond', values='mae3')
        for lab, x in [('naive-oracle', p.naive - p.oracle), ('gated-naive', p.gated - p.naive), ('repair-naive', p.repair - p.naive),
                       ('repair+gate-naive', p['repair+gate'] - p.naive), ('repair-oracle', p.repair - p.oracle)]:
            out.append(dict(rate=rate, contrast=lab, mean_diff=x.mean(), ci95=1.96 * x.std() / np.sqrt(max(len(x), 1)), seeds_worse=int((x > 0).sum()), n=len(x)))
    pd.DataFrame(out).round(4).to_csv('v4_gru_paired_contrasts.csv', index=False)
    print(agg.pivot(index='rate', columns='cond', values='mae3').round(4).to_string()); return raw

if __name__ == '__main__':
    run()