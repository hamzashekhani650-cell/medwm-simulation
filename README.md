Identity Repair and Quarantine in Longitudinal Health Data

Independent Undergraduate Research Project
Research area: Health Informatics, Patient Identity Resolution and Clinical Data Quality
Intended venue: AAAI 2027 Workshop on Medical World Models (MedWM)
Author: Hamza Shekhani
Year: 2026

About the Project

Patient identity errors can affect the quality of longitudinal electronic health records (EHRs). For example, records belonging to different patients may be incorrectly merged, while records belonging to the same patient may be separated.

These problems can affect how clinical information is interpreted and potentially influence machine learning models trained on longitudinal health data.

This project explores how patient identity errors affect clinical prediction models and compares three approaches to handling corrupted records:

1. Naive: Use the corrupted dataset without attempting identity correction.
2. Quarantine: Exclude records identified as unreliable from model training.
3. Repair: Use probabilistic record linkage to identify and reconstruct patient records using available demographic information.

The main question behind the project is whether removing unreliable records is always the best approach, or whether attempting to repair patient identities can preserve useful information for downstream analysis.

Research Questions

The experiments explore three questions:

* How does increasing identity corruption affect longitudinal prediction performance?
* Does removing unreliable records reduce model performance by limiting the available training data?
* Can probabilistic record linkage help preserve prediction performance when identity errors are introduced?

Experimental Approach

1. Synthetic Dataset

The project uses synthetic patient data generated with Synthea.

The final dataset contains 2,672 synthetic patients, filtered to include individuals with 8–12 longitudinal systolic blood pressure (SBP) observations.

No real patient records are used.

2. Introducing Identity Errors

Identity corruption is simulated at different rates:

0%, 5%, 10%, 20%, 30%, 50%

The experiments investigate the effects of identity collisions and associated demographic inconsistencies on longitudinal patient records.

3. Identity Repair

The repair approach uses an implementation of the Fellegi–Sunter probabilistic record linkage method, with Expectation-Maximization (EM) for parameter estimation.

The matching process uses three observable demographic attributes:

* Date of birth
* Sex
* Name prefix

Ground-truth patient identifiers are not used as matching inputs. They are reserved for simulation and evaluation purposes.

4. Prediction Model

A Gated Recurrent Unit (GRU) model, implemented using PyTorch, is used to predict longitudinal SBP trajectories.

Model performance is measured using Mean Absolute Error (MAE) over a three-step prediction rollout. Lower MAE indicates lower prediction error.

Each experiment is repeated across 10 independent random seeds. Results are compared against a clean-data baseline, referred to as the Oracle.

Results

The final v6 experiment produced the following mean MAE values:

Approach	Corruption rate	Mean MAE
Oracle (clean baseline)	0%	0.3435
Naive	50%	0.3510
Quarantine	50%	0.3508
Repair	50%	0.3456

At 50% corruption, the Naive and Quarantine approaches show higher prediction error than the clean baseline.

The Repair approach produces a result closer to the Oracle baseline.

Paired Comparisons

At 50% corruption, the paired comparisons reported:

* Quarantine: Mean MAE difference of 0.0073 compared with Oracle; higher error in all 10 seeds.
* Repair: Mean MAE difference of 0.0021 compared with Oracle; higher error in 8 out of 10 seeds.

These results suggest that, under the conditions simulated in this project, probabilistic repair may preserve more useful longitudinal information than excluding potentially unreliable records.

Repository Contents

File	Description
medwm_v6.py	Main simulation script, including corruption, matching and model evaluation
plot_fix.py	Script used to generate the results figure
v6_summary.csv	Aggregated experimental results
v6_raw.csv	Results from individual experimental runs
v6_paired_contrasts.csv	Paired comparisons against the Oracle baseline
synthea_sbp_trajectories.csv	Synthetic longitudinal SBP dataset
requirements.txt	Required Python packages

Running the Project

Requirements

* Python 3.9 or later
* Packages listed in requirements.txt

Install the dependencies:

pip install -r requirements.txt

Run the simulation

Ensure that the synthetic dataset is available in the repository’s root directory.

python medwm_v6.py --data synthea --csv synthea_sbp_trajectories.csv

Generate the results figure

python plot_fix.py

The plotting script generates the results visualisation using the experimental outputs.

Limitations

This project is a simulation-based investigation using synthetic health data. Its findings should not be interpreted as evidence that the same results would necessarily occur in real hospital information systems or deployed health information exchanges.

The experiments focus on a limited number of demographic attributes and a single clinical variable, systolic blood pressure. Real-world identity resolution involves additional challenges, including incomplete demographic information, inconsistent data entry, changing patient information and differences between healthcare systems.

Further testing with more diverse datasets and clinical variables would be needed to understand how well these approaches generalise to practical healthcare settings.

Development and AI Assistance

This project was undertaken as an independent undergraduate research initiative.

AI-assisted tools were used extensively throughout the development process, particularly for Python implementation, debugging, methodological exploration and documentation.

The research idea, choice of problem, experimental objectives and direction of development were led by the author. The experimental workflow was developed iteratively, with outputs and methodological issues reviewed throughout the process.

The repository is shared to document the project, its experimental approach and its results.

Citation

If referring to this project, please use the following citation:

@misc{shekhani2026identity,
  author    = {Shekhani, Hamza},
  title     = {Identity Trust in Health Information Exchange},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22770188}
}

Acknowledgements

Synthetic patient data were generated using Synthea.
