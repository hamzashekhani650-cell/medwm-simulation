Identity Repair and Quarantine in Longitudinal Health Data

A small undergraduate research project exploring how patient-identity
errors can affect longitudinal health data.

────────

About the project

Health information systems may contain records that are incorrectly
linked to the same patient or split across different patient identities.
These errors can make a patient’s medical history incomplete or
misleading.

This project explores a practical question:

> **When patient identity information is unreliable, is it better to
> repair the records, quarantine uncertain records, or leave the data
> unchanged?**

The experiments use synthetic health data to compare these approaches
and examine how they affect a simple model that predicts future systolic
blood pressure (SBP).

This is an exploratory simulation, not a clinical system or a
validated patient-matching tool.

What the project compares

Three approaches are evaluated under different levels of simulated
identity corruption:

────────

Approach                            What it does

────────

Naive                           Uses the records without attempting
identity correction.

Quarantine                      Removes records considered
unreliable.

Repair                          Uses probabilistic record linkage to identify records that may belong together.

A clean-data Oracle result is also included as a reference point. It
represents performance when the original patient identities are known.

How the experiment was set up

• Data: Synthetic patient records generated using
Synthea.
• Sample: 2,672 synthetic patients, with longitudinal systolic
blood pressure observations selected for the experiment.
• Corruption levels: 0%, 5%, 10%, 20%, 30% and 50%.
• Record linkage: An Expectation-Maximisation implementation of
the Fellegi–Sunter probabilistic record-linkage approach.
• Matching information: Date of birth, sex and name prefix. The
true patient identifiers are not used as matching inputs.
• Prediction task: A GRU model implemented in PyTorch, using
earlier SBP observations to predict later values.
• Evaluation: Three-step rollout mean absolute error (MAE),
averaged over 10 random seeds. Lower MAE indicates lower prediction
error.

The true synthetic patient identifiers are available for constructing
and evaluating the simulated corruption, but are kept separate from the
matching information.

Results

The table below shows the mean MAE at the highest tested corruption
level (50%).

Approach                              Mean MAE

────────

Oracle (clean identity reference)       0.3435
Repair                                  0.3456
Quarantine                              0.3508
Naive                                   0.3510

At 50% corruption, the repair approach had a lower mean error than the
naive and quarantine approaches in this experiment. Its result was also
closer to the clean-data Oracle reference.

These results are specific to this synthetic setup. They should not
be interpreted as proof that record repair will always outperform
quarantine, or that the approach is ready for use with real patient
records.

Results figure

If the generated figure is included in the repository, it can be
displayed here:

Experimental results

Repository contents

The repository is intended to include the following files:

────────

File                                Purpose

────────

medwm_v6.py                       Main experiment script.

plot_fix.py                       Script used to generate the results
figure.

synthea_sbp_trajectories.csv      Synthetic SBP trajectory data used
by the experiment.

v6_summary.csv                    Summary results across corruption
levels.

v6_raw.csv                        Results from individual
experimental runs.

v6_paired_contrasts.csv           Paired comparisons against the
clean-data reference.

requirements.txt                  Python package requirements.

v6_results_paper_ready.png        Experimental results figure.

File names may need to be adjusted if the repository contents change.

Running the project

The following commands show the intended workflow. Run them from the
project folder in a Python environment with the required packages
installed.

1. Install the dependencies

pip install -r requirements.txt

2. Run the experiment

python medwm_v6.py --data synthea --csv synthea_sbp_trajectories.csv

3. Generate the figure

python plot_fix.py

The scripts and file names should be checked against the current
repository before relying on these commands.

Limitations

There are several important limitations to this project:

• The experiments use synthetic data rather than real hospital or
health-information-exchange records.
• The study focuses on a limited set of matching attributes: date of
birth, sex and name prefix.
• The prediction task uses one clinical variable, systolic blood
pressure.
• The results depend on the simulation design, corruption process,
matching assumptions and model configuration.
• The experiment does not establish how the methods would perform in a
live hospital environment.
• The findings should be treated as exploratory and would need testing
with appropriate real-world data before any practical use.

Development and use of AI tools

AI-assisted tools were used extensively during the development of this
project, particularly for Python implementation, debugging, exploring
methods and preparing documentation.

The research question, choice of problem, experimental objectives and
overall direction were developed and guided by me. I worked iteratively
with the tools, reviewed the outputs and considered methodological
issues as the project developed.

I have included this note to describe the development process
transparently. The code and results should be reviewed and reproduced
before the work is used or cited.

Research status

This project is being prepared with the AAAI 2027 Workshop on Medical
World Models (MedWM) in mind. This indicates an intended submission
venue only; it should not be read as an acceptance or publication.

Reference

A related project record is available on Zenodo:

Shekhani, H. (2026). Identity Trust in Health Information Exchange.
Zenodo.
https://doi.org/10.5281/zenodo.22770188

Acknowledgement

Synthetic patient data were generated using
Synthea.
