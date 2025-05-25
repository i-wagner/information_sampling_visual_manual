This repository contains the data and analysis code, required to reproduce the results for the publication
Wagner, I., Tünnermann, J., Schubö, A., & Schütz, A. C. (2024). Trade-off between search costs and accuracy in oculomotor and manual search tasks. Journal of Neurophysiology. https://doi.org/10.1152/jn.00488.2024

The data is available in the Zenodo repository, under the DOI 10.5281/zenodo.15179979

This folder follow the following structure:
```bash
/
└── information_sampling_visual_manual
    ├── 1_code (code to ro run the experiment)
    │   ├── exp_main.m (Entry point; runs one condition of the main experiment for a participant)
    │   ├── exp_main.m (Entry point; runs one set of demo trials for a given condition and a given participant)
    │   ├── _lib (various helper scripts and functions)
    ├── 3_analysis (Scripts to reproduce all analysis from the article)
    │   ├── analysis_pipeline.m (Entry point; runs the entire analysis pipeline end to end)
    │   ├── model (Scripts for model fitting)
    │   └── statistics (Scripts, .jasp files, and .csv exports to reproduce statistical analysis)
    │       ├── tTestsRobust.R (Entry point to run robust t-tests)
    │       └── *.jasp (JASP files for remaining statistical tests)
```
