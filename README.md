# Understanding Fairness and Explainability in Multimodal Approaches within Healthcare
## NHSE PhD Internship Project

### About the Project

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)
[![PyPI - License](https://img.shields.io/pypi/l/nhssynth)](https://github.com/nhsengland/nhssynth/blob/main/LICENSE)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/nhsengland/mm-healthfair/main.svg)](https://results.pre-commit.ci/latest/github/nhsengland/mm-healthfair/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

This repository holds code for the Understanding Fairness and Explainability in Multimodal Approaches within Healthcare project (MM-HealthFair). The MM-HealthFair framework was designed to support flexible classification pipelines, providing an end-to-end
pipeline for multimodal fusion, evaluation and fairness investigation.
See the [original project proposal](https://nhsx.github.io/nhsx-internship-projects/advances-modalities-explainability/) for more information.

_**Note:** Only public or fake data are shared in this repository._

### Project Stucture

- The main code is found in the root of the repository (see Usage below for more information)
- A summary of the key functionalities of the project is available on the [index](./docs/index.md) page.
- The report from the previous project iteration [report](./report/) is also available in the `reports` folder
- More information about the code usage can be found in the [model card](./model_card.md)

### Built With

[![Python v3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)

In the latest iteration, the framework was developed locally using [**Python** v3.10.11](https://www.python.org/downloads/release/python-31011/) and tested on a Windows 11 machine with GPU support (NVIDIA GeForce RTX 3080, 16 GiB VRAM). Additionally, model training and evaluation were performed on a Microsoft Azure machine using a Windows 10 Server with the following specifications: 
- 1 x NVIDIA Tesla T4 GPU
- 4 x vCPUs (28 GiB memory)

### Getting Started

#### Installation

To get a local copy up and running follow these simple steps.

To clone the repo:

`git clone https://github.com/nhsengland/mm-healthfair`

To create a suitable environment:

1. Use pip + requirements.txt
- ```python -m venv _env```
- `source _env/bin/activate`
- `pip install -r requirements.txt`

2. Use poetry (*recommended*)
- Install poetry (see [website](https://python-poetry.org) for documentation)
- Navigate to project root directory `cd mm-healthfair`
- Create environment from poetry lock file: `poetry install`
- Run scripts using `poetry run python3 xxx.py`

_Note:_ There are known issues when installing the scispacy package for Python versions >3.10 or Apple M1 chips. Project dependencies strictly require py3.10 to avoid this, however OSX users may need to manually install nmslib with `CFLAGS="-mavx -DWARN(a)=(a)" pip install nmslib` to circumvent this issue (see open issue https://github.com/nmslib/nmslib/issues/476).

_Note:_ To enable support for platforms with CPU-only compute units, you should remove the `source="pytorch-gpu"` arguments from `pyproject.toml` before installing the PyTorch libraries.

### Usage
This repository contains code used to generate an evaluate multimodal deep learning pipelines for risk prediction using demographic, time-series and clinical notes data from [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/). Additionally, it includes functionalities for adversarial mitigation (controlling model dependence on sensitive attributes), fairness analysis with bootstrapping and explainability using [SHAP](https://shap.readthedocs.io/en/latest/) and [MM-SHAP](https://github.com/Heidelberg-NLP/MM-SHAP/) scores for examining multimodal feature importance.

To reproduce the experiments, refer to the [Getting Started](./docs/getting_started.md) page for a detailed walkthrough.


#### Outputs
- Preprocessed multimodal features from MIMIC-IV 3.1 and related dictionaries.
- Multimodal learner artifacts (model checkpoints).
- Performance, fairness and explainability summaries mapped by artifact name (coded as `<outcome>_<fusion_type>_<modalities>`, e.g. `ext_stay_7_concat_static_timeseries_notes`).
- Notebooks for debugging, inference relative to the generated dictionary files throughout the pipeline.

#### Datasets
The MIMIC-IV dataset (v3.1) can be downloaded from [PhysioNet.org](https://physionet.org) after completion of mandatory training. This project makes use of four main modules linked to the MIMIC-IV dataset:

- _hosp_: measurements recorded during hospital stay for training, including demographics, lab tests, prescriptions, diagnoses and care provider orders
- _ed_: records metadata during ED attendance in an externally linked database
- _icu_: records individuals with associated ICU admission during the episode with additional metadata (used mainly for measuring the ICU admission outcome)
- _note_: records deidentified discharge summaries as long form narratives which describe reason for admission and relevant hospital events

Additional linked datasets include [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/) (v2.2), [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) (v2.2) and [MIMIC-IV-Ext-BHC](https://physionet.org/content/labelled-notes-hospital-course/1.2.0/) (v1.2.0) as an external dataset for extracting Brief Hospital Course segments within a discharge summary. Further information can be found in PhysioNet's [documentation](https://mimic.mit.edu/).

### Roadmap

See the repo [issues](https://github.com/nhsengland/mm-healthfair/issues) for a list of proposed features (and known issues).

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### License

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

### Contact

To find out more about the [Analytics Unit](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch at [england.tdau@nhs.net](mailto:england.tdau@nhs.net).

<!-- ### Acknowledgements -->
