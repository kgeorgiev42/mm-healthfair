# Model Card: Multimodal Deep Fusion Models

## Model Details

The implementation of the multi-modal fusion models within this repository is based on work done by Sophie Martin and Konstantin Georgiev during an NHSE PhD internship. In this project, we use an intermediate fusion approach to fuse three data modalities (tabular EHR data, time-series vitals and lab tests data and free-text discharge summaries) using concatenation with equal weight. Each modality is trained on a separate deep neural net component and fused at the final hidden layer. The currently implemented network components are:
- **EHR (MLP)**: Multi-layer Perceptron classifier for tabular data.
- **TS (LSTM)**: 2-component LSTM classifier (for vital signs and lab measurements data).
- **NT (TF-E)**: Transformer-encoder network for free-text (discharge summaries) data.
Features from the previous project iteration include the multi-modal attention gate (MAG+) which was adapted from [this repository](https://github.com/emnlp-mimic/mimic/blob/main/base.py#L136) and is inspired by the work of [Zhao et al.](https://ieeexplore.ieee.org/document/9746536) to integrate multiple data modalities in and end-to-end training regime.

Tested multimodal combinations include:
- **IF-EHR+TS**: Intermediate fusion with concatenation over tabular and time-series data.
- **IF-EHR+TS+NT**: Intermediate fusion with concatenation over all three modalities.

## Model Use

### Intended Use

This model is intended for use in training a risk prediction model, supporting classification over four come outcomes at point of arrival to the emergency department: **in-hospital death**, **extended stay**, **non-home discharge** and **admission to ICU**.

## Training Data

Data was downloaded from [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/). MIMIC-IV is a publiclly accessible repository (subject to data usage agreements and mandatory training) containing multimodal secondary care data across healthcare centers in the US.

## Performance and Limitations

Performance across unimodal and multimodal combinations over the four outcomes can be observed in the table below (taken from NHSE report).

![Model Performance](https://github.com/nhsengland/mm-healthfair/blob/main/report/MMHealthFair%20Performance%20Summary.png)

## Acknowledgements
