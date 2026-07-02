# Liquefaction Classification APP

A Streamlit application for classifying soil liquefaction using trained machine-learning classifiers.

Developed by: Mohammad Jawed Roshan

## Models included

The app can run the following classifiers:

- KNN
- SVM
- ANN
- DT
- RF
- LightGBM
- XGBoost
- Naive Bayes

The trained models are stored in the `models/` folder as `.joblib` files.

## Input variables

The app uses the following input variables:

| Column name | Description |
|---|---|
| `Critical_Depth_m` | Critical depth, z (m) |
| `sv_eff_kPa` | Effective vertical stress, σ′v (kPa) |
| `rd` | Stress reduction coefficient, rd |
| `amax_g` | Peak horizontal acceleration, amax (g) |
| `Magnitude_for_KMw` | Earthquake magnitude, Mw |
| `CSR` | Cyclic stress ratio |
| `FC_percent` | Fines content, FC (%) |
| `N1_60` | Corrected SPT blow count, (N1)60 |

For batch evaluation, include an actual class column if you want performance plots. The app can detect common target names such as `Liquefied_Binary`, `Liquefaction`, `Class`, `Target`, or `Label`.

## Folder structure

```text
liquefaction-classification-app/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── ANN_search.joblib
│   ├── DT_search.joblib
│   ├── KNN_search.joblib
│   ├── LightGBM_search.joblib
│   ├── Naive_Bayes_search.joblib
│   ├── Random_Forest_search.joblib
│   ├── SVM_search.joblib
│   └── XGBoost_search.joblib
├── assets/
│   └── logos/


## Installation

Use Python 3.10. On Windows, the following commands can be run from Command Prompt or PowerShell:

```bash
py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For macOS or Linux:

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

## Note
- The ANN model requires TensorFlow, so startup may take longer than the other models.

