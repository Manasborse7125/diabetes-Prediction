# GlucoGuide — Diabetes Risk Estimator

A simple Streamlit frontend for the diabetes classification model in this project. Enter the eight measurements, then select **Calculate estimate** to see the model's output.

## Run locally

From the project folder, create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Streamlit will print a local URL to open in your browser.

On Windows, activate the environment with `.venv\Scripts\activate` instead of the `source` command.

## What’s included

- `app.py` — Streamlit interface, input form, example profiles, adjustable classification cutoff, and results.
- `classifier.joblib` — trained Random Forest classifier.
- `scaler_params.json` — feature scaling values used during model training. The app applies these before prediction.
- `requirements.txt` — Python dependencies, including the scikit-learn version used to save the classifier.
- `Diabetes_Classification.ipynb` and `diabetes.csv` — notebook and dataset used in the project.

The app needs `app.py`, `classifier.joblib`, and `scaler_params.json` in the same folder. The notebook and CSV are not needed to run the frontend.

## Important

This project is for educational use only. Its output is a machine learning estimate, not a diagnosis or a substitute for professional medical advice, testing, or care.
