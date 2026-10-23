import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

DEFAULTS = {
    "pregnancies": 1,
    "glucose": 117.0,
    "blood_pressure": 72.0,
    "skin_thickness": 23.0,
    "insulin": 80.0,
    "bmi": 28.5,
    "dpf": 0.45,
    "age": 35,
}
EXAMPLE_PROFILES = {
    "Example A": DEFAULTS.copy(),
    "Example B": {
        "pregnancies": 5,
        "glucose": 148.0,
        "blood_pressure": 84.0,
        "skin_thickness": 28.0,
        "insulin": 130.0,
        "bmi": 34.0,
        "dpf": 0.62,
        "age": 52,
    },
}


def apply_profile(profile_name):
    st.session_state.update(EXAMPLE_PROFILES[profile_name])
    st.session_state.pop("last_probability", None)
    st.session_state.pop("prediction_error", None)


def reset_form():
    st.session_state.update(DEFAULTS)
    st.session_state.pop("last_probability", None)
    st.session_state.pop("prediction_error", None)



@st.cache_resource
def load_assets():
    model = joblib.load(BASE_DIR / "classifier.joblib")
    with (BASE_DIR / "scaler_params.json").open(encoding="utf-8") as file:
        scaler = json.load(file)
    return model, np.asarray(scaler["mean"]), np.asarray(scaler["scale"])


st.set_page_config(
    page_title="GlucoGuide · Diabetes risk estimate",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink: #172b2a; --muted: #71817d; --teal: #117a70; --mint: #e7f5ef; }
      .stApp { background: #f5f8f6; color: var(--ink); font-family: 'DM Sans', sans-serif; }
      .block-container { max-width: 1120px; padding-top: 3.5rem !important; padding-bottom: 3rem; }
      h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); }
      h1 { line-height: 1.3 !important; padding-top: .12em; overflow: visible; }
      .topline { color: #147c70; font-weight: 700; font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; }
      .hero { background: linear-gradient(115deg, #e5f3ec 0%, #f2f8f4 55%, #edf5f1 100%); border: 1px solid #dcebe3; border-radius: 22px; padding: 1.8rem 2rem; margin: 1rem 0 1.5rem; overflow: visible; }
      .hero p { color: #526661; margin: .45rem 0 0; font-size: 1rem; }
      .eyebrow { color: #73817d; font-size: .8rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; }
      .result { background: #fff; border: 1px solid #e3ebe7; border-radius: 18px; padding: 1.45rem 1.6rem; }
      .result-low { border-left: 5px solid #1e9a70; }
      .result-high { border-left: 5px solid #d9853d; }
      .result-title { color: #64756f; font-size: .82rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
      .result-value { font-family: 'Manrope', sans-serif; color: var(--ink); font-size: 1.55rem; line-height: 1.35; font-weight: 800; margin: .25rem 0; }
      .result-copy { color: #667771; font-size: .94rem; }
      div.stButton > button { background: #117a70; color: white; border: 0; border-radius: 11px; min-height: 3rem; font-weight: 700; }
      div.stButton > button:hover { background: #0b665d; color: white; border: 0; }
      div[data-testid="stNumberInput"] input { border-radius: 10px; }
      .footnote { color: #7a8984; font-size: .8rem; }
      [data-testid="stMetric"] { background: #f7faf8; border: 1px solid #e8efeb; border-radius: 12px; padding: .8rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="topline">✳ &nbsp; GlucoGuide</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
      <div style="color:#147c70;font-size:.78rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.45rem">DIABETES RISK ESTIMATOR</div>
      <h1 style="margin:0;font-size:2.15rem">A clearer starting point for your health.</h1>
      <p>Enter a few health measurements to see a diabetes risk estimate from the project’s trained model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.markdown('<div class="eyebrow">Your measurements</div>', unsafe_allow_html=True)
    st.markdown("### Patient profile")
    st.caption("Use your latest measurements where possible, or start with an example profile.")
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("decision_threshold", 0.50)

    sample_a, sample_b, reset = st.columns([1, 1, 1])
    sample_a.button("Try example A", on_click=apply_profile, args=("Example A",), use_container_width=True)
    sample_b.button("Try example B", on_click=apply_profile, args=("Example B",), use_container_width=True)
    reset.button("Reset", on_click=reset_form, use_container_width=True)

    with st.form("measurement_form"):
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, key="pregnancies", help="Number of times pregnant")
            glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, step=1.0, key="glucose", help="Plasma glucose concentration")
            blood_pressure = st.number_input("Blood pressure (mm Hg)", min_value=0.0, max_value=200.0, step=1.0, key="blood_pressure", help="Diastolic blood pressure")
            skin_thickness = st.number_input("Skin thickness (mm)", min_value=0.0, max_value=100.0, step=1.0, key="skin_thickness", help="Triceps skin fold thickness")
        with col_b:
            insulin = st.number_input("Insulin (µU/mL)", min_value=0.0, max_value=900.0, step=1.0, key="insulin", help="2-hour serum insulin")
            bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1, key="bmi", help="Body mass index")
            dpf = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, step=0.01, key="dpf", help="Family history score used by the dataset")
            age = st.number_input("Age (years)", min_value=1, max_value=120, step=1, key="age")
        submitted = st.form_submit_button("Calculate estimate", type="primary", use_container_width=True)

    with st.expander("Explore the model cutoff"):
        st.slider(
            "Positive result cutoff",
            min_value=0.10,
            max_value=0.90,
            step=0.05,
            key="decision_threshold",
            help="A higher cutoff requires a higher model probability before it labels a result elevated. The default is 50%.",
        )
        st.caption("Changing this cutoff changes the label, not the model probability. The default 50% cutoff matches the classifier’s standard decision rule.")

with right:
    st.markdown('<div class="eyebrow">Your result</div>', unsafe_allow_html=True)
    st.markdown("### Risk overview")
    if submitted:
        try:
            model, scaler_mean, scaler_scale = load_assets()
            values = np.asarray(
                [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
                dtype=float,
            )
            scaled_values = ((values - scaler_mean) / scaler_scale).reshape(1, -1)
            probability = float(model.predict_proba(scaled_values)[0, 1])
            st.session_state["last_probability"] = probability
            st.session_state.pop("prediction_error", None)
        except FileNotFoundError as exc:
            st.session_state["prediction_error"] = f"A required model file is missing: {exc.filename}"
        except Exception as exc:
            st.session_state["prediction_error"] = f"Could not calculate an estimate: {exc}"

    if "prediction_error" in st.session_state:
        st.error(st.session_state["prediction_error"])
    if "last_probability" in st.session_state:
        probability = st.session_state["last_probability"]
        positive = probability >= st.session_state["decision_threshold"]
        title = "Elevated risk estimate" if positive else "Lower risk estimate"
        detail = (
            "The model indicates a higher likelihood based on these measurements. Consider discussing them with a healthcare professional."
            if positive
            else "The model indicates a lower likelihood based on these measurements. Keep up with routine health checkups."
        )
        result_class = "result-high" if positive else "result-low"
        st.markdown(
            f'<div class="result {result_class}"><div class="result-title">Model assessment</div>'
            f'<div class="result-value">{title}</div><div class="result-copy">{detail}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(" ")
        st.metric("Estimated model probability", f"{probability:.0%}")
        st.progress(probability, text="Estimated probability from this model")
        st.caption("This percentage is a model output, not a clinical diagnosis or a personal medical probability.")
    else:
        st.markdown(
            '<div class="result"><div class="result-title">Ready when you are</div>'
            '<div class="result-value">Your estimate will appear here</div>'
            '<div class="result-copy">Complete the measurements and select Calculate estimate to view the result.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(" ")
        st.info("The model uses a Random Forest trained on a diabetes dataset. Results are for educational use only.", icon="ℹ️")

st.markdown("---")
st.markdown(
    '<div class="footnote">For educational use only. This tool does not diagnose, prevent, or treat diabetes. '
    'It is not a substitute for professional medical advice, testing, or care.</div>',
    unsafe_allow_html=True,
)
