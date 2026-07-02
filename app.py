from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

MODELS_DIR = APP_DIR / "models"
LOGOS_DIR = APP_DIR / "logos"



class KerasANNWrapper:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        self._keras_model = None

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._keras_model = None

    def _ensure_model(self):
        if self._keras_model is not None:
            return self._keras_model
        try:
            import numpy as np
            from tensorflow import keras
        except Exception as exc:
            raise RuntimeError(
                "ANN model requires TensorFlow. Install it only if you want to use ANN: "
                "pip install tensorflow==2.10.1 protobuf<3.20"
            ) from exc

        json_cfg = getattr(self, "_serialized_model_json", None)
        weights = getattr(self, "_serialized_model_weights", None)
        if json_cfg is None or weights is None:
            raise RuntimeError("ANN model file does not contain serialized Keras architecture/weights.")

        model = keras.models.model_from_json(json_cfg)
        model.set_weights(weights)
        try:
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        except Exception:
            pass
        self._keras_model = model
        return model

    def predict_proba(self, X):
        import numpy as np
        model = self._ensure_model()
        p1 = model.predict(X, verbose=0).reshape(-1)
        p1 = np.clip(p1.astype(float), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)



import types
_models_module = types.ModuleType("models")
_models_module.KerasANNWrapper = KerasANNWrapper
sys.modules.setdefault("models", _models_module)

MODEL_FILES = {
    "KNN": "KNN_search.joblib",
    "SVM": "SVM_search.joblib",
    "ANN": "ANN_search.joblib",
    "DT": "DT_search.joblib",
    "RF": "Random_Forest_search.joblib",
    "LightGBM": "LightGBM_search.joblib",
    "XGBoost": "XGBoost_search.joblib",
    "Naive Bayes": "Naive_Bayes_search.joblib",
}

DEFAULT_FEATURES = [
    "Critical_Depth_m",
    "sv_eff_kPa",
    "rd",
    "amax_g",
    "Magnitude_for_KMw",
    "CSR",
    "FC_percent",
    "N1_60",
]

FEATURE_LABELS = {
    "Critical_Depth_m": "Critical depth, z (m)",
    "sv_eff_kPa": "Effective vertical stress, σ′v (kPa)",
    "rd": "Stress reduction coefficient, rd",
    "amax_g": "Peak horizontal acceleration, amax (g)",
    "Magnitude_for_KMw": "Earthquake magnitude, Mw",
    "CSR": "Cyclic stress ratio, CSR",
    "FC_percent": "Fines content, FC (%)",
    "N1_60": "Corrected SPT blow count, (N1)60",
}

DEFAULT_VALUES = {
    "Critical_Depth_m": 5.0,
    "sv_eff_kPa": 80.0,
    "rd": 0.90,
    "amax_g": 0.25,
    "Magnitude_for_KMw": 7.5,
    "CSR": 0.20,
    "FC_percent": 15.0,
    "N1_60": 12.0,
}

HELP_TEXT = {
    "Critical_Depth_m": "Critical/depth value in meters.",
    "sv_eff_kPa": "Effective vertical stress in kPa.",
    "rd": "Stress reduction coefficient.",
    "amax_g": "Peak ground acceleration normalized by gravity.",
    "Magnitude_for_KMw": "Moment magnitude used for magnitude correction.",
    "CSR": "Cyclic stress ratio.",
    "FC_percent": "Fines content as a percentage.",
    "N1_60": "Energy-corrected SPT blow count.",
}


def page_config() -> None:
    st.set_page_config(
        page_title="Liquefaction Classification",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_css() -> None:
    st.markdown(
        """
        <style>
        .main-title {text-align:center; font-size:2.4rem; font-weight:800; margin-bottom:0.1rem;}
        .subtitle {text-align:center; color:#555; font-size:1.05rem; margin-bottom:1.1rem;}
        .developer {text-align:center; color:#666; font-size:0.95rem; margin-bottom:1.4rem;}
        .result-card {border-radius:16px; padding:18px; border:1px solid #e5e7eb; background:#ffffff; box-shadow:0 1px 4px rgba(0,0,0,0.06);}
        .ok {color:#166534; font-weight:700;}
        .bad {color:#991b1b; font-weight:700;}
        .small-note {font-size:0.9rem; color:#666;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_logos() -> None:
    logos = []
    if LOGOS_DIR.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            logos.extend(sorted(LOGOS_DIR.glob(ext)))
    if not logos:
        return
    cols = st.columns(min(len(logos), 4))
    for col, logo in zip(cols, logos[:4]):
        with col:
            st.image(str(logo), width=130)


def render_header() -> None:
    render_logos()
    st.markdown('<div class="main-title">Liquefaction Classification App</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Machine-learning based classification of liquefaction occurrence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="developer">Models: KNN, SVM, ANN, DT, RF, LightGBM, XGBoost, and Naive Bayes</div>',
        unsafe_allow_html=True,
    )


def unwrap_model_object(obj: Any) -> Tuple[Any, List[str], Any, Dict[str, Any]]:
    """Return estimator, feature columns, scaler, and metadata from a saved object."""
    if isinstance(obj, dict):
        estimator = obj.get("model")
        features = obj.get("feature_cols") or DEFAULT_FEATURES
        scaler = obj.get("scaler")
        metadata = {k: v for k, v in obj.items() if k not in {"model", "scaler"}}
        return estimator, list(features), scaler, metadata
    features = list(getattr(obj, "feature_names_in_", DEFAULT_FEATURES))
    return obj, features, None, {}


@st.cache_resource(show_spinner=False)
def load_models() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    for display_name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if not path.exists():
            errors[display_name] = f"Missing file: {path}"
            continue
        try:
            obj = joblib.load(path)
            estimator, features, scaler, metadata = unwrap_model_object(obj)
            if estimator is None:
                raise ValueError("The saved object does not contain a model/estimator.")
            loaded[display_name] = {
                "estimator": estimator,
                "features": features,
                "scaler": scaler,
                "metadata": metadata,
                "path": str(path),
            }
        except Exception as exc:
            errors[display_name] = f"{type(exc).__name__}: {exc}"
    return loaded, errors


def collect_inputs(feature_names: List[str]) -> pd.DataFrame:
    values = {}
    st.sidebar.header("Input parameters")
    for feature in feature_names:
        default = float(DEFAULT_VALUES.get(feature, 0.0))
        values[feature] = st.sidebar.number_input(
            FEATURE_LABELS.get(feature, feature),
            value=default,
            step=0.01,
            format="%.4f",
            help=HELP_TEXT.get(feature, None),
        )
    return pd.DataFrame([values], columns=feature_names)


def prepare_input(raw: pd.DataFrame, model_info: Dict[str, Any]) -> pd.DataFrame | np.ndarray:
    features = model_info["features"]
    X = raw.reindex(columns=features)
    scaler = model_info.get("scaler")
    if scaler is not None:
        return scaler.transform(X)
    return X


def predict_one(model_name: str, model_info: Dict[str, Any], raw_input: pd.DataFrame) -> Dict[str, Any]:
    estimator = model_info["estimator"]
    X = prepare_input(raw_input, model_info)

    pred = estimator.predict(X)
    pred_value = int(np.asarray(pred).ravel()[0])

    probability = None
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X)
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                probability = float(proba[0, 1])
            elif proba.ndim == 1:
                probability = float(proba[0])
        except Exception:
            probability = None

    label = "Liquefied" if pred_value == 1 else "Non-liquefied"
    return {
        "Model": model_name,
        "Class": label,
        "Encoded output": pred_value,
        "Probability of liquefaction": probability,
    }


def render_results(results: List[Dict[str, Any]]) -> None:
    table = pd.DataFrame(results)
    if "Probability of liquefaction" in table.columns:
        table["Probability of liquefaction"] = table["Probability of liquefaction"].apply(
            lambda x: "Not available" if pd.isna(x) else f"{x:.3f}"
        )
    st.subheader("Classification results")
    st.dataframe(table, use_container_width=True, hide_index=True)

    liquefied_votes = sum(1 for r in results if r["Encoded output"] == 1)
    total = len(results)
    majority = "Liquefied" if liquefied_votes >= (total / 2) else "Non-liquefied"
    cls = "bad" if majority == "Liquefied" else "ok"
    st.markdown(
        f"""
        <div class="result-card">
            <h3>Majority vote: <span class="{cls}">{majority}</span></h3>
            <p>{liquefied_votes} out of {total} loaded models classified the case as liquefied.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    page_config()
    apply_css()
    render_header()

    loaded, errors = load_models()

    with st.expander("Model loading status", expanded=not bool(loaded)):
        if loaded:
            st.success(f"Loaded {len(loaded)} model(s): {', '.join(loaded.keys())}")
        if errors:
            st.warning("Some models could not be loaded. Check Python/package versions if needed.")
            for name, err in errors.items():
                st.write(f"**{name}:** {err}")
            st.caption("Recommended core environment: Python 3.10, scikit-learn==1.2.2, pandas==1.5.3, numpy==1.23.5.")

    if not loaded:
        st.error("No trained classifiers could be loaded. Please check that the .joblib files are inside the models folder and install the compatible requirements.")
        st.stop()

    reference_features = next(iter(loaded.values()))["features"]
    raw_input = collect_inputs(reference_features)

    st.subheader("Input data")
    st.dataframe(raw_input, use_container_width=True, hide_index=True)

    st.sidebar.header("Model selection")
    selected_models = st.sidebar.multiselect(
        "Select classifiers",
        options=list(MODEL_FILES.keys()),
        default=[name for name in MODEL_FILES.keys() if name in loaded],
    )
    selected_models = [m for m in selected_models if m in loaded]

    if st.button("Classify liquefaction", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one loaded classifier.")
            st.stop()
        results = []
        runtime_errors = []
        for name in selected_models:
            try:
                results.append(predict_one(name, loaded[name], raw_input))
            except Exception as exc:
                runtime_errors.append((name, f"{type(exc).__name__}: {exc}"))
        if results:
            render_results(results)
        if runtime_errors:
            st.error("Some selected models failed during prediction:")
            for name, err in runtime_errors:
                st.write(f"**{name}:** {err}")

    with st.expander("About this app"):
        st.write(
            "This Streamlit app uses trained machine-learning classifiers to predict binary liquefaction occurrence. "
            "The model names are shown using standard abbreviations: KNN, SVM, ANN, DT, RF, LightGBM, XGBoost, and Naive Bayes."
        )
        st.write("Place logos in the `logos` folder if you want them displayed at the top of the app.")


if __name__ == "__main__":
    main()
