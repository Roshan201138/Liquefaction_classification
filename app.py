from __future__ import annotations

import sys
import types
import io
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
LOGOS_DIR = APP_DIR / "logos"
ASSETS_LOGOS_DIR = APP_DIR / "assets" / "logos"



def cache_resource_compatible(func=None, **kwargs):
    """Use st.cache_resource when available; fall back for older Streamlit."""
    if hasattr(st, "cache_resource"):
        return st.cache_resource(**kwargs)(func) if func is not None else st.cache_resource(**kwargs)
    if hasattr(st, "cache"):
        return st.cache(allow_output_mutation=True, show_spinner=kwargs.get("show_spinner", True))(func) if func is not None else st.cache(allow_output_mutation=True, show_spinner=kwargs.get("show_spinner", True))
    return func if func is not None else (lambda f: f)


def dataframe_compatible(df: pd.DataFrame, height: Optional[int] = None) -> None:
    try:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)
    except TypeError:
        try:
            st.dataframe(df, use_container_width=True, height=height)
        except TypeError:
            st.dataframe(df)


def image_compatible(path: str, width: int = 190) -> bool:
    """Display local logos robustly.

    Streamlit can show broken icons on some Windows/browser combinations when a
    local path is passed directly. This function reads the image bytes and, if
    needed, falls back to an embedded base64 <img> tag. It returns True only
    when a display method was successfully emitted.
    """
    try:
        p = Path(path)
        if not p.exists() or not p.is_file() or p.stat().st_size == 0:
            return False
        ext = p.suffix.lower()
        raw = p.read_bytes()
        if ext == ".svg":
            encoded = base64.b64encode(raw).decode("utf-8")
            st.markdown(
                f'<img src="data:image/svg+xml;base64,{encoded}" style="max-width:{width}px; max-height:125px; display:block; margin:auto; object-fit:contain;"/>',
                unsafe_allow_html=True,
            )
            return True
        try:
            st.image(raw, width=width)
            return True
        except Exception:
            mime = "jpeg" if ext in {".jpg", ".jpeg"} else ext.replace(".", "")
            encoded = base64.b64encode(raw).decode("utf-8")
            st.markdown(
                f'<img src="data:image/{mime};base64,{encoded}" style="max-width:{width}px; max-height:125px; display:block; margin:auto; object-fit:contain;"/>',
                unsafe_allow_html=True,
            )
            return True
    except Exception:
        return False


def download_button_compatible(label: str, data, file_name: str, mime: str) -> None:
    try:
        st.download_button(label, data=data, file_name=file_name, mime=mime, use_container_width=True)
    except TypeError:
        st.download_button(label, data=data, file_name=file_name, mime=mime)


def button_compatible(label: str) -> bool:
    try:
        return st.button(label, type="primary", use_container_width=True)
    except TypeError:
        try:
            return st.button(label, type="primary")
        except TypeError:
            return st.button(label)



try:
    numeric_module = types.ModuleType("pandas.core.indexes.numeric")
    numeric_module.Int64Index = pd.Index
    numeric_module.UInt64Index = pd.Index
    numeric_module.Float64Index = pd.Index
    sys.modules.setdefault("pandas.core.indexes.numeric", numeric_module)
except Exception:
    pass


class KerasANNWrapper:
    """Compatibility class for ANN models saved as models.KerasANNWrapper."""

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
            from tensorflow import keras
        except Exception as exc:
            raise RuntimeError(
                "ANN requires TensorFlow. Install with: pip install -r requirements_all_models.txt"
            ) from exc

        json_cfg = getattr(self, "_serialized_model_json", None)
        weights = getattr(self, "_serialized_model_weights", None)
        if json_cfg is None or weights is None:
            raise RuntimeError("ANN file does not contain serialized Keras architecture/weights.")

        model = keras.models.model_from_json(json_cfg)
        model.set_weights(weights)
        try:
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        except Exception:
            pass
        self._keras_model = model
        return model

    def predict_proba(self, X):
        model = self._ensure_model()
        p1 = np.asarray(model.predict(X, verbose=0)).reshape(-1).astype(float)
        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        classes = getattr(self, "classes_", np.array([0, 1]))
        encoded = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        try:
            return np.asarray(classes)[encoded]
        except Exception:
            return encoded


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
    "Critical_Depth_m", "sv_eff_kPa", "rd", "amax_g",
    "Magnitude_for_KMw", "CSR", "FC_percent", "N1_60",
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
    "Critical_Depth_m": 5.0, "sv_eff_kPa": 80.0, "rd": 0.90, "amax_g": 0.25,
    "Magnitude_for_KMw": 7.5, "CSR": 0.20, "FC_percent": 15.0, "N1_60": 12.0,
}



FALLBACK_TRAINING_RANGES = {
    "Critical_Depth_m": (1.0524, 14.5),
    "sv_eff_kPa": (10.37810269, 182.7178544),
    "rd": (0.491374367, 0.999064073),
    "amax_g": (0.051, 0.7),
    "Magnitude_for_KMw": (5.9, 8.3),
    "CSR": (0.047613915, 0.634452466),
    "FC_percent": (0.0, 91.0),
    "N1_60": (2.190479236, 65.46861626),
}

TARGET_CANDIDATES = [
    "Liquefaction", "liquefaction", "Class", "class", "Target", "target",
    "Label", "label", "y", "Y", "Liquefied", "liquefied",
    "Liquefied_Binary", "liquefied_binary",
]

DEVELOPER_TEXT = "Developed by: Mohammad Jawed Roshan, António Gomes Correia, Ionut Dragos Moldovan, Miguel Azenha"



def page_config() -> None:
    st.set_page_config(page_title="Liquefaction Classification APP", page_icon="🌍", layout="wide")


def apply_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"], .stMarkdown, .stText, .stButton, .stSelectbox, .stMultiSelect,
        .stNumberInput, .stDataFrame, .stFileUploader, .stDownloadButton {
            font-family: "Times New Roman", Times, serif !important;
        }
        .main-title {
            text-align:center;
            font-size:2.45rem;
            font-weight:800;
            margin-top:0.1rem;
            margin-bottom:0.20rem;
            font-family:"Times New Roman", Times, serif !important;
        }
        .developer {
            text-align:center;
            color:#444;
            font-size:1.05rem;
            margin-bottom:1.15rem;
            font-family:"Times New Roman", Times, serif !important;
        }
        .logo-placeholder {
            min-height:105px;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
            border:1px solid rgba(23,54,93,0.35);
            border-radius:12px;
            background:#f8fafc;
            color:#17365d;
            font-weight:700;
            padding:8px;
        }
        .range-note {
            color:#555;
            font-size:0.92rem;
            margin-top:-0.45rem;
            margin-bottom:0.7rem;
        }
        .section-card {
            border-radius:16px;
            padding:18px;
            border:1px solid #e5e7eb;
            background:white;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);
            margin-bottom:1rem;
        }
        .ok {color:#166534;font-weight:800;}
        .bad {color:#991b1b;font-weight:800;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_logo_paths() -> List[Path]:
    """Find logo files robustly.

    Supported locations:
    - assets/logos/
    - logos/
    - logo/
    - app folder root

    The search is recursive inside the logo folders, so logos placed in subfolders are also found.
    """
    logo_dirs = [APP_DIR / "assets" / "logos", APP_DIR / "logos", APP_DIR / "logo", APP_DIR]
    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
    found: List[Path] = []
    for folder in logo_dirs:
        if folder.exists():
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix.lower() in valid_ext:
                    # Avoid accidental app-generated plot/template files.
                    if any(part.lower() in {"venv", "__pycache__", "models"} for part in p.parts):
                        continue
                    found.append(p)
    # Prefer files with logo/university/uminho/isise/lab in their name.
    def priority(p: Path):
        name = p.name.lower()
        key_terms = ["logo", "uminho", "isise", "lab", "university", "civil"]
        return (0 if any(k in name for k in key_terms) else 1, name)
    unique = []
    seen = set()
    for p in sorted(found, key=priority):
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique[:4]


def render_logo_or_placeholder(col, logo: Optional[Path], text: str) -> None:
    with col:
        shown = False
        if logo is not None and logo.exists():
            st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
            shown = image_compatible(str(logo), width=190)
            st.markdown('</div>', unsafe_allow_html=True)
        if not shown:
            
            
            st.markdown(f'<div class="logo-placeholder">{text}</div>', unsafe_allow_html=True)


def render_header() -> None:
    logos = get_logo_paths()
    logo_row = st.columns(4)
    for i, col in enumerate(logo_row):
        logo = logos[i] if i < len(logos) else None
        render_logo_or_placeholder(col, logo, f"Logo {i+1}")
    
    st.markdown('<div class="main-title">Liquefaction Classification APP</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="developer">{DEVELOPER_TEXT}</div>', unsafe_allow_html=True)
    st.markdown(
        '<hr style="border: 1.2px solid #17365d; margin-top: 0.6rem; margin-bottom: 1.5rem;">',
        unsafe_allow_html=True,
    )




def unwrap_model_object(obj: Any) -> Tuple[Any, List[str], Any, Dict[str, Any]]:
    if isinstance(obj, dict):
        estimator = obj.get("model")
        features = obj.get("feature_cols") or DEFAULT_FEATURES
        scaler = obj.get("scaler")
        metadata = {k: v for k, v in obj.items() if k not in {"model", "scaler"}}
        return estimator, list(features), scaler, metadata
    features = list(getattr(obj, "feature_names_in_", DEFAULT_FEATURES))
    return obj, features, None, {}


@cache_resource_compatible(show_spinner=False)
def load_models() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    for display_name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if not path.exists():
            errors[display_name] = f"Missing file: {path.name} in models folder"
            continue
        try:
            obj = joblib.load(path)
            estimator, features, scaler, metadata = unwrap_model_object(obj)
            if estimator is None:
                raise ValueError("Saved object does not contain a model estimator.")
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


def normalize_binary_label(value: Any) -> int:
    if pd.isna(value):
        return 0
    try:
        return int(round(float(value)))
    except Exception:
        txt = str(value).strip().lower()
        if txt in {"1", "yes", "true", "liquefied", "liquefaction", "l"}:
            return 1
        return 0


def prepare_input(raw: pd.DataFrame, model_info: Dict[str, Any]):
    features = model_info["features"]
    X = raw.reindex(columns=features)
    scaler = model_info.get("scaler")
    if scaler is not None:
        return scaler.transform(X)
    return X


def get_probability_or_score(estimator: Any, X: Any) -> Tuple[Optional[np.ndarray], str]:
    if hasattr(estimator, "predict_proba"):
        try:
            proba = np.asarray(estimator.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1].astype(float), "Probability"
            if proba.ndim == 1:
                return proba.astype(float), "Probability"
        except Exception:
            pass
    if hasattr(estimator, "decision_function"):
        try:
            score = np.asarray(estimator.decision_function(X)).reshape(-1).astype(float)
            return score, "Decision score"
        except Exception:
            pass
    return None, "Not available"


def predict_dataframe(model_name: str, model_info: Dict[str, Any], raw_input: pd.DataFrame) -> pd.DataFrame:
    estimator = model_info["estimator"]
    X = prepare_input(raw_input, model_info)
    pred_raw = np.asarray(estimator.predict(X)).ravel()
    pred = np.array([normalize_binary_label(v) for v in pred_raw], dtype=int)
    return pd.DataFrame({
        "Model": model_name,
        "Predicted class": np.where(pred == 1, "Liquefied", "Non-liquefied"),
        "Encoded prediction": pred,
    })


def predict_one(model_name: str, model_info: Dict[str, Any], raw_input: pd.DataFrame) -> Dict[str, Any]:
    df = predict_dataframe(model_name, model_info, raw_input)
    row = df.iloc[0].to_dict()
    return {
        "Model": model_name,
        "Class": row["Predicted class"],
        "Encoded output": int(row["Encoded prediction"]),
    }


def get_training_ranges_from_models(loaded: Dict[str, Dict[str, Any]], feature_names: List[str]) -> Dict[str, Tuple[float, float]]:
   
    return {f: FALLBACK_TRAINING_RANGES.get(f, (float("nan"), float("nan"))) for f in feature_names}


def format_range(rng: Tuple[float, float]) -> str:
    mn, mx = rng
    if np.isnan(mn) or np.isnan(mx):
        return "Training range: not available"
    return f"Training range: {mn:.4g} to {mx:.4g}"


def make_template_dataframe(feature_names: List[str], include_target: bool = True) -> pd.DataFrame:
    row = {f: DEFAULT_VALUES.get(f, 0.0) for f in feature_names}
    df = pd.DataFrame([row])
    if include_target:
        df["Liquefied_Binary"] = 0
    return df


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()


def figure_to_png_bytes(fig) -> bytes:
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=300, bbox_inches="tight")
    output.seek(0)
    return output.getvalue()


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: Optional[np.ndarray]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        try:
            out["AUC"] = roc_auc_score(y_true, score)
        except Exception:
            out["AUC"] = np.nan
    else:
        out["AUC"] = np.nan
    return out


def sort_metrics_by_performance(metrics_df: pd.DataFrame, sort_metric: str = "Accuracy") -> pd.DataFrame:
    df = metrics_df.copy()
    if sort_metric not in df.columns:
        sort_metric = "Accuracy" if "Accuracy" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    return df.sort_values(by=sort_metric, ascending=False, na_position="last").reset_index(drop=True)


def make_performance_comparison_plot(metrics_df: pd.DataFrame, sort_metric: str = "Accuracy"):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})
    metrics = [c for c in ["Accuracy", "Precision", "Recall", "F1-score", "AUC"] if c in metrics_df.columns]
    ordered = sort_metrics_by_performance(metrics_df, sort_metric)
    plot_df = ordered.set_index("Model")[metrics].copy()
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_xlabel("")
    ax.set_title("Performance Comparison of Selected ML Models", pad=14)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    ax.text(0.5, -0.20, "Models ordered by " + sort_metric, ha="center", va="top", transform=ax.transAxes)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, title="Metrics")
    fig.tight_layout(rect=[0, 0.10, 0.84, 1])
    return fig



def collect_inputs_main(feature_names: List[str], training_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    st.subheader("Input variables")
    values: Dict[str, float] = {}
    cols = st.columns(4)
    for i, feature in enumerate(feature_names):
        with cols[i % 4]:
            rng = training_ranges.get(feature, FALLBACK_TRAINING_RANGES.get(feature, (float("nan"), float("nan"))))
            values[feature] = st.number_input(
                FEATURE_LABELS.get(feature, feature),
                value=float(DEFAULT_VALUES.get(feature, 0.0)),
                step=0.01,
                format="%.4f",
                key=f"single_{feature}",
            )
            st.markdown(f'<div class="range-note">{format_range(rng)}</div>', unsafe_allow_html=True)
    return pd.DataFrame([values], columns=feature_names)


def render_single_prediction(loaded: Dict[str, Dict[str, Any]], selected_models: List[str]) -> None:
    reference_features = next(iter(loaded.values()))["features"]
    training_ranges = get_training_ranges_from_models(loaded, reference_features)
    raw_input = collect_inputs_main(reference_features, training_ranges)
    st.markdown("---")

    if button_compatible("Classify liquefaction"):
        if not selected_models:
            st.error("Please select at least one loaded classifier from the sidebar.")
            return
        results, runtime_errors = [], []
        for name in selected_models:
            try:
                results.append(predict_one(name, loaded[name], raw_input))
            except Exception as exc:
                runtime_errors.append((name, f"{type(exc).__name__}: {exc}"))

        if results:
            table = pd.DataFrame(results)
            st.subheader("Classification results")
            dataframe_compatible(table)
            download_button_compatible("Download single prediction table as CSV", table.to_csv(index=False).encode("utf-8"), "single_prediction_results.csv", "text/csv")

            liquefied_votes = sum(1 for r in results if r["Encoded output"] == 1)
            total = len(results)
            majority = "Liquefied" if liquefied_votes >= total / 2 else "Non-liquefied"
            cls = "bad" if majority == "Liquefied" else "ok"
            st.markdown(
                f'<div class="section-card"><h3>Majority vote: <span class="{cls}">{majority}</span></h3>'
                f'<p>{liquefied_votes} out of {total} selected models classified the case as liquefied.</p></div>',
                unsafe_allow_html=True,
            )

        if runtime_errors:
            st.error("Some selected models failed during prediction:")
            for name, err in runtime_errors:
                st.write(f"**{name}:** {err}")



def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def find_default_target_column(df: pd.DataFrame) -> Optional[str]:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    return None


def confusion_matrix_dataframe(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Actual non-liquefied (0)", "Actual liquefied (1)"],
        columns=["Predicted non-liquefied (0)", "Predicted liquefied (1)"],
    )


def make_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    plt.rcParams["font.family"] = "Times New Roman"
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # For labels=[0,1]: TN=cm[0,0], FP=cm[0,1], FN=cm[1,0], TP=cm[1,1]
    cell_names = np.array([["TN", "FP"], ["FN", "TP"]])
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix - {model_name}", pad=12)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-liquefied", "Liquefied"])
    ax.set_yticklabels(["Non-liquefied", "Liquefied"])
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, f"{cell_names[i, j]}\n{cm[i, j]}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def make_roc_plot(y_true: np.ndarray, score: Optional[np.ndarray], model_name: str, score_name: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    if score is None or len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "ROC curve is not available\n(actual classes must include both 0 and 1, and model must provide probability/score)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        return fig
    fpr, tpr, _ = roc_curve(y_true, score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set_title(f"ROC Curve - {model_name}", pad=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def make_actual_predicted_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    idx = np.arange(1, len(y_true) + 1)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(idx, y_true, marker="o", linestyle="-", label="Actual")
    ax.plot(idx, y_pred, marker="x", linestyle="--", label="Predicted")
    ax.set_title(f"Actual vs Predicted Comparison - {model_name}", pad=12)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Class (0 = Non-liquefied, 1 = Liquefied)")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.15, 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def render_batch_prediction(loaded: Dict[str, Dict[str, Any]], selected_models: List[str]) -> None:
    st.subheader("Batch prediction")
    st.write("Upload a CSV or Excel file containing the required input variables.")
    reference_features = next(iter(loaded.values()))["features"]
    training_ranges = get_training_ranges_from_models(loaded, reference_features)

    range_table = pd.DataFrame({
        "Required column name": reference_features,
        "Input variable": [FEATURE_LABELS.get(f, f) for f in reference_features],
        "Training min": [training_ranges[f][0] for f in reference_features],
        "Training max": [training_ranges[f][1] for f in reference_features],
    })
    with st.expander("Required input columns and training ranges", expanded=True):
        dataframe_compatible(range_table)
        rdl1, rdl2 = st.columns(2)
        with rdl1:
            download_button_compatible("Download training ranges as CSV", range_table.to_csv(index=False).encode("utf-8"), "training_ranges.csv", "text/csv")
        with rdl2:
            download_button_compatible("Download training ranges as Excel", dataframe_to_excel_bytes(range_table, sheet_name="Training ranges"), "training_ranges.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    template_df = make_template_dataframe(reference_features, include_target=True)
    ctmp1, ctmp2 = st.columns(2)
    with ctmp1:
        download_button_compatible(
            "Download Excel template",
            dataframe_to_excel_bytes(template_df, sheet_name="Template"),
            "liquefaction_batch_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with ctmp2:
        download_button_compatible(
            "Download CSV template",
            template_df.to_csv(index=False).encode("utf-8"),
            "liquefaction_batch_template.csv",
            "text/csv",
        )

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        return

    try:
        data = read_uploaded_table(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {type(exc).__name__}: {exc}")
        return

    st.subheader("Uploaded data preview")
    dataframe_compatible(data.head(20))
    ud1, ud2 = st.columns(2)
    with ud1:
        download_button_compatible("Download uploaded data as CSV", data.to_csv(index=False).encode("utf-8"), "uploaded_data.csv", "text/csv")
    with ud2:
        download_button_compatible("Download uploaded data as Excel", dataframe_to_excel_bytes(data, sheet_name="Uploaded data"), "uploaded_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    missing = [c for c in reference_features if c not in data.columns]
    if missing:
        st.error("The uploaded file is missing these required columns: " + ", ".join(missing))
        return

    default_target = find_default_target_column(data)
    target_options = ["No actual/target column"] + list(data.columns)
    default_index = target_options.index(default_target) if default_target in target_options else 0
    target_col = st.selectbox("Actual/target column for evaluation plots", target_options, index=default_index)

    if not selected_models:
        st.error("Please select at least one loaded classifier from the sidebar.")
        return

    if not button_compatible("Run batch prediction"):
        return

    feature_data = data[reference_features].copy()
    all_predictions: List[pd.DataFrame] = []
    per_model: Dict[str, Dict[str, Any]] = {}
    errors: List[Tuple[str, str]] = []

    for name in selected_models:
        try:
            pred_df = predict_dataframe(name, loaded[name], feature_data)
            pred_df.insert(0, "Sample", np.arange(1, len(pred_df) + 1))
            all_predictions.append(pred_df)

            estimator = loaded[name]["estimator"]
            X = prepare_input(feature_data, loaded[name])
            score, score_name = get_probability_or_score(estimator, X)
            per_model[name] = {
                "prediction": pred_df["Encoded prediction"].to_numpy(dtype=int),
                "score": score,
                "score_name": score_name,
                "table": pred_df,
            }
        except Exception as exc:
            errors.append((name, f"{type(exc).__name__}: {exc}"))

    if all_predictions:
        result_table = pd.concat(all_predictions, ignore_index=True)
        st.subheader("Batch prediction results")
        dataframe_compatible(result_table, height=350)
        csv = result_table.to_csv(index=False).encode("utf-8")
        d1, d2 = st.columns(2)
        with d1:
            download_button_compatible("Download prediction results as CSV", csv, "liquefaction_batch_predictions.csv", "text/csv")
        with d2:
            download_button_compatible("Download prediction results as Excel", dataframe_to_excel_bytes(result_table, sheet_name="Predictions"), "liquefaction_batch_predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if errors:
        st.error("Some selected models failed during batch prediction:")
        for name, err in errors:
            st.write(f"**{name}:** {err}")

    if target_col == "No actual/target column" or not per_model:
        st.info("Select an actual/target column to display confusion matrix, ROC curve, and actual-vs-predicted plots.")
        return

    y_true = np.array([normalize_binary_label(v) for v in data[target_col]], dtype=int)

    metrics_rows = []
    for name, info in per_model.items():
        row = {"Model": name}
        row.update(compute_classification_metrics(y_true, info["prediction"], info["score"]))
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        st.subheader("Performance comparison of selected ML models")
        sortable_metrics = [c for c in ["Accuracy", "F1-score", "AUC", "Precision", "Recall"] if c in metrics_df.columns]
        sort_metric = st.selectbox("Order comparison by", sortable_metrics, index=0) if len(sortable_metrics) > 1 else sortable_metrics[0]
        ordered_metrics = sort_metrics_by_performance(metrics_df, sort_metric)
        display_metrics = ordered_metrics.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]:
            if col in display_metrics.columns:
                display_metrics[col] = display_metrics[col].apply(lambda x: "—" if pd.isna(x) else f"{float(x):.4f}")
        dataframe_compatible(display_metrics)
        dm1, dm2 = st.columns(2)
        with dm1:
            download_button_compatible("Download performance table as CSV", ordered_metrics.to_csv(index=False).encode("utf-8"), "model_performance_comparison.csv", "text/csv")
        with dm2:
            download_button_compatible("Download performance table as Excel", dataframe_to_excel_bytes(ordered_metrics, sheet_name="Performance"), "model_performance_comparison.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if len(metrics_df) >= 2:
            perf_fig = make_performance_comparison_plot(metrics_df, sort_metric)
            st.pyplot(perf_fig)
            download_button_compatible("Download performance comparison plot", figure_to_png_bytes(perf_fig), "performance_comparison.png", "image/png")

    st.subheader("Batch evaluation plots")

    model_names = list(per_model.keys())
    if hasattr(st, "tabs"):
        tabs = st.tabs(model_names)
        containers = zip(tabs, model_names)
    else:
        chosen = st.selectbox("Select model for plots", model_names)
        containers = [(st.container(), chosen)]

    for container, name in containers:
        with container:
            y_pred = per_model[name]["prediction"]
            score = per_model[name]["score"]
            score_name = per_model[name]["score_name"]
            cm_fig = make_confusion_matrix_plot(y_true, y_pred, name)
            cm_table = confusion_matrix_dataframe(y_true, y_pred)
            roc_fig = make_roc_plot(y_true, score, name, score_name)
            ap_fig = make_actual_predicted_plot(y_true, y_pred, name)
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(cm_fig)
                download_button_compatible(f"Download {name} confusion matrix plot", figure_to_png_bytes(cm_fig), f"{name}_confusion_matrix.png", "image/png")
                download_button_compatible(f"Download {name} confusion matrix table", cm_table.to_csv().encode("utf-8"), f"{name}_confusion_matrix.csv", "text/csv")
            with c2:
                st.pyplot(roc_fig)
                download_button_compatible(f"Download {name} ROC curve", figure_to_png_bytes(roc_fig), f"{name}_roc_curve.png", "image/png")
            st.pyplot(ap_fig)
            download_button_compatible(f"Download {name} actual vs predicted plot", figure_to_png_bytes(ap_fig), f"{name}_actual_vs_predicted.png", "image/png")



def main() -> None:
    page_config()
    apply_css()
    render_header()

    loaded, errors = load_models()
    if errors:
        with st.expander("Model loading status", expanded=False):
            if loaded:
                st.success(f"Loaded {len(loaded)} model(s): {', '.join(loaded.keys())}")
            for name, err in errors.items():
                st.write(f"**{name}:** {err}")

    if not loaded:
        st.error("No trained classifiers could be loaded. Check the models folder and install the compatible environment.")
        st.stop()

    st.sidebar.header("Prediction type")
    page = st.sidebar.radio("Select page", ["Single prediction", "Batch prediction"])

    st.sidebar.header("Model selection")
    loaded_model_names = [name for name in MODEL_FILES.keys() if name in loaded]
    selected_models = st.sidebar.multiselect(
        "Select classifiers",
        options=loaded_model_names,
        default=loaded_model_names,
    )

    if page == "Single prediction":
        render_single_prediction(loaded, selected_models)
    else:
        render_batch_prediction(loaded, selected_models)


if __name__ == "__main__":
    main()
