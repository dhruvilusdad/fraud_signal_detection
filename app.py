"""Streamlit interface for the saved credit-card fraud classifiers."""

from __future__ import annotations

import math
import hashlib
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
_configured_data_path = os.environ.get("FRAUD_DATA_PATH")
DATA_PATH = Path(_configured_data_path).expanduser() if _configured_data_path else ROOT / "creditcard.csv"
if not DATA_PATH.is_absolute():
    DATA_PATH = ROOT / DATA_PATH
DATA_CANDIDATES = (DATA_PATH, ROOT / "creditcard.csv", ROOT / "data" / "creditcard.csv")
MAX_DATASET_UPLOAD_BYTES = 200 * 1024 * 1024
MODEL_PATHS = {
    "Logistic Regression": ROOT / "logistic_regression_model.pkl",
    "Random Forest": ROOT / "random_forest_model.pkl",
}
FEATURES = ("Time", *(f"V{index}" for index in range(1, 29)), "Amount")
CLASS_LABELS = {0: "Genuine", 1: "Fraud"}

st.set_page_config(
    page_title="Fraud Signal | Transaction Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


THEMES = {
    "Dark": {
        "page": "#0b1220",
        "surface": "#111c2e",
        "surface_alt": "#1b2a40",
        "text": "#f1f5f9",
        "muted": "#b8c4d6",
        "border": "#334155",
        "accent": "#60a5fa",
        "on_accent": "#0b1220",
        "fraud": "#ef4444",
        "genuine": "#10b981",
        "warning": "#f59e0b",
        "fraud_bg": "#3b1920",
        "fraud_text": "#fecaca",
        "success_bg": "#103328",
        "success_text": "#a7f3d0",
        "warning_bg": "#3a2b13",
        "warning_text": "#fde68a",
        "sidebar": "#0e192a",
        "grid": "#334155",
    },
    "Light": {
        "page": "#f7f9fc",
        "surface": "#ffffff",
        "surface_alt": "#f1f5f9",
        "text": "#172b4d",
        "muted": "#526174",
        "border": "#e2e8f0",
        "accent": "#2563eb",
        "on_accent": "#ffffff",
        "fraud": "#ef4444",
        "genuine": "#10b981",
        "warning": "#f59e0b",
        "fraud_bg": "#fef2f2",
        "fraud_text": "#b91c1c",
        "success_bg": "#ecfdf5",
        "success_text": "#047857",
        "warning_bg": "#fffbeb",
        "warning_text": "#92400e",
        "sidebar": "#eff4fb",
        "grid": "#e2e8f0",
    },
}


def _apply_theme(theme_name: str) -> None:
    palette = THEMES[theme_name]
    color_scheme = "dark" if theme_name == "Dark" else "light"
    variables = "\n".join(
        f"    --{key.replace('_', '-')}: {value};"
        for key, value in palette.items()
    )
    st.markdown(
        f"""
        <style>
        :root {{
        {variables}
            color-scheme: {color_scheme};
        }}
        html, body, [data-testid="stAppViewContainer"] {{
            background: var(--page);
            color: var(--text);
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 1440px;
            color: var(--text);
            padding-top: 1.6rem;
        }}
        [data-testid="stMainBlockContainer"] h1,
        [data-testid="stMainBlockContainer"] h2,
        [data-testid="stMainBlockContainer"] h3,
        [data-testid="stMainBlockContainer"] h4,
        [data-testid="stMainBlockContainer"] p,
        [data-testid="stMainBlockContainer"] li,
        [data-testid="stMainBlockContainer"] label,
        [data-testid="stMainBlockContainer"] [data-testid="stWidgetLabel"],
        [data-testid="stMainBlockContainer"] [data-testid="stWidgetLabel"] * {{
            color: var(--text) !important;
        }}
        [data-testid="stMainBlockContainer"] small,
        [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] {{
            color: var(--muted) !important;
        }}
        [data-testid="stMainBlockContainer"] .hero-kicker {{
            color: var(--genuine) !important;
        }}
        [data-testid="stMainBlockContainer"] .hero-copy,
        [data-testid="stMainBlockContainer"] .note-card {{
            color: var(--muted) !important;
        }}
        [data-testid="stSidebar"] {{
            background: var(--sidebar);
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] * {{ color: var(--text); }}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: var(--muted) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{
            padding: .42rem .58rem;
            margin: .12rem 0;
            border: 1px solid transparent;
            border-radius: 9px;
            transition: background-color 120ms ease, border-color 120ms ease;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
            background-color: var(--surface-alt) !important;
            border-color: var(--border) !important;
            box-shadow: inset 3px 0 var(--accent) !important;
        }}
        [data-testid="stMetric"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 18px rgba(0, 10, 25, 0.08);
        }}
        [data-testid="stMetric"] label,
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {{
            color: var(--text) !important;
        }}
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] [role="combobox"],
        [data-testid="stMultiSelect"] [role="combobox"] {{
            color: var(--text) !important;
            background: var(--surface-alt) !important;
            border-color: var(--border) !important;
        }}
        [data-baseweb="popover"], [data-baseweb="menu"],
        [role="listbox"], [role="option"] {{
            color: var(--text) !important;
            background: var(--surface) !important;
        }}
        [role="option"]:hover {{ background: var(--surface-alt) !important; }}
        [data-testid="stButton"] button,
        [data-testid="stFormSubmitButton"] button {{
            border-radius: 9px;
            border-color: var(--accent);
            color: var(--text);
            background: var(--surface-alt);
            transition: border-color 120ms ease, background-color 120ms ease;
        }}
        [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {{
            background: var(--accent);
            border-color: var(--accent);
            color: var(--on-accent) !important;
            font-weight: 650;
        }}
        [data-testid="stButton"] button[kind="primary"] {{
            background: var(--accent);
            border-color: var(--accent);
            color: var(--on-accent) !important;
        }}
        [data-testid="stButton"] button:hover,
        [data-testid="stFormSubmitButton"] button:hover {{
            border-color: var(--genuine);
            background: var(--surface-alt);
        }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }}
        [data-testid="stAlert"] {{
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: 10px;
            background: var(--surface);
            color: var(--text);
        }}
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {{
            color: var(--text) !important;
        }}
        [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {{
            border-left-color: var(--warning);
        }}
        [data-testid="stAlertContentWarning"] {{
            background: var(--warning-bg) !important;
            border-radius: 8px;
        }}
        [data-testid="stAlertContentWarning"] p,
        [data-testid="stAlertContentWarning"] svg {{
            color: var(--warning-text) !important;
        }}
        [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{
            border-left-color: var(--fraud);
        }}
        [data-testid="stAlertContentError"] {{
            background: var(--fraud-bg) !important;
            border-radius: 8px;
        }}
        [data-testid="stAlertContentError"] p,
        [data-testid="stAlertContentError"] svg {{
            color: var(--fraud-text) !important;
        }}
        [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {{
            border-left-color: var(--genuine);
        }}
        [data-testid="stAlertContentSuccess"] {{
            background: var(--success-bg) !important;
            border-radius: 8px;
        }}
        [data-testid="stAlertContentSuccess"] p,
        [data-testid="stAlertContentSuccess"] svg {{
            color: var(--success-text) !important;
        }}
        [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {{
            border-left-color: var(--accent);
        }}
        [data-testid="stExpander"] {{
            border-color: var(--border);
            border-radius: 10px;
            background: var(--surface);
        }}
        .hero {{
            padding: 1.35rem 1.5rem;
            margin: 0 0 1.25rem;
            border: 1px solid var(--border);
            border-left: 5px solid var(--accent);
            border-radius: 14px;
            background: var(--surface);
            box-shadow: 0 6px 18px rgba(0, 10, 25, 0.07);
        }}
        .hero-kicker {{
            margin: 0 0 .4rem;
            color: var(--genuine);
            font-size: .76rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .08em;
        }}
        .hero-title {{
            margin: 0;
            color: var(--text);
            font-size: 1.75rem;
            font-weight: 720;
            line-height: 1.2;
        }}
        .hero-copy {{ margin: .5rem 0 0; color: var(--muted); }}
        .result-card {{
            margin: 1rem 0;
            padding: 1.1rem 1.3rem;
            border: 1px solid var(--border);
            border-left: 5px solid var(--result-accent);
            border-radius: 12px;
            background: var(--result-bg);
        }}
        .result-card h3 {{ margin: 0 0 .25rem; color: var(--result-text); }}
        .result-card p {{ margin: 0; color: var(--result-text); }}
        [data-testid="stMainBlockContainer"] .result-card h3,
        [data-testid="stMainBlockContainer"] .result-card p {{
            color: var(--result-text) !important;
        }}
        .note-card {{
            padding: .85rem 1rem;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--surface);
            color: var(--muted);
        }}
        .stPlotlyChart {{ border-radius: 10px; }}
        @media (max-width: 760px) {{
            [data-testid="stMainBlockContainer"] {{
                padding: 1rem .75rem 2rem;
            }}
            .hero {{ padding: 1rem; }}
            .hero-title {{ font-size: 1.4rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _asset_signature(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path), -1, -1)


def _resolve_data_path() -> Path:
    """Resolve an explicitly configured dataset or either supported local layout."""
    if _configured_data_path:
        return DATA_PATH
    for candidate in dict.fromkeys(DATA_CANDIDATES):
        if candidate.is_file():
            return candidate
    return DATA_PATH


def _stage_uploaded_dataset(uploaded_file: Any) -> Path:
    """Stage a user's real CSV outside the repository for this app process."""
    file_size = int(getattr(uploaded_file, "size", 0))
    if file_size > MAX_DATASET_UPLOAD_BYTES:
        raise ValueError(
            "The dataset exceeds the 200 MB upload limit. Configure FRAUD_DATA_PATH "
            "to a readable mounted file instead."
        )
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
    upload_dir = Path(tempfile.gettempdir()) / "fraud_signal_datasets"
    upload_dir.mkdir(parents=True, exist_ok=True)
    staged_path = upload_dir / f"creditcard-{digest}.csv"
    if not staged_path.is_file():
        uploaded_file.seek(0)
        with staged_path.open("wb") as destination:
            shutil.copyfileobj(uploaded_file, destination)
    return staged_path


def _model_signature() -> tuple[tuple[str, str, int, int], ...]:
    signature = []
    for name, path in MODEL_PATHS.items():
        _, modified, size = _asset_signature(path)
        signature.append((name, str(path), modified, size))
    return tuple(signature)


@st.cache_data(show_spinner="Reading transaction history...")
def load_transactions(path: str, modified: int, size: int) -> pd.DataFrame:
    """Load and validate the real transaction dataset."""
    del modified, size
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("The transaction dataset is empty.")
    required = [*FEATURES, "Class"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    nonnumeric = [
        column for column in required
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if nonnumeric:
        raise ValueError(f"Expected numeric transaction columns: {', '.join(nonnumeric)}")
    if frame[required].isna().any().any():
        raise ValueError("The dataset contains missing values in model fields.")
    classes = set(frame["Class"].unique().tolist())
    if not classes.issubset({0, 1}) or classes != {0, 1}:
        raise ValueError("Class must contain both binary labels: 0 (genuine) and 1 (fraud).")
    return frame


@st.cache_resource(show_spinner="Loading saved classifiers...")
def load_models(
    signature: tuple[tuple[str, str, int, int], ...],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Load existing model files once; never train models at app startup."""
    del signature
    loaded: dict[str, Any] = {}
    errors: dict[str, str] = {}
    version_warnings: dict[str, str] = {}
    for name, path in MODEL_PATHS.items():
        if not path.is_file():
            errors[name] = f"Saved model not found at {path}."
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = joblib.load(path)
            feature_names = getattr(model, "feature_names_in_", None)
            if feature_names is None:
                errors[name] = "This model does not expose its training feature order."
                continue
            if tuple(str(item) for item in feature_names) != FEATURES:
                errors[name] = "The model's saved feature order does not match the dataset contract."
                continue
            if int(getattr(model, "n_features_in_", -1)) != len(FEATURES):
                errors[name] = f"Expected {len(FEATURES)} trained features; model metadata differs."
                continue
            classes = set(np.asarray(getattr(model, "classes_", [])).tolist())
            if classes != {0, 1}:
                errors[name] = "The saved model does not expose both expected classes (0 and 1)."
                continue
            loaded[name] = model
            for warning in caught:
                original = getattr(warning.message, "original_sklearn_version", None)
                current = getattr(warning.message, "current_sklearn_version", None)
                if type(warning.message).__name__ == "InconsistentVersionWarning" and original and current and original != current:
                    version_warnings[name] = f"Saved with scikit-learn {original}; running with {current}."
        except Exception as error:
            errors[name] = f"Could not load this model: {type(error).__name__}: {error}"
    return loaded, errors, version_warnings


@st.cache_data(show_spinner="Scoring saved models on the held-out transactions...")
def evaluate_saved_models(
    data_signature: tuple[str, int, int],
    model_signature: tuple[tuple[str, str, int, int], ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], int, int]:
    """Evaluate unchanged saved models on the recovered, stratified holdout."""
    frame = load_transactions(*data_signature)
    models, model_errors, _ = load_models(model_signature)
    X = frame.loc[:, list(FEATURES)]
    y = frame["Class"].astype("int64")
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        try:
            predictions = np.asarray(model.predict(X_test.loc[:, list(FEATURES)])).astype(int)
            matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
            results[name] = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
                "confusion_matrix": matrix.tolist(),
                "fraud_test_count": int(y_test.sum()),
                "test_count": int(len(y_test)),
            }
        except Exception as error:
            model_errors[name] = f"Evaluation failed: {type(error).__name__}: {error}"
    return results, model_errors, int(len(y_test)), int(y_test.sum())


def _chart_style(theme_name: str) -> dict[str, str]:
    palette = THEMES[theme_name]
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font_color": palette["text"],
        "grid": palette["grid"],
    }


def _style_figure(figure: go.Figure, theme_name: str) -> go.Figure:
    style = _chart_style(theme_name)
    palette = THEMES[theme_name]
    figure.update_layout(
        paper_bgcolor=style["paper_bgcolor"],
        plot_bgcolor=style["plot_bgcolor"],
        font_color=style["font_color"],
        margin=dict(l=12, r=12, t=54, b=12),
        legend_title_text="",
        hoverlabel=dict(
            bgcolor=palette["surface"],
            bordercolor=palette["border"],
            font=dict(color=palette["text"]),
        ),
    )
    figure.update_xaxes(
        gridcolor=style["grid"],
        zerolinecolor=style["grid"],
        tickfont=dict(color=palette["muted"]),
        title_font=dict(color=palette["muted"]),
    )
    figure.update_yaxes(
        gridcolor=style["grid"],
        zerolinecolor=style["grid"],
        tickfont=dict(color=palette["muted"]),
        title_font=dict(color=palette["muted"]),
    )
    return figure


def _hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<section class="hero"><p class="hero-kicker">{kicker}</p>'
        f'<h1 class="hero-title">{title}</h1><p class="hero-copy">{copy}</p></section>',
        unsafe_allow_html=True,
    )


def _fraud_result(is_fraud: bool, score: float | None) -> None:
    if is_fraud:
        title = "Fraud Detected"
        copy = "The selected model classified this transaction as potentially fraudulent."
        bg = "var(--fraud-bg)"
        color = "var(--fraud-text)"
        accent = "var(--fraud)"
        icon = "!"
    else:
        title = "Genuine Transaction"
        copy = "The selected model classified this transaction as genuine. This is not proof that it is safe."
        bg = "var(--success-bg)"
        color = "var(--success-text)"
        accent = "var(--genuine)"
        icon = "✓"
    st.markdown(
        f'<div class="result-card" style="--result-bg:{bg};--result-text:{color};'
        f'--result-accent:{accent}"><h3>{icon} &nbsp;{title}</h3><p>{copy}</p></div>',
        unsafe_allow_html=True,
    )
    if score is not None:
        st.metric("Model-estimated fraud score", f"{score:.2%}")
        st.caption(
            "This is the model's output, not a calibrated real-world risk probability. "
            "It must not be used alone to approve or decline a transaction."
        )


def _predict(model: Any, values: dict[str, float]) -> tuple[bool, float | None]:
    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        raise ValueError(f"Missing model inputs: {', '.join(missing)}")
    array = np.asarray([values[feature] for feature in FEATURES], dtype=np.float64)
    if array.shape != (len(FEATURES),) or not np.isfinite(array).all():
        raise ValueError("Enter a finite numeric value for every transaction feature.")
    transaction = pd.DataFrame([array], columns=list(FEATURES))
    prediction = int(np.asarray(model.predict(transaction)).reshape(-1)[0])
    if prediction not in (0, 1):
        raise ValueError("The selected model returned an unsupported class label.")
    score: float | None = None
    if hasattr(model, "predict_proba"):
        classes = list(np.asarray(model.classes_))
        if 1 in classes:
            probabilities = np.asarray(model.predict_proba(transaction))[0]
            score = float(probabilities[classes.index(1)])
            if not math.isfinite(score):
                score = None
    return prediction == 1, score


def _example_indices(frame: pd.DataFrame, label: int) -> list[int]:
    eligible = np.flatnonzero(frame["Class"].to_numpy() == label)
    if not len(eligible):
        return []
    count = min(5, len(eligible))
    positions = np.linspace(0, len(eligible) - 1, num=count, dtype=int)
    return [int(index) for index in eligible[positions]]


def _render_overview(frame: pd.DataFrame, theme_name: str) -> None:
    _hero(
        "Transaction intelligence · historical data",
        "Fraud Signal",
        "Explore an anonymized credit-card transaction dataset and compare saved classifiers. "
        "These are historical records, not live bank transactions.",
    )
    counts = frame["Class"].value_counts().reindex([0, 1], fill_value=0)
    total = int(len(frame))
    genuine = int(counts.loc[0])
    fraud = int(counts.loc[1])
    fraud_rate = (fraud / total * 100) if total else 0.0
    metrics = st.columns(4)
    metrics[0].metric("Historical transactions", f"{total:,}")
    metrics[1].metric("Genuine records", f"{genuine:,}")
    metrics[2].metric("Fraud records", f"{fraud:,}")
    metrics[3].metric("Fraud share", f"{fraud_rate:.3f}%")

    left, right = st.columns([1, 1.35])
    with left:
        st.subheader("Class balance")
        distribution = pd.DataFrame(
            {"Class": ["Genuine", "Fraud"], "Transactions": [genuine, fraud]}
        )
        figure = px.bar(
            distribution,
            x="Class",
            y="Transactions",
            color="Class",
            color_discrete_map={
                "Genuine": THEMES[theme_name]["genuine"],
                "Fraud": THEMES[theme_name]["fraud"],
            },
            log_y=True,
            text="Transactions",
            title="Count · logarithmic axis",
        )
        figure.update_traces(texttemplate="%{text:,}", textposition="outside")
        _style_figure(figure, theme_name)
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        st.caption("Log scale keeps the minority fraud class visible beside genuine transactions.")

    with right:
        st.subheader("Activity by elapsed hour")
        hourly = frame.assign(ElapsedHour=(frame["Time"] // 3600).astype(int)).groupby("ElapsedHour").agg(
            Transactions=("Class", "size"), Frauds=("Class", "sum")
        ).reset_index()
        hourly["Fraud share"] = hourly["Frauds"] / hourly["Transactions"] * 100
        figure = px.line(
            hourly,
            x="ElapsedHour",
            y="Fraud share",
            markers=True,
            hover_data=["Transactions", "Frauds"],
            title="Fraud share by hour since dataset start",
            color_discrete_sequence=[THEMES[theme_name]["accent"]],
        )
        figure.update_yaxes(title="Fraud share (%)")
        figure.update_xaxes(title="Elapsed hour")
        _style_figure(figure, theme_name)
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Transaction amounts")
    amount_edges = [-0.001, 0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, np.inf]
    amount_labels = ["0", "0–1", "1–5", "5–10", "10–25", "25–50", "50–100", "100–250", "250–500", "500–1k", "1k–2.5k", "2.5k–5k", ">5k"]
    amount_frame = frame.loc[:, ["Amount", "Class"]].copy()
    amount_frame["Amount band"] = pd.cut(
        amount_frame["Amount"], bins=amount_edges, labels=amount_labels, include_lowest=True
    )
    amount_counts = (
        amount_frame.groupby(["Amount band", "Class"], observed=False)
        .size()
        .rename("Transactions")
        .reset_index()
    )
    amount_counts["Type"] = amount_counts["Class"].map(CLASS_LABELS)
    figure = px.bar(
        amount_counts,
        x="Amount band",
        y="Transactions",
        color="Type",
        barmode="group",
        color_discrete_map={
            "Genuine": THEMES[theme_name]["genuine"],
            "Fraud": THEMES[theme_name]["fraud"],
        },
        title="Historical transactions by amount band",
    )
    figure.update_xaxes(title="Amount (dataset currency units)")
    figure.update_yaxes(title="Transactions")
    _style_figure(figure, theme_name)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    st.info(
        f"Fraud represents {fraud_rate:.3f}% of the records. This severe class imbalance means "
        "accuracy alone can hide missed fraud; review recall, precision, F1, and the confusion matrix."
    )
    st.caption("All dashboard counts and charts are computed from the local CSV. No live bank feed is connected.")


def _render_prediction(frame: pd.DataFrame, models: dict[str, Any], errors: dict[str, str]) -> None:
    _hero(
        "Saved classifiers · one transaction at a time",
        "Transaction assessment",
        "Choose a saved model, then use a real historical example or enter every feature explicitly.",
    )
    if not models:
        st.error("No compatible saved model is available. Check the model files and feature-order messages in the sidebar.")
        return
    selected_model_name = st.selectbox("Classifier", list(models.keys()))
    selected_model = models[selected_model_name]
    input_mode = st.radio(
        "Transaction input",
        ["Use a real dataset example", "Enter manually"],
        horizontal=True,
    )
    historical_label: int | None = None
    values: dict[str, float] | None = None
    submitted = False

    if input_mode == "Use a real dataset example":
        class_choice = st.selectbox(
            "Example class",
            options=[0, 1],
            format_func=lambda label: "Genuine transaction" if label == 0 else "Fraud transaction",
        )
        indices = _example_indices(frame, int(class_choice))
        if not indices:
            st.warning("No real example is available for this class in the dataset.")
            return
        example_index = st.selectbox(
            "Historical record",
            options=indices,
            format_func=lambda index: f"Dataset row {index + 1:,} · Amount {float(frame.iloc[index]['Amount']):,.2f}",
        )
        st.caption("Examples are rows from the actual local dataset; no example transaction is synthesized.")
        submitted = st.button("Assess this example", type="primary", use_container_width=True)
        if submitted:
            row = frame.iloc[example_index]
            values = {feature: float(row[feature]) for feature in FEATURES}
            historical_label = int(row["Class"])
    else:
        st.markdown("Enter `Time` and `Amount`, then expand the PCA field section and fill all 28 features. Blank fields are not replaced with zero.")
        basics = st.columns(2)
        time_value = basics[0].number_input(
            "Time", min_value=0.0, value=None, step=1.0, format="%.0f",
            placeholder="Required", help="Seconds elapsed from the start of the source dataset.",
            key="manual_Time",
        )
        amount_value = basics[1].number_input(
            "Amount", min_value=0.0, value=None, step=0.01, format="%.2f",
            placeholder="Required", help="Use the amount scale present in the training dataset.",
            key="manual_Amount",
        )
        entered: dict[str, float | None] = {"Time": time_value, "Amount": amount_value}
        with st.expander("Advanced PCA features · V1–V28", expanded=False):
            st.caption("These anonymized PCA components are required by the saved models.")
            for start in range(1, 29, 4):
                columns = st.columns(4)
                for offset, column in enumerate(columns):
                    feature_index = start + offset
                    if feature_index <= 28:
                        feature = f"V{feature_index}"
                        entered[feature] = column.number_input(
                            feature,
                            value=None,
                            step=0.001,
                            format="%.6f",
                            placeholder="Required",
                            key=f"manual_{feature}",
                        )
        submitted = st.button("Assess transaction", type="primary", use_container_width=True)
        if submitted:
            missing = [feature for feature in FEATURES if entered.get(feature) is None]
            if missing:
                st.error(f"Complete all {len(FEATURES)} fields before prediction. Missing: {', '.join(missing)}")
            else:
                values = {
                    feature: float(value)
                    for feature in FEATURES
                    if (value := entered.get(feature)) is not None
                }

    if submitted and values is not None:
        try:
            is_fraud, score = _predict(selected_model, values)
            _fraud_result(is_fraud, score)
            if historical_label is not None:
                st.caption(f"Historical dataset label for this example: {CLASS_LABELS[historical_label]}.")
        except (TypeError, ValueError) as error:
            st.error(f"Input validation failed: {error}")
        except Exception as error:
            st.error(f"Prediction could not be completed: {type(error).__name__}: {error}")

    st.warning(
        "Educational prototype only. It is not a production banking security system, "
        "and its output must not be used as the sole basis for a real transaction decision."
    )


def _render_model_performance(
    data_signature: tuple[str, int, int],
    model_signature: tuple[tuple[str, str, int, int], ...],
    theme_name: str,
    model_errors: dict[str, str],
) -> None:
    _hero(
        "Reproducible evaluation · saved models only",
        "Model performance",
        "Metrics are calculated on the stratified held-out split; neither model is retrained by this app.",
    )
    try:
        results, evaluation_errors, test_count, fraud_count = evaluate_saved_models(
            data_signature, model_signature
        )
    except Exception as error:
        st.error(f"Could not evaluate the saved classifiers: {type(error).__name__}: {error}")
        return
    all_errors = {**model_errors, **evaluation_errors}
    if all_errors:
        for name, message in all_errors.items():
            st.warning(f"{name}: {message}")
    if not results:
        st.info("No saved model could be evaluated.")
        return
    st.caption(
        f"Held-out records: {test_count:,} · fraud-positive cases: {fraud_count:,}. "
        "Split: 80/20, stratified by Class, random_state=42. The saved scaler statistics "
        "match this training partition; the estimators are loaded unchanged."
    )
    tabs = st.tabs(list(results))
    for tab, (name, result) in zip(tabs, results.items()):
        with tab:
            columns = st.columns(4)
            columns[0].metric("Accuracy", f"{result['accuracy']:.3f}")
            columns[1].metric("Precision · fraud", f"{result['precision']:.3f}")
            columns[2].metric("Recall · fraud", f"{result['recall']:.3f}")
            columns[3].metric("F1 · fraud", f"{result['f1']:.3f}")
            matrix = np.asarray(result["confusion_matrix"], dtype=int)
            figure = px.imshow(
                matrix,
                x=["Predicted genuine", "Predicted fraud"],
                y=["Actual genuine", "Actual fraud"],
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"{name} · confusion matrix",
                aspect="auto",
            )
            _style_figure(figure, theme_name)
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=pd.Index(["Actual genuine", "Actual fraud"], name="Actual"),
                    columns=["Predicted genuine", "Predicted fraud"],
                ),
                use_container_width=True,
            )
    st.info(
        "Fraud (Class = 1) is the positive class. Accuracy is not sufficient for this highly "
        "imbalanced dataset: a model can appear accurate while missing many fraud cases. "
        "Review fraud recall alongside precision and F1."
    )


def _render_data_explorer(frame: pd.DataFrame, theme_name: str) -> None:
    _hero(
        "Historical sample · anonymized PCA features",
        "Data explorer",
        "Filter the real dataset and inspect transaction amount, elapsed time, and class labels.",
    )
    filters = st.columns(3)
    class_filter = filters[0].selectbox("Class filter", ["All", "Genuine", "Fraud"])
    max_amount = float(frame["Amount"].max())
    amount_filter = filters[1].slider(
        "Maximum amount", min_value=0.0, max_value=max_amount, value=max_amount,
        format="%.2f",
    )
    max_time = float(frame["Time"].max())
    time_filter = filters[2].slider(
        "Elapsed time range", min_value=0.0, max_value=max_time,
        value=(float(frame["Time"].min()), max_time), format="%.0f",
    )
    filtered = frame.loc[
        frame["Amount"].le(amount_filter)
        & frame["Time"].between(time_filter[0], time_filter[1])
    ]
    if class_filter != "All":
        label = 1 if class_filter == "Fraud" else 0
        filtered = filtered.loc[filtered["Class"].eq(label)]
    st.caption(f"{len(filtered):,} matching historical records")
    if filtered.empty:
        st.info("No transactions match these filters.")
        return
    chart_data = filtered.sample(n=min(12000, len(filtered)), random_state=42) if len(filtered) > 12000 else filtered
    figure = px.scatter(
        chart_data,
        x="Time",
        y="Amount",
        color=chart_data["Class"].map(CLASS_LABELS),
        color_discrete_map={
            "Genuine": THEMES[theme_name]["genuine"],
            "Fraud": THEMES[theme_name]["fraud"],
        },
        opacity=0.6,
        title="Amount by elapsed time · deterministic chart sample when filtered set is large",
        labels={"color": "Class", "Time": "Elapsed seconds", "Amount": "Amount"},
    )
    _style_figure(figure, theme_name)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    st.caption("The chart samples actual matching rows only when more than 12,000 records match; summary metrics remain based on the full dataset.")


def _render_about(frame: pd.DataFrame, model_errors: dict[str, str]) -> None:
    _hero(
        "Responsible use · model transparency",
        "About this prototype",
        "A local educational interface around two existing saved classifiers and a historical dataset.",
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Dataset")
        st.write(
            f"The local file contains **{len(frame):,} historical transactions** and "
            f"**{int(frame['Class'].sum()):,} fraud labels** ({frame['Class'].mean():.3%}). "
            "Features include elapsed time, 28 anonymized PCA components, amount, and the binary class."
        )
        st.write(
            "The data is strongly imbalanced. Its statistics describe the supplied dataset, "
            "not current bank activity or future transaction rates."
        )
    with right:
        st.subheader("Saved models")
        for name, path in MODEL_PATHS.items():
            status = "Available" if path.is_file() and name not in model_errors else "Unavailable"
            st.write(f"**{name}:** {status}")
        st.write(f"Runtime scikit-learn: `{sklearn.__version__}`")
        st.write("Required serialized-model version: `1.7.2` (see requirements.txt).")
    st.subheader("Limitations and safe use")
    st.markdown(
        "- The model-estimated probability is not a calibrated real-world risk probability.\n"
        "- A `Genuine` prediction is not a guarantee that a transaction is safe.\n"
        "- This is an educational prototype, not a production banking security system.\n"
        "- Validate model behavior, fairness, drift, security, and operational controls before any real-world use.\n"
        "- Pickle/joblib files can execute code when loaded; only load trusted model artifacts."
    )


def main() -> None:
    if "fraud_theme" not in st.session_state:
        st.session_state["fraud_theme"] = "Dark"
    with st.sidebar:
        st.markdown("## ◈ Fraud Signal")
        st.caption("Credit-card transaction intelligence")
        selected_theme = st.radio(
            "Appearance", ["Dark", "Light"], horizontal=True,
            index=0 if st.session_state["fraud_theme"] == "Dark" else 1,
            key="fraud_theme_choice",
        )
        st.session_state["fraud_theme"] = selected_theme
    _apply_theme(selected_theme)

    with st.sidebar:
        page = st.radio(
            "Workspace",
            ["Overview", "Transaction prediction", "Model performance", "Data explorer", "About"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Historical dataset · not a live bank feed")
        st.caption(f"Scikit-learn runtime: {sklearn.__version__}")

    data_path = _resolve_data_path()
    if not data_path.is_file():
        _hero(
            "Dataset setup",
            "Add the transaction dataset",
            "This deployment needs the original credit-card CSV before its analytics and model tools can load.",
        )
        st.warning(
            "The real creditcard.csv is not bundled with this deployment. "
            "It is excluded from GitHub because it exceeds the per-file size limit."
        )
        with st.container(border=True):
            st.subheader("Provide the authorized dataset")
            st.caption(
                "Upload the original CSV. It must contain Time, V1–V28, Amount, and Class; "
                "Class must include both 0 (genuine) and 1 (fraud). No synthetic data is created."
            )
            uploaded_dataset = st.file_uploader(
                "Choose creditcard.csv",
                type=["csv"],
                key="creditcard_dataset_upload",
                help="Upload the original creditcard.csv (maximum 200 MB).",
            )
        if uploaded_dataset is None:
            st.info(
                "Upload the CSV to continue. Locally, place it beside `app.py` or at "
                "`data/creditcard.csv`; for mounted storage, set `FRAUD_DATA_PATH`."
            )
            st.stop()
        try:
            data_path = _stage_uploaded_dataset(uploaded_dataset)
        except (OSError, ValueError) as error:
            st.error(f"Unable to stage the uploaded dataset: {error}")
            st.stop()

    data_signature = _asset_signature(data_path)
    try:
        frame = load_transactions(*data_signature)
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"Unable to load the transaction dataset: {error}")
        st.info(f"Expected local file: `{DATA_PATH}`")
        st.stop()

    model_signature = _model_signature()
    models, model_errors, version_warnings = load_models(model_signature)
    for name, message in model_errors.items():
        st.sidebar.error(f"{name}: {message}")
    for name, message in version_warnings.items():
        st.sidebar.warning(f"{name}: {message} Install the pinned requirements before deployment.")

    if page == "Overview":
        _render_overview(frame, selected_theme)
    elif page == "Transaction prediction":
        _render_prediction(frame, models, model_errors)
    elif page == "Model performance":
        _render_model_performance(data_signature, model_signature, selected_theme, model_errors)
    elif page == "Data explorer":
        _render_data_explorer(frame, selected_theme)
    else:
        _render_about(frame, model_errors)


if __name__ == "__main__":
    main()
