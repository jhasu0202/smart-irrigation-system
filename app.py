from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "smart_irrigation_system.csv"
TARGET = "Irrigation_Need"

st.set_page_config(
    page_title="Smart Irrigation | Decision Support",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f4faf5 0%, #ffffff 40%); }
      .block-container { padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1400px; }
      .hero { padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#14532d,#16a34a);
              color: white; margin-bottom: 1rem; }
      .hero h1 { color: white; margin: 0; font-size: 2.2rem; }
      .hero p { color: #e7fbe9; margin: .45rem 0 0 0; font-size: 1.05rem; }
      div[data-testid="stMetric"] { background: white; padding: 1rem; border: 1px solid #e2eee4;
                                   border-radius: 14px; box-shadow: 0 2px 8px rgba(20,83,45,.05); }
      .note { background:#eff6ff; border-left:4px solid #3b82f6; padding:.8rem 1rem; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🌱 Smart Irrigation System</h1>
      <p>Machine-learning decision support for irrigation planning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Educational decision-support prototype. Predictions depend on the supplied dataset and are not a substitute for agronomic advice.")

@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.columns = [str(c).strip() for c in data.columns]
    return data

@st.cache_resource(show_spinner="Training the model on the dataset…")
def train_model(csv_path: str, file_mtime: float):
    data = load_dataset(csv_path).copy()
    if TARGET not in data.columns:
        raise ValueError(f"Required target column '{TARGET}' was not found in the CSV.")
    data = data.dropna(subset=[TARGET])
    data[TARGET] = data[TARGET].astype(str).str.strip()
    data = data[data[TARGET] != ""]
    if data.empty:
        raise ValueError(f"No usable rows found for target column '{TARGET}'.")
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    if X.shape[1] == 0:
        raise ValueError("The dataset has no feature columns.")

    # Keep the same feature names and preprocessing for training and user predictions.
    from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype

    categorical = [
  	c for c in X.columns
	if is_object_dtype(X[c].dtype)
	or isinstance(X[c].dtype, pd.CategoricalDtype)
	or is_bool_dtype(X[c].dtype)
	or is_string_dtype(X[c].dtype)
    ]
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric:
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        transformers.append(("categorical", categorical_pipe, categorical))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline([
        ("preprocess", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=4,
            n_jobs=-1,
        )),
    ])

    # Use a stratified holdout only when every class has enough samples.
    counts = y.value_counts()
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    labels = sorted(y.unique().tolist())
    matrix = confusion_matrix(y_test, predicted, labels=labels)
    # Refit on all available rows for the interactive prediction after evaluation.
    model.fit(X, y)
    return data, X, y, model, accuracy, report, labels, matrix, categorical, numeric, len(X_test)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}. Put the CSV in the same folder as app.py.")
    st.stop()

try:
    file_mtime = DATA_PATH.stat().st_mtime
    data, X, y, model, accuracy, report, labels, matrix, categorical, numeric, test_rows = train_model(str(DATA_PATH), file_mtime)
except Exception as exc:
    st.error(f"Could not load the dataset or train the model: {exc}")
    st.info("Check that smart_irrigation_system.csv is in the same folder as app.py and contains the Irrigation_Need column.")
    st.stop()

with st.sidebar:
    st.header("About this dashboard")
    st.write("The model is trained from the CSV when the app starts.")
    st.write(f"**Target:** `{TARGET}`")
    st.write(f"**Features:** {X.shape[1]}")
    st.write(f"**Evaluation holdout:** {test_rows:,} rows")
    st.divider()
    st.caption("Tip: change the inputs to explore how the trained model responds. Results reflect learned dataset patterns, not guaranteed real-world water requirements.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Dataset records", f"{len(data):,}")
m2.metric("Input features", f"{X.shape[1]}")
m3.metric("Holdout accuracy", f"{accuracy:.1%}")
m4.metric("Irrigation classes", f"{y.nunique()}")

st.divider()
prediction_tab, insights_tab, data_tab = st.tabs(["💧 Predict irrigation need", "📊 Model insights", "🗂️ Dataset overview"])

with prediction_tab:
    st.subheader("Field and weather inputs")
    st.write("Enter values for the features below. Category choices are taken directly from the CSV, avoiding mismatched hard-coded encodings.")
    with st.expander("What do Low, Medium, and High mean?"):
        st.markdown(
            """
            - **🟢 Low:** The model classifies the field's irrigation need as low compared with patterns in its training data.
            - **🟡 Medium:** The model classifies the field's irrigation need as medium compared with patterns in its training data.
            - **🔴 High:** The model classifies the field's irrigation need as high compared with patterns in its training data.

            **Important:** These are categories learned from the dataset, not measured water quantities. This model does **not** calculate millimetres or litres of water. Use local crop-specific irrigation guidance and consider current soil moisture, effective rainfall, crop stage, soil type, and irrigation-system efficiency before deciding how much to apply.
            """
        )
    with st.form("prediction_form"):
        values = {}
        sections = st.columns(2)
        for idx, col in enumerate(X.columns):
            with sections[idx % 2]:
                series = X[col]
                if col in categorical:
                    choices = sorted(series.dropna().astype(str).unique().tolist())
                    if not choices:
                        values[col] = ""
                        st.text_input(col.replace("_", " "), value="", disabled=True)
                    else:
                        # Mode gives a sensible dataset-based initial value.
                        mode = series.dropna().astype(str).mode()
                        default = mode.iloc[0] if not mode.empty else choices[0]
                        default_index = choices.index(default) if default in choices else 0
                        values[col] = st.selectbox(col.replace("_", " "), choices, index=default_index, key=f"input_{col}")
                else:
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    finite = numeric_series.dropna()
                    if finite.empty:
                        values[col] = 0.0
                        st.number_input(col.replace("_", " "), value=0.0, key=f"input_{col}")
                    else:
                        low = float(finite.min())
                        high = float(finite.max())
                        median = float(finite.median())
                        # Use number_input to avoid hiding valid dataset values behind arbitrary slider bounds.
                        if low == high:
                            values[col] = low
                            st.number_input(col.replace("_", " "), value=low, disabled=True, key=f"input_{col}")
                        else:
                            values[col] = st.number_input(
                                col.replace("_", " "),
                                min_value=low,
                                max_value=high,
                                value=min(max(median, low), high),
                                key=f"input_{col}",
                                help=f"Dataset range: {low:g} to {high:g}",
                            )
        submitted = st.form_submit_button("🌿 Predict irrigation need", type="primary", width="stretch")

    if submitted:
        row = pd.DataFrame([{col: values[col] for col in X.columns}], columns=X.columns)
        for col in categorical:
            row[col] = row[col].astype(str)
        for col in numeric:
            row[col] = pd.to_numeric(row[col], errors="coerce")
        try:
            result = str(model.predict(row)[0])
            probabilities = model.predict_proba(row)[0]
            class_names = model.named_steps["classifier"].classes_
            confidence = float(max(probabilities))
            st.divider()
            st.subheader("Prediction")
            if result.lower() == "high":
                st.error(f"🔴 Predicted irrigation need: **{result}**")
            elif result.lower() == "medium":
                st.warning(f"🟡 Predicted irrigation need: **{result}**")
            elif result.lower() == "low":
                st.success(f"🟢 Predicted irrigation need: **{result}**")
            else:
                st.info(f"Predicted irrigation need: **{result}**")
            st.metric("Model confidence for this prediction", f"{confidence:.1%}")
            st.caption("This is the model's estimated class probability, not a calibrated guarantee of correctness.")

            # Show field area when the dataset includes a recognizable area feature.
            area_columns = [
                c for c in X.columns
                if "area" in c.lower() and ("hectare" in c.lower() or "hect" in c.lower() or c.lower().endswith("_ha"))
            ]
            if area_columns:
                area_col = area_columns[0]
                try:
                    area_value = float(row.iloc[0][area_col])
                    st.metric("Field area entered", f"{area_value:,.2f} ha")
                except (TypeError, ValueError):
                    pass

            st.info(
                "**How to use this result:** Low, Medium, and High describe the category predicted from the dataset's patterns. "
                "They do not specify an amount of water. This app does not currently calculate litres or millimetres, so do not "
                "treat the category alone as an irrigation dose. Check crop-specific local guidance and field conditions before watering."
            )
            with st.expander("What should I consider before deciding how much water to apply?"):
                st.markdown(
                    """
                    - Crop and growth stage
                    - Current soil moisture and soil type
                    - Recent and forecast rainfall
                    - Temperature, humidity, wind, and sunlight
                    - Field area and irrigation-system efficiency

                    The model's category is decision support only. A numeric water recommendation would require validated crop- and location-appropriate irrigation guidance or a dataset with reliable water-quantity targets.
                    """
                )
            prob_df = pd.DataFrame({"Class": class_names, "Estimated probability": probabilities}).sort_values("Estimated probability", ascending=False)
            st.write("**Estimated probabilities by category**")
            st.bar_chart(prob_df.set_index("Class"), width="stretch")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with insights_tab:
    st.subheader("Evaluation on a held-out test split")
    st.markdown('<div class="note">The accuracy shown is measured on a random holdout from this dataset. It does not establish performance on new farms or real-world irrigation outcomes.</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.write("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(matrix, interpolation="nearest")
        ax.set_title("Actual vs predicted classes")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.write("**Per-class evaluation**")
        report_df = pd.DataFrame(report).T
        keep = [c for c in ["precision", "recall", "f1-score", "support"] if c in report_df.columns]
        st.dataframe(report_df[keep].round(3), width="stretch")
    st.caption("The holdout score is computed before the model is refit on all rows for interactive predictions.")

with data_tab:
    st.subheader("Dataset overview")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Irrigation class distribution**")
        counts = y.value_counts().rename_axis("Irrigation need").to_frame("Records")
        st.bar_chart(counts)
    with c2:
        st.write("**Missing values by column**")
        missing = data.isna().sum().rename_axis("Column").to_frame("Missing values")
        st.dataframe(missing, width="stretch")
    st.write("**Sample records**")
    st.dataframe(data.head(20), width="stretch")
    st.download_button(
        "Download dataset copy",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="smart_irrigation_system.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Smart Irrigation System • Built with Streamlit and scikit-learn")
