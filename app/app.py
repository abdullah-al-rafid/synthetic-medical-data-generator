import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Synthetic Data Generator",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

# app.py is inside:
# C:\Heart Disease\app\app.py
#
# Therefore BASE_DIR becomes:
# C:\Heart Disease

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# FILE FINDER
# ============================================================

def find_file(filename, preferred_folder=None):
    """
    Find a project file.

    Priority:
    1. preferred_folder/filename
    2. project root/filename

    This allows the app to work both:
    - locally
    - on GitHub / Streamlit Cloud
    """

    possible_paths = []

    if preferred_folder:
        possible_paths.append(
            BASE_DIR / preferred_folder / filename
        )

    possible_paths.append(
        BASE_DIR / filename
    )

    for path in possible_paths:
        if path.exists():
            return path

    # Return the preferred path even if it doesn't exist.
    # This allows the error message to show the expected location.
    return possible_paths[0]


# ============================================================
# DATASET PATHS
# ============================================================

DATASETS = {

    # Original synthetic datasets
    "medGAN": find_file(
        "medGAN_Synthetic_Data.csv",
        "synthetic_data"
    ),

    "CTGAN": find_file(
        "CTGAN_Synthetic_Data.csv",
        "synthetic_data"
    ),

    "TVAE": find_file(
        "TVAE_Synthetic_Data.csv",
        "synthetic_data"
    ),

    "Gaussian Copula": find_file(
        "GaussianCopula_Synthetic_Data.csv",
        "synthetic_data"
    ),

    "CopulaGAN": find_file(
        "CopulaGAN_Synthetic_Data.csv",
        "synthetic_data"
    ),

    "CTAB-GAN": find_file(
        "CTABGAN_Synthetic_Data.csv",
        "synthetic_data"
    ),

    # Hybrid / final datasets
    "medGAN + CTAB-GAN": find_file(
        "medgan_ctabgan.csv",
        "final_model"
    ),

    "Current Hybrid": find_file(
        "final_hybrid_dataset.csv",
        "final_model"
    ),
}


# ============================================================
# OPTIONAL RESULT FILES
# ============================================================

MODEL_COMPARISON_FILE = find_file(
    "Model_Comparison.csv",
    "results"
)

MODEL_RANKING_FILE = find_file(
    "Final_Model_Ranking.csv",
    "results"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1.5rem;
    }

    /* Header */
    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 5px;
        color: white;
    }

    .subtitle {
        font-size: 17px;
        color: #e5e7eb;
        margin-bottom: 35px;
    }

    /* Section title */
    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Metric labels */
    div[data-testid="stMetricLabel"] {
        font-size: 15px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 32px;
    }

    /* Download button */
    .stDownloadButton button {
        width: 100%;
        font-weight: 600;
    }

    /* Divider */
    hr {
        margin-top: 25px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">❤️ Heart Disease Synthetic Data Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Generate, preview and download synthetic heart disease data using the selected model.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_dataset(file_path):
    """
    Load a CSV dataset.
    """
    return pd.read_csv(file_path)


def get_heart_disease_rate(df):
    """
    Calculate heart disease rate from the target column.
    """

    possible_columns = [
        "Heart Disease Status_Yes",
        "Heart Disease Status",
        "Heart_Disease_Status",
        "HeartDisease"
    ]

    target = None

    for col in possible_columns:
        if col in df.columns:
            target = col
            break

    if target is None:
        return None

    series = df[target]

    # Handle numeric target
    if pd.api.types.is_numeric_dtype(series):
        return float(series.mean() * 100)

    # Handle text/bool target
    series_str = series.astype(str).str.lower().str.strip()

    positive_values = {
        "1",
        "true",
        "yes",
        "positive",
        "heart disease",
        "heart_disease"
    }

    positive_count = series_str.isin(positive_values).sum()

    if len(series_str) == 0:
        return None

    return float((positive_count / len(series_str)) * 100)


def get_missing_values(df):
    """
    Return total missing values.
    """
    return int(df.isnull().sum().sum())


def get_duplicate_rows(df):
    """
    Return number of duplicate rows.
    """
    return int(df.duplicated().sum())


def format_rate(rate):
    """
    Format percentage.
    """
    if rate is None:
        return "N/A"

    return f"{rate:.2f}%"


# ============================================================
# MODEL SELECTION
# ============================================================

st.markdown("### Select Synthetic Data Model")

model_options = list(DATASETS.keys())

selected_model = st.selectbox(
    "Select Synthetic Data Model",
    model_options,
    index=0,
    label_visibility="collapsed"
)

selected_file = DATASETS[selected_model]


# ============================================================
# DATASET LOADING
# ============================================================

if not selected_file.exists():

    st.error(
        f"{selected_file.name} not found."
    )

    st.info(
        f"Expected location:\n\n{selected_file}"
    )

    st.stop()


try:

    df = load_dataset(selected_file)

except Exception as e:

    st.error("Unable to load the selected dataset.")

    st.exception(e)

    st.stop()


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    f"{selected_model} dataset loaded successfully."
)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

records = len(df)
features = len(df.columns)

heart_rate = get_heart_disease_rate(df)
missing_values = get_missing_values(df)


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Records",
        f"{records:,}"
    )

with col2:

    st.metric(
        "Features",
        f"{features}"
    )

with col3:

    st.metric(
        "Heart Disease Rate",
        format_rate(heart_rate)
    )

with col4:

    st.metric(
        "Missing Values",
        f"{missing_values:,}"
    )


st.divider()


# ============================================================
# SYNTHETIC DATA PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📊 Synthetic Data Preview</div>',
    unsafe_allow_html=True
)

preview_rows = st.slider(
    "Number of rows to display",
    min_value=10,
    max_value=min(1000, len(df)),
    value=min(10, len(df)),
    step=10
)

st.dataframe(
    df.head(preview_rows),
    use_container_width=True,
    height=430
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📋 Dataset Information</div>',
    unsafe_allow_html=True
)

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.markdown("#### Dataset Shape")

    st.write(
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

with info_col2:

    st.markdown("#### Duplicate Rows")

    duplicate_rows = get_duplicate_rows(df)

    st.write(
        f"{duplicate_rows:,}"
    )


# ============================================================
# COLUMN INFORMATION
# ============================================================

with st.expander("View Dataset Columns"):

    for i, column in enumerate(df.columns, start=1):

        st.write(
            f"{i}. `{column}`"
        )


# ============================================================
# DATA TYPES
# ============================================================

with st.expander("View Data Types"):

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        dtype_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

with st.expander("View Statistical Summary"):

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:

        st.dataframe(
            numeric_df.describe().T,
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns available."
        )


# ============================================================
# HEART DISEASE DISTRIBUTION
# ============================================================

target_candidates = [
    "Heart Disease Status_Yes",
    "Heart Disease Status",
    "Heart_Disease_Status",
    "HeartDisease"
]

target_column = None

for column in target_candidates:

    if column in df.columns:

        target_column = column

        break


if target_column is not None:

    st.markdown(
        '<div class="section-title">❤️ Heart Disease Distribution</div>',
        unsafe_allow_html=True
    )

    distribution = (
        df[target_column]
        .value_counts(dropna=False)
        .rename_axis(target_column)
        .reset_index(name="Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    st.dataframe(
        distribution,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD DATASET
# ============================================================

st.markdown(
    '<div class="section-title">⬇️ Download Synthetic Dataset</div>',
    unsafe_allow_html=True
)

csv_data = df.to_csv(index=False).encode("utf-8")

download_name = (
    selected_model
    .replace(" ", "_")
    .replace("+", "plus")
    .replace("-", "")
    + "_Synthetic_Data.csv"
)

st.download_button(
    label=f"Download {selected_model} Dataset",
    data=csv_data,
    file_name=download_name,
    mime="text/csv"
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Model Comparison</div>',
    unsafe_allow_html=True
)

if MODEL_COMPARISON_FILE.exists():

    try:

        comparison_df = pd.read_csv(
            MODEL_COMPARISON_FILE
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception:

        st.warning(
            "Model_Comparison.csv exists but could not be loaded."
        )

else:

    st.info(
        "Model_Comparison.csv is not available yet."
    )


# ============================================================
# FINAL MODEL RANKING
# ============================================================

st.markdown(
    '<div class="section-title">🏆 Final Model Ranking</div>',
    unsafe_allow_html=True
)

if MODEL_RANKING_FILE.exists():

    try:

        ranking_df = pd.read_csv(
            MODEL_RANKING_FILE
        )

        st.dataframe(
            ranking_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception:

        st.warning(
            "Final_Model_Ranking.csv exists but could not be loaded."
        )

else:

    st.info(
        "Final_Model_Ranking.csv is not available yet."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Heart Disease Synthetic Data Generation Project"
)