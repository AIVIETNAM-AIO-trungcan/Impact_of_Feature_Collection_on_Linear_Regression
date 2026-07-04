# =====================================================================
# MODULE: Streamlit Dashboard (Academic Edition)
# DESCRIPTION: Interactive Web UI for displaying MLOps pipeline results,
#              with a minimalist, scientific research aesthetic.
# =====================================================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & ACADEMIC STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Feature Selection Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: Minimalist, scientific research style
st.markdown(
    """
    <style>
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,700;1,300&family=Source+Sans+Pro:wght@400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Source Sans Pro', sans-serif;
        color: #2c3e50;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Merriweather', serif;
        color: #1a252f;
    }
    
    .academic-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 15px;
        margin-bottom: 30px;
    }
    
    .academic-subtitle {
        font-size: 1.1rem;
        text-align: center;
        font-style: italic;
        color: #7f8c8d;
        margin-bottom: 40px;
    }

    /* Blocks: Problem - Reasoning - Solution */
    .block-problem { border-left: 4px solid #e74c3c; background-color: #fdf2f0; padding: 25px; margin-bottom: 20px; border-radius: 0 8px 8px 0; font-size: 1.1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    .block-reasoning { border-left: 4px solid #f39c12; background-color: #fef9f1; padding: 25px; margin-bottom: 20px; border-radius: 0 8px 8px 0; font-size: 1.1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    .block-solution { border-left: 4px solid #27ae60; background-color: #f0fbf4; padding: 25px; margin-bottom: 25px; border-radius: 0 8px 8px 0; font-size: 1.1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    
    .block-title { font-weight: 700; margin-bottom: 10px; font-size: 1.2rem; }
    
    /* Pipeline UI */
    .pipeline-stage { font-family: monospace; background: #ecf0f1; padding: 2px 6px; border-radius: 4px; }
    
    /* Chart Container */
    .chart-box { border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; background: #fff; margin-top: 15px;}
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 2. PATH DEFINITIONS & DATA LOADING
# ---------------------------------------------------------
REPORTS_DIR = Path("reports")
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
LEADERBOARD_FILE = REPORTS_DIR / "experiment_leaderboard.csv"


@st.cache_data
def load_leaderboard():
    if LEADERBOARD_FILE.exists():
        return pd.read_csv(LEADERBOARD_FILE)
    return None


@st.cache_data
def load_trace(file_name):
    trace_path = METRICS_DIR / file_name
    if trace_path.exists():
        return pd.read_csv(trace_path)
    return None


# Renderer for Mermaid SVG
def render_mermaid(code):
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: transparent;
                overflow: hidden;
            }}
            .mermaid-wrapper {{
                display: flex;
                justify-content: center;
                width: 100%;
                transform: scale(1.3);
                transform-origin: top center;
                margin-top: 20px;
                padding-bottom: 200px;
            }}
            .mermaid-container {{
                display: flex;
                justify-content: center;
                width: 100%;
            }}
            .mermaid-container svg {{
                max-width: 100% !important;
                height: auto !important;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid-wrapper">
            <div class="mermaid-container">
                <pre class="mermaid">
{code}
                </pre>
            </div>
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ 
                startOnLoad: true, 
                theme: 'base',
                themeVariables: {{
                    primaryColor: '#f8fafc',
                    primaryBorderColor: '#94a3b8',
                    lineColor: '#64748b',
                    fontFamily: 'Source Sans Pro',
                    fontSize: '16px'
                }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=1100, scrolling=False)


# Generate Mock Data for Visualizations in Data Methodology
@st.cache_data
def generate_skewed_data():
    np.random.seed(42)
    skewed_prices = np.random.lognormal(mean=10, sigma=1.2, size=1000)
    log_prices = np.log1p(skewed_prices)

    hist_skewed, bins_skewed = np.histogram(skewed_prices, bins=40)
    hist_normal, bins_normal = np.histogram(log_prices, bins=40)

    df_skew = pd.DataFrame(
        {"Observation Count": hist_skewed}, index=np.round(bins_skewed[:-1], 0)
    )
    df_norm = pd.DataFrame(
        {"Observation Count": hist_normal}, index=np.round(bins_normal[:-1], 2)
    )
    return df_skew, df_norm


# ---------------------------------------------------------
# 3. SIDEBAR (TEAM & METADATA & NAVIGATION)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 Untitled Team Project 1")
    st.caption("Module 1: Linear Regression & Feature Selection")
    st.divider()

    # NAVIGATION MENU
    st.markdown("### 🧭 REPORT NAVIGATION")
    page_selection = st.radio(
        "Select a reporting module:",
        [
            "🏠 Home & Pipeline Overview",
            "📖 1. Data Methodology",
            "🔍 2. Feature Selection Traces",
            "📊 3. Experimental Results & Charts",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Research Team:**")
    st.markdown("""
    * **Đào Trung Can** (Lead / MLOps)
    * **Lê Hoàng Trọng Minh** (Tech Lead)
    * **Tùng Nguyễn** (Data Engineer)
    * **Cao Bá Hoàng** (Model Engineer)
    * **Trần Phương Bình** (QA / Reviewer)
    """)

# ---------------------------------------------------------
# 4. HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="academic-title">IMPACT OF FEATURE SELECTION TECHNIQUES ON REGRESSION MODEL PERFORMANCE</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="academic-subtitle">An End-to-End MLOps Experiment on the House Pricing Dataset</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 5. MAIN CONTENT ROUTING
# ---------------------------------------------------------

# ==========================================
# PAGE 0: HOME & PIPELINE OVERVIEW
# ==========================================
if page_selection.startswith("🏠"):
    st.markdown("### ⚙️ Overall Pipeline Architecture")
    st.write(
        "MLOps-standard data flow diagram from raw data ingestion to experimental evaluation reporting."
    )

    mermaid_pipeline = """
    graph TD
        RawData[("data/raw/")] -->|"1. Ingest Dataset"| DataLoader["src/data_loader.py"]
        DataLoader -->|"2. Basic Clean"| Preprocess["src/preprocess.py"]
        
        Preprocess -->|"3. Split 80/20 - Prevent Data Leakage"| Splitter["src/data_splitter.py"]
        
        Splitter -->|"Train Data 80%"| Processor["src/processor.py (Feature Eng)"]
        Splitter -->|"Test Data 20%"| Processor
        
        Processor -->|"4. Fit Imputer on Train & Transform both"| FeatureSelector["src/feature_selector_apply.py"]
        
        subgraph PipelineEngine ["Managed by Pipeline Engine (main.py)"]
            FeatureSelector -->|"Config A: Baseline"| ModelBase["src/model.py - Baseline"]
            FeatureSelector -->|"Config B: Subsets/Filter..."| ModelOpt["src/model.py - Optimized"]
        end
        
        ModelBase --> Eval["Extract Performance Metrics"]
        ModelOpt --> Eval
        
        Eval --> Bench["Model Benchmarking: R2, RMSE, AIC/BIC"]
        Bench -->|"Export Artifacts & Figures"| Reports[("reports/")]
        
        style RawData fill:#1e1e2e,stroke:#a6adc8,stroke-width:1px,color:#fff
        style Preprocess fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px,color:#fff
        style Splitter fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff
        style Processor fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff
        style Reports fill:#181825,stroke:#a6e3a1,stroke-width:1px,color:#fff
        style FeatureSelector fill:#fef9f1,stroke:#f39c12,stroke-width:2px
    """

    render_mermaid(mermaid_pipeline)

    st.divider()

    st.markdown("### 📁 Source Code Structure Directory")
    st.write(
        "Overview of the project's repository structure. Select a specific file below to understand its detailed role."
    )

    # --- THÊM SƠ ĐỒ CÂY THƯ MỤC Ở ĐÂY ---
    tree_structure = """
    PROJECT_ROOT/
    ├── data/
    │   ├── processed/                # Cleaned & partitioned data (train.csv, test.csv)
    │   └── raw/                      # Original immutable datasets
    ├── models/                       # Serialized production models (.pkl)
    ├── reports/
    │   ├── figures/                  # Analytical charts (Scatter, Trade-off)
    │   ├── metrics/                  # JSON reports & CSV step traces
    │   └── experiment_leaderboard.csv
    ├── src/                          # Core source code modules
    │   ├── app.py                    # Streamlit Dashboard UI
    │   ├── config.py                 # Path & environment management
    │   ├── data_loader.py            # Data ingestion
    │   ├── data_splitter.py          # Train/Test partitioning
    │   ├── feature_selector_apply.py # Selection algorithms engine
    │   ├── model.py                  # Hybrid training engine
    │   ├── preprocess.py             # Basic cleaning & dropping
    │   ├── processor.py              # Feature engineering (Impute, Encode, Log)
    │   └── report.py                 # Evaluation & artifact generation
    ├── config.yaml                   # Centralized control configuration
    ├── main.py                       # Pipeline orchestrator
    └── requirements.txt              # Environment dependencies
    """
    st.code(tree_structure, language="text")

    st.markdown("#### 🔍 Detailed File Specifications")

    file_explanations = {
        "config.yaml": "Central configuration file (Control Station). Contains algorithm parameters, feature selection settings, and normalization targets, enabling pipeline control without modifying code.",
        "main.py": "Main orchestrator script. Links all modules and executes the experimental pipeline end-to-end. Manages the loop across different feature selection methods.",
        "src/data_loader.py": "Data engineering module. Responsible for safely reading raw data from the physical directory (data/raw/) with automated error handling.",
        "src/preprocess.py": "Initial data cleaning module. Performs the removal of completely non-analytical columns (e.g., IDs, identifiers).",
        "src/data_splitter.py": "Splits the dataset into Train/Test sets and physically saves them to data/processed/. Performing the physical split at this stage strictly guarantees the prevention of Data Leakage in subsequent steps.",
        "src/processor.py": "Executes high-level Feature Engineering: Uses the Train set to learn imputation parameters and applies them to both sets. Performs categorical encoding (One-Hot Encoding) and distribution transformations (Log Transform).",
        "src/feature_selector_apply.py": "The core feature selection algorithm engine. Executes Filter, Forward, Backward, Lasso, and Best Subset algorithms, recording execution traces based on AIC/BIC criteria.",
        "src/model.py": "Hybrid model training engine. Uses Scikit-learn for production prediction while simultaneously using Statsmodels to extract in-depth statistical parameters.",
        "src/report.py": "Report rendering engine. Responsible for exporting JSON reports, generating the aggregated Leaderboard CSV, and plotting analytical charts saved to the reports/ directory.",
    }

    selected_file = st.selectbox(
        "📌 Please select a source code file to view its function:",
        list(file_explanations.keys()),
    )

    st.info(
        f"**Path / Filename:** `{selected_file}`\n\n**Function and Task:** {file_explanations[selected_file]}"
    )

# ==========================================
# PAGE 1: DATA METHODOLOGY
# ==========================================
elif page_selection.startswith("📖 1"):
    st.markdown("### Data Processing and Transformation (Data Engineering)")
    st.write(
        "Before being fed into the feature selection algorithm, raw data requires careful refinement. Use the slider below to follow the team's problem-solving logic."
    )

    methodologies = {
        "1. Missing Values & Noise": {
            "problem": "The dataset contains non-predictive identifier columns (e.g., ID) and many critical features have missing values (NaN).",
            "reasoning": "Retaining ID columns introduces noise (local overfitting). Dropping all rows with missing data leads to a massive loss of valuable training information. Note: Imputation parameters must not be calculated on the entire dataset to prevent Data Leakage.",
            "solution": "- **Drop:** Completely remove redundant columns (ID, URL...).<br>- **Imputation (Fit on Train):** Learn the Median (for numerical variables) and Mode (for categorical variables) from the Train set (80%) to impute missing values in both Train and Test sets.",
        },
        "2. Categorical Encoding": {
            "problem": "Linear Regression (OLS) models and related mathematical algorithms only accept real numerical matrices as input.",
            "reasoning": "If Label Encoding is used (assigning 1, 2, 3), the model misinterprets an ordinal relationship (e.g., 3 is greater than 1). We need an encoding method that avoids creating this artificial ordinality.",
            "solution": "Apply **One-Hot Encoding (Dummy Variables)**. Each category becomes an independent binary column (0 or 1). Combined with the `drop_first=True` parameter to avoid the Dummy Variable Trap (multicollinearity).",
        },
        "3. Skewed Distribution": {
            "problem": "The target variable (House Price) and the area feature exhibit a strong right-skewed distribution, with extreme outliers possessing extremely high values.",
            "reasoning": "Linear regression carries a crucial assumption that the residuals must be normally distributed. Severe right-skewness violates this assumption, causing the RMSE to be heavily distorted by ultra-expensive houses.",
            "solution": "Apply the Natural Logarithm transformation **`np.log1p()`** to the skewed variables. This compresses the distance between values, bringing the distribution closer to normal (Bell Curve), thereby enhancing the stability and generalization capability of the model.",
        },
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Knowledge Domain:")
        selected_topic = st.selectbox(
            "Select processing topic:",
            list(methodologies.keys()),
            label_visibility="collapsed",
        )

    st.markdown("---")
    st.markdown("#### Logical Reasoning Process:")

    progress = st.select_slider(
        "Drag to view the analytical process:",
        options=[
            "🔴 1. Problem Identification",
            "🧠 2. Logical Reasoning",
            "✅ 3. Proposed Solution",
        ],
        value="🔴 1. Problem Identification",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    topic_data = methodologies[selected_topic]

    # ---------------------------------------------------------
    # STATE 1: PROBLEM
    # ---------------------------------------------------------
    if progress.startswith("🔴"):
        st.markdown(
            f"""
        <div class="block-problem">
            <div class="block-title">🔴 The Problem</div>
            {topic_data['problem']}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        if selected_topic.startswith("1"):
            st.markdown("##### 📉 Raw Data State Illustration (Sample)")
            df_raw_example = pd.DataFrame(
                {
                    "Property_ID": ["ID_001", "ID_002", "ID_003", "ID_004", "ID_005"],
                    "Price": [1200000, 1500000, 1100000, 2000000, 1300000],
                    "Room_Floor": [3, np.nan, 2, np.nan, 1],
                    "Amenity_View": ["City", "Park", np.nan, "City", np.nan],
                }
            )
            st.dataframe(
                df_raw_example.style.highlight_null(color="#ffcccc"),
                use_container_width=True,
            )
            st.caption(
                "⚠️ **Problem Analysis:** The `Property_ID` column is entirely meaningless for prediction. The red-highlighted cells represent missing data (`NaN`), which will cause immediate system errors if fed into the algorithm."
            )

        elif selected_topic.startswith("2"):
            st.markdown("##### 📉 Raw Data State Illustration")
            st.write(
                "Text data cannot be fed into the computational matrix of the Regression algorithm:"
            )
            df_cat_problem = pd.DataFrame(
                {
                    "Property_ID": ["P01", "P02", "P03"],
                    "Location": ["Downtown", "Suburb", "Downtown"],
                    "Property Type (Raw)": ["Apartment", "Villa", "Townhouse"],
                    "Label Encode Mistake": [1, 3, 2],
                }
            )
            st.dataframe(df_cat_problem, use_container_width=True, hide_index=True)
            st.caption(
                "⚠️ If encoded as [1, 2, 3], the model will incorrectly learn that a Villa (3) is worth 3 times an Apartment (1)."
            )

        elif selected_topic.startswith("3"):
            st.markdown("##### 📉 Raw Data State Illustration")
            df_skew, _ = generate_skewed_data()
            st.bar_chart(df_skew, color="#e74c3c", height=350)
            st.caption(
                "⚠️ Right-skewed variable with a long tail containing ultra-high value houses (Outliers)."
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # STATE 2: REASONING
    # ---------------------------------------------------------
    elif progress.startswith("🧠"):
        st.markdown(
            f"""
        <div class="block-reasoning">
            <div class="block-title">🧠 Reasoning & Evaluation</div>
            {topic_data['reasoning']}
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # STATE 3: SOLUTION
    # ---------------------------------------------------------
    elif progress.startswith("✅"):
        st.markdown(
            f"""
        <div class="block-solution">
            <div class="block-title">✅ Technical Solution</div>
            {topic_data['solution']}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="chart-box">', unsafe_allow_html=True)

        if selected_topic.startswith("1"):
            st.markdown("##### 📈 Post-Processing Results (Data Imputation)")
            c1, c2, c3 = st.columns(3)
            c1.metric(
                label="Deleted Rows",
                value="0",
                delta="100% Data Preserved",
                delta_color="normal",
            )
            c2.metric(
                label="Room_Floor Treatment",
                value="Median Imputed",
                delta="Fill Value: 2",
                delta_color="normal",
            )
            c3.metric(
                label="Amenity_View Treatment",
                value="Mode Imputed",
                delta="Fill Value: 'City'",
                delta_color="normal",
            )

            st.write("Sample 5-row dataset after processing functions:")
            df_clean_example = pd.DataFrame(
                {
                    "Price": [1200000, 1500000, 1100000, 2000000, 1300000],
                    "Room_Floor": [3.0, 2.0, 2.0, 2.0, 1.0],
                    "Amenity_View": ["City", "Park", "City", "City", "City"],
                }
            )

            def highlight_imputed(row):
                styles = [""] * len(row)
                if row.name in [1, 3]:
                    styles[df_clean_example.columns.get_loc("Room_Floor")] = (
                        "background-color: #d4edda; color: #155724; font-weight: bold"
                    )
                if row.name in [2, 4]:
                    styles[df_clean_example.columns.get_loc("Amenity_View")] = (
                        "background-color: #d4edda; color: #155724; font-weight: bold"
                    )
                return styles

            st.dataframe(
                df_clean_example.style.apply(highlight_imputed, axis=1),
                use_container_width=True,
            )
            st.caption(
                "✨ **Result:** Redundant columns are removed. Empty cells (green) are filled with representative distribution values. The model can now learn without losing any valuable observation rows!"
            )

        elif selected_topic.startswith("2"):
            st.markdown("##### 📈 Processing Results (Post-Pipeline)")
            st.write(
                "Using One-Hot Encoding (Sparse Matrix): Splits 1 column into $N$ independent binary columns."
            )
            df_cat_solved = pd.DataFrame(
                {
                    "Property_ID": ["P01", "P02", "P03"],
                    "Is_Apartment": [1, 0, 0],
                    "Is_Townhouse": [0, 0, 1],
                    "Is_Villa": [0, 1, 0],
                }
            )
            st.dataframe(df_cat_solved, use_container_width=True, hide_index=True)
            st.caption(
                "✨ Variables are now completely independent, eliminating artificial ordinal relationships. The machine can fully comprehend (0s and 1s)."
            )

        elif selected_topic.startswith("3"):
            st.markdown("##### 📈 Processing Results (Post-Pipeline)")
            c1, c2 = st.columns([1, 4])
            with c1:
                st.metric(
                    label="Skewness",
                    value="~0.1",
                    delta="Near-Normal",
                    delta_color="normal",
                )
                st.metric(
                    label="Outlier Tail",
                    value="Compressed",
                    delta="Noise Reduced",
                    delta_color="normal",
                )
            with c2:
                _, df_norm = generate_skewed_data()
                st.bar_chart(df_norm, color="#3498db", height=300)
            st.caption(
                "✨ The distribution post `np.log1p()` transformation forms a Bell Curve, strictly adhering to the Linear Regression (OLS) assumption."
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 2: FEATURE SELECTION TRACES
# ==========================================
elif page_selection.startswith("🔍 2"):
    st.markdown("### Convergence Process of Selection Algorithms")
    st.write(
        "Observe how each algorithm makes decisions to add (Forward), remove (Backward), or evaluate feature importance (Filter/Lasso) across iterations."
    )

    trace_files = list(METRICS_DIR.glob("*_step_trace.csv"))

    if not trace_files:
        st.warning("No Trace data available. Please run `python main.py` first.")
    else:
        trace_dict = {
            f.name.replace("_step_trace.csv", "").replace("_", " ").title(): f.name
            for f in trace_files
        }

        selected_method = st.selectbox(
            "📌 Select a Method to observe:", list(trace_dict.keys())
        )

        df_trace = load_trace(trace_dict[selected_method])

        if df_trace is not None:
            criterion_val = (
                str(df_trace["Criterion"].iloc[0]).upper()
                if ("Criterion" in df_trace.columns and not df_trace.empty)
                else ""
            )

            if criterion_val == "ALPHA":
                score_label = "Alpha Parameter (L1)"
            elif criterion_val in ["AIC", "BIC", "ADJ_R2"]:
                score_label = f"{criterion_val} Score"
            else:
                score_label = "Evaluation Score"

            col_cfg = {
                "Step": st.column_config.TextColumn("Step"),
                "Action": st.column_config.TextColumn("Action"),
                "Features_Used": st.column_config.TextColumn("Current Feature Set"),
                "Criterion": None,
            }

            if "Criterion_Score" in df_trace.columns:
                if df_trace["Criterion_Score"].isnull().all():
                    col_cfg["Criterion_Score"] = None
                else:
                    col_cfg["Criterion_Score"] = st.column_config.NumberColumn(
                        score_label, format="%.4f"
                    )

            if "MSE_Score" in df_trace.columns:
                if df_trace["MSE_Score"].isnull().all():
                    col_cfg["MSE_Score"] = None
                else:
                    col_cfg["MSE_Score"] = st.column_config.NumberColumn(
                        "MSE", format="%.4f"
                    )

            st.dataframe(
                df_trace,
                use_container_width=True,
                hide_index=True,
                column_config=col_cfg,
            )

# ==========================================
# PAGE 3: RESULTS & CHARTS (SELECTBOX DRIVEN)
# ==========================================
elif page_selection.startswith("📊 3"):
    st.markdown("### Performance Evaluation Analysis")

    view_options = [
        "1. Aggregated Leaderboard",
        "2. Error Comparison (RMSE)",
        "3. Explanatory Power Comparison (R²)",
        "4. Trade-off Analysis (Features vs. Time)",
        "5. Actual vs. Predicted Distribution (Scatter Plots)",
    ]

    selected_view = st.selectbox("Please select a visual report to view:", view_options)

    st.markdown("---")

    if selected_view.startswith("1"):
        df_leaderboard = load_leaderboard()
        if df_leaderboard is not None:
            st.markdown("#### Model Performance Leaderboard (Test Set)")
            st.dataframe(
                df_leaderboard,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Test_RMSE": st.column_config.NumberColumn(
                        "Test RMSE", format="%.0f"
                    ),
                    "Test_R2": st.column_config.NumberColumn("Test R²", format="%.4f"),
                    "Train_AIC": st.column_config.NumberColumn(
                        "Train AIC", format="%.0f"
                    ),
                    "Execution_Time_sec": st.column_config.NumberColumn(
                        "Time (s)", format="%.2f"
                    ),
                },
            )
        else:
            st.warning("Leaderboard data not found.")

    elif selected_view.startswith("2"):
        img_rmse = FIGURES_DIR / "leaderboard_01_rmse_comparison.png"
        if img_rmse.exists():
            st.image(
                Image.open(img_rmse),
                caption="RMSE Error Comparison Chart across methods",
                use_container_width=True,
            )
            st.info(
                "💡 **Note:** The red bar represents the optimal model (Lowest Error)."
            )

    elif selected_view.startswith("3"):
        img_r2 = FIGURES_DIR / "leaderboard_03_r2_comparison.png"
        if img_r2.exists():
            st.image(
                Image.open(img_r2),
                caption="R² Coefficient Comparison Chart",
                use_container_width=True,
            )
            st.info(
                "💡 **Note:** The green bar represents the model explaining the most data variance."
            )

    elif selected_view.startswith("4"):
        img_tradeoff = FIGURES_DIR / "leaderboard_02_efficiency_tradeoff.png"
        if img_tradeoff.exists():
            st.image(
                Image.open(img_tradeoff),
                caption="Trade-off Analysis between Number of Features (Left Axis) and Execution Time (Right Axis)",
                use_container_width=True,
            )
            st.info(
                "💡 **Note:** Provides perspective on Computational Cost. Best Subset often consumes significant time despite using fewer features."
            )

    elif selected_view.startswith("5"):
        scatter_plots = sorted(list(FIGURES_DIR.glob("*_actual_vs_predicted.png")))
        if scatter_plots:
            st.markdown("#### Dispersion Chart: Actual vs Predicted")

            plot_names = [
                p.name.replace("_actual_vs_predicted.png", "").upper()
                for p in scatter_plots
            ]
            view_all_option = "🌟 Aggregate of all models"
            options = plot_names + [view_all_option]

            selected_scatter = st.selectbox(
                "📌 Select a model to view details:", options
            )

            st.markdown("---")

            if selected_scatter == view_all_option:
                st.info(
                    "💡 **Aggregate View:** Displays a comparison grid of all algorithms."
                )
                cols = st.columns(2)
                for i, plot_path in enumerate(scatter_plots):
                    with cols[i % 2]:
                        st.image(
                            Image.open(plot_path),
                            caption=f"Model: {plot_names[i]}",
                            use_container_width=True,
                        )
            else:
                idx = plot_names.index(selected_scatter)
                selected_path = scatter_plots[idx]

                col1, col2, col3 = st.columns([1, 4, 1])
                with col2:
                    st.image(
                        Image.open(selected_path),
                        caption=f"Model Details: {selected_scatter}",
                        use_container_width=True,
                    )
                    st.info(
                        "💡 **Note:** The closer the blue data points adhere to the red dashed line (Ideal Fit), the more accurate the model's predictions."
                    )
        else:
            st.warning("No Scatter charts available. Please run the Pipeline first.")
