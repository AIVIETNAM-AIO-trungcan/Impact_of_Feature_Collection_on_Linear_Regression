Impact of Feature Selection on Predictive Modeling (MLOps Pipeline)

📝 Project Description

Evaluating the impact of Filter, Wrapper, and Embedded Feature Selection techniques on predictive models (Linear Regression & Random Forest).
Unlike standard academic scripts, this project is engineered as an End-to-End MLOps Pipeline. It features dynamic configuration, hybrid metric extraction (Scikit-learn + Statsmodels for AIC/BIC), step-by-step algorithm tracing, and automated leaderboard generation tracked via MLflow.

Built with Python, Scikit-learn, Statsmodels, and MLflow.

🛠️ System Architecture & Pipeline Schema

The system is designed with an embedded experimental loop to systematically evaluate different feature configurations in a single run, strictly adhering to our MLOps Coding Standards.

graph TD
%% Config & Data Ingestion Layer
Config[(config.yaml)] -.-> PipelineEngine
RawData[(data/raw/data.csv)] -->|Ingest Dataset| DataLoader(src/data_loader.py)
DataLoader -->|Raw DataFrame| Preprocess(src/preprocess.py)
Preprocess -->|Train/Test Split| Splitter(src/data_splitter.py)
Splitter -->|Persist Clean Data| ProcessedData[(data/processed/)]

    %% Advanced Processing
    ProcessedData -->|Fit & Transform| Processor(src/processor.py)
    Processor -->|Dynamic Log/Raw Transform| FeatureSelector

    %% Feature Selection & Experiment Loop
    subgraph MLflow Experiment Loop [Managed by main.py]
        FeatureSelector{{src/feature_selector_apply.py}} -->|Best Subset/Forward/<br>Backward/Lasso/Filter| ModelEngine(src/model.py)

        %% Hybrid Training
        ModelEngine -->|Sklearn| ProdModel[(models/*.pkl)]
        ModelEngine -->|Statsmodels| StatMetrics[Extract AIC, BIC, Adj R²]

        %% Evaluation
        ProdModel --> Evaluator(src/report.py)
        StatMetrics --> Evaluator
    end

    %% Output & Reporting Layer
    FeatureSelector -->|Export Path| Traces[(reports/metrics/*_trace.csv)]
    Evaluator -->|Log Metrics| MLflow[(MLflow Tracking UI)]
    Evaluator -->|Aggregated Benchmarks| Leaderboard[experiment_leaderboard.csv]
    Evaluator -->|Scatter & Trade-off Plots| Visuals[(reports/figures/)]

    %% Schematic Formatting
    style RawData fill:#1e1e2e,stroke:#a6adc8,stroke-width:1px
    style ProcessedData fill:#181825,stroke:#89b4fa,stroke-width:1px
    style Traces fill:#181825,stroke:#fab387,stroke-width:1px
    style MLflow fill:#181825,stroke:#74c7ec,stroke-width:1px
    style Config fill:#1e1e2e,stroke:#f9e2af,stroke-width:1px

👥 Team Members (AI Vietnam_UntitledTeam - Module 1)

Đào Trung Can (Leader / AI Engineer - Pipeline & MLOps Architect)

Lê Hoàng Trọng Minh (Tech Leader / Workspace & Standards)

Tùng Nguyễn (AI Engineer - Data)

Cao Bá Hoàng (AI Engineer - Model)

Trần Phương Bình (QA / Reviewer)

📁 Repository Structure

config.yaml: Centralized control station for features, algorithm params, and selection criteria (AIC/BIC).

data/: Contains raw/ and processed/ datasets (ignored by Git).

docs/: Holds coding_standards.md for consistent team collaboration.

src/: Core source code for the automated data and modeling pipeline.

reports/: Auto-generated JSON metrics, step traces, Leaderboard CSV, and comparative PNG charts.

models/: Serialized .pkl models ready for production.

🚀 How to Run

1. Setup Environment:

python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

2. Execute the Full Pipeline:
   The pipeline will automatically read config.yaml and execute the experiment loop.

python main.py

3. Run Specific Selection Methods:
   Override the default methods using terminal arguments:

python main.py --methods baseline lasso forward best_subset

4. View MLflow Dashboard:

mlflow ui

📊 Results Summary

Execution Traces: Check reports/metrics/\*\_step_trace.csv to see the exact variables added/removed at each step alongside their corresponding MSE and AIC/BIC scores.

Leaderboard: A summarized CSV and set of analytical charts (RMSE comparison, R² evaluation, and Efficiency Trade-off) are generated automatically in reports/ post-execution.
