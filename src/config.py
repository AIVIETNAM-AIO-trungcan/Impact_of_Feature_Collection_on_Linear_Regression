# =====================================================================
# MODULE: Configuration & Path Management
# DESCRIPTION: Resolves absolute project paths, initializes directory
#              structures, and loads YAML configurations.
# =====================================================================

from pathlib import Path
import yaml

# ---------------------------------------------------------
# 1. ENVIRONMENT DETECTION & PATH RESOLUTION
# ---------------------------------------------------------
try:
    # Triggered when running as a standard Python script (.py) from src/
    ROOT_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for Jupyter Notebook environment execution
    cwd = Path.cwd()
    # If the current working directory is 'notebooks/', step back one level
    ROOT_DIR = cwd.parent if cwd.name == "notebooks" else cwd

# ---------------------------------------------------------
# 2. DIRECTORY STRUCTURE DEFINITIONS
# ---------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# ---------------------------------------------------------
# 3. DIRECTORY INITIALIZATION
# ---------------------------------------------------------
# Automatically generate essential directories if they do not exist
for folder in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    NOTEBOOKS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 4. CONFIGURATION LOADER
# ---------------------------------------------------------
def load_pipeline_config():
    """
    Loads the pipeline configuration settings from config.yaml.

    Returns:
        dict: Parsed YAML configuration dictionary.

    Raises:
        FileNotFoundError: If the config.yaml file is missing in the root directory.
    """
    config_path = ROOT_DIR / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"🚨 Missing configuration file at {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


# ---------------------------------------------------------
# 5. MODULE EXECUTION & VALIDATION (Testing Block)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("⚙️ WORKSPACE ARCHITECTURE VALIDATION")
    print("=" * 50)

    print(f"[*] Project Root resolved to: {ROOT_DIR}")
    print("[*] Directory infrastructure linked successfully!\n")

    # --- Data Readiness Check ---
    print("[STAGE 1] Validating Data Infrastructure...")
    target_data_file = RAW_DATA_DIR / "data.csv"

    if target_data_file.exists():
        print("      [>] Data is ready for pipeline ingestion!")
        print(f"      [~] File extension: {target_data_file.suffix}")
        print(f"      [~] File name: {target_data_file.stem}")
        print(f"      [~] Full file path: {target_data_file}\n")
    else:
        print(
            "      [!] Pipeline status: Waiting for data.csv to be uploaded into data/raw/\n"
        )

    # --- YAML Configuration Check ---
    print("[STAGE 2] Validating Configuration (config.yaml)...")
    try:
        cfg = load_pipeline_config()
        print("      [>] Configuration loaded successfully.")

        num_features = len(cfg.get("features", {}).get("numerical_cols", []))
        cat_features = len(cfg.get("features", {}).get("categorical_cols", []))
        algorithm = cfg.get("model_params", {}).get("algorithm", "Unknown")

        print(
            f"      [~] Extracted features: {num_features} Numerical, {cat_features} Categorical"
        )
        print(f"      [~] Target Algorithm: {algorithm}")

    except Exception as e:
        print(f"❌ [ERROR] Failed to load config.yaml: {e}")

    print("\n" + "=" * 50 + "\n")
