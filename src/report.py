import json
from src.config import REPORTS_DIR

def generate_benchmark_report(metrics, output_file="baseline_report.json"):
    """
    Export model metrics to a JSON file in the reports/ directory.
    """
    print("[-] Generating benchmark report...")
    
    # Directly use the REPORTS_DIR variable configured in src/config.py
    report_path = REPORTS_DIR / output_file
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"[+] Benchmark report generated at: {report_path}")