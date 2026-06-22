from pathlib import Path
import yaml

# 1. Tự động nhận diện môi trường (Jupyter hoặc Script) để tìm thư mục gốc
try:
    # Nếu chạy dạng file script (.py) bình thường (từ thư mục src)
    ROOT_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # Nếu chạy trên giao diện Jupyter Notebook
    cwd = Path.cwd()
    # Nếu đang đứng ở thư mục notebooks/ thì lùi lại 1 cấp, ngược lại lấy luôn thư mục hiện tại
    ROOT_DIR = cwd.parent if cwd.name == 'notebooks' else cwd

# 2. Định nghĩa các đường dẫn tuyệt đối 
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"


# 3. Tự động tạo thư mục nếu máy đồng đội chưa có
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 4. Hàm đọc file cấu hình YAML (MỚI THÊM CHO SPRINT 2)
def load_pipeline_config():
    """Đọc file config.yaml từ thư mục gốc và trả về dạng Dictionary"""
    config_path = ROOT_DIR / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"🚨 Không tìm thấy file cấu hình tại {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    return config

# 5. In thông báo và kiểm tra file (Tích hợp Data & Config)
if __name__ == "__main__":
    print(f"📌 Project Root: {ROOT_DIR}")
    print("✅ Hạ tầng thư mục đã được liên kết!\n")
    
    # --- Kiểm tra Data ---
    target_data_file = RAW_DATA_DIR / 'data.csv'
    
    if target_data_file.exists():
        print('🚀 Data is ready!')
        print('File extension: ', target_data_file.suffix)
        print('File name: ', target_data_file.stem)
        print(f'Full file path: {target_data_file}\n')
    else:
        print('⚠️ Pipeline status: Waiting for Tung Nguyen to upload data.csv into data/raw/\n')

    # --- Kiểm tra Config YAML (Phần mới thêm) ---
    print("="*40)
    print("⚙️  KIỂM TRA FILE CẤU HÌNH (config.yaml)")
    try:
        cfg = load_pipeline_config()
        print("✅ Đã nạp thành công cấu hình từ config.yaml")
        print(f"📊 Số lượng features tự động nhận diện:")
        print(f"   - Numerical: {len(cfg['features']['numerical_cols'])} cột")
        print(f"   - Categorical: {len(cfg['features']['categorical_cols'])} cột")
        print(f" Thuật toán Model: {cfg['model_params']['algorithm']}")
    except Exception as e:
        print(f"❌ Lỗi khi đọc config.yaml: {e}")
    print("="*40)
