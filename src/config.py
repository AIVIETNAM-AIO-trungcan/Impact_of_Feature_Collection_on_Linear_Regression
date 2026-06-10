from pathlib import Path

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

# 4. In thông báo và kiểm tra file (Kế thừa từ code cũ của Can)
if __name__ == "__main__":
    print(f"📌 Project Root: {ROOT_DIR}")
    print("✅ Hạ tầng đường dẫn đã được liên kết!\n")
    
    # Kiểm tra file dữ liệu cho Tùng
    target_data_file = RAW_DATA_DIR / 'data.csv'
    
    if target_data_file.exists():
        print('🚀 Data is ready!')
        print('File extension: ', target_data_file.suffix)
        print('File name: ', target_data_file.stem)
        print(f'Full file path: {target_data_file}')
    else:
        print('⚠️ Pipeline status: Waiting for Tung Nguyen to upload data.csv into data/raw/')
