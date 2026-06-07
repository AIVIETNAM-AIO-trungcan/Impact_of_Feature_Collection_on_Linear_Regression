# %%

from pathlib import Path

# 1. Tự động nhận diện môi trường Jupiter hoặc Script thông thường
try:
    # Nếu chạy dạng file script (.py) thông thương, dòng này sẽ đúng
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent
except NameError:
    # Nếu chạy trên giao diện Jupiter, biến __file__ bị lỗi -> dùng Path.cwd() thế chỗ
    project_root = Path.cwd()

# 2. Thiết lập đường dẫn đến file data.csv mục tiêu
target_data_file = project_root / 'data' / 'raw' / 'data.csv'
print('Full file path: ', target_data_file )

# 3. Kiểm tra file dữ liệu
if target_data_file.exists():
    print('🚀 Data is ready!')
    print('File extention: ', target_data_file.suffix)
    print('File name: ', target_data_file.stem)
else:
    print('⚠️ Pipeline status: Waiting for Tung Nguyen to upload data.csv into data/raw/')







# %%
