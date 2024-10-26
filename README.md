# APK Malware Detection Project

Dựa trên thông tin từ các file trong folder, có thể thấy một số điểm chính về dự án này:

1. Đây là một dự án phát hiện malware trong các file APK Android sử dụng học máy.

2. Dự án sử dụng mô hình BERT để phân loại dựa trên quyền hạn của APK.

3. Có 2 file Python chính:
   - `Train.py`: Dùng để huấn luyện mô hình
   - `Inference.py`: Cung cấp giao diện web Streamlit để phân tích APK

4. Dự án sử dụng PyTorch và Transformers để xây dựng và huấn luyện mô hình.

5. Dữ liệu huấn luyện được lưu trong file `APKPermissions.json`.

6. Mô hình được lưu trong thư mục `Checkpoints`.

7. Có sử dụng mixed precision training và gradient accumulation để tối ưu hóa quá trình huấn luyện.

8. Giao diện web cho phép người dùng nhập văn bản quyền hạn để phân loại.

9. Dự án có file requirements.txt để quản lý các thư viện phụ thuộc.

10. Có file .gitignore để loại trừ các file và thư mục không cần thiết khỏi version control.

## Cài đặt và Sử dụng

1. Cài đặt các thư viện cần thiết:
   ```
   pip install -r requirements.txt
   ```

2. Huấn luyện mô hình:
   ```
   python Train.py
   ```

3. Chạy giao diện web:
   ```
   streamlit run Inference.py
   ```
