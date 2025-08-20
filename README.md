# 🧠 Digit Recognition with CNN

Dự án nhận diện chữ số viết tay sử dụng mạng nơ-ron tích chập (CNN) huấn luyện trên bộ dữ liệu MNIST.

## 📦 Mô hình
- Kiến trúc: 2 lớp convolution + maxpool + fully connected
- Dữ liệu: MNIST (60,000 ảnh huấn luyện, 10,000 ảnh kiểm tra)
- Accuracy: ~98% trên tập kiểm tra

## ▶️ Cách sử dụng
```python
# Huấn luyện mô hình
python train.py

# Dự đoán ảnh mới
python predict.py --img_path path_to_image.png
# kết quả 
![pngtree-drawing-digital-ink-1-png-image_459066](https://github.com/user-attachments/assets/9c208937-6fa1-4c17-85cd-26f79355b82b)
"C:\Users\Dell\OneDrive\Pictures\Screenshots\Ảnh chụp màn hình 2025-08-20 100248.png"
