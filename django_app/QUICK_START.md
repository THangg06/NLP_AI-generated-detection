# ⚡ Quick Start - Django Fake News Detector

## ✅ Bước 1: Kích hoạt Environment

```powershell
# Windows PowerShell
.\fake-news-env\Scripts\Activate.ps1

# Hoặc nếu gặp lỗi
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\fake-news-env\Scripts\Activate.ps1
```

## ✅ Bước 2: Cài đặt Django

```powershell
pip install Django==4.2 Pillow==10.1.0
```

## ✅ Bước 3: Thiết lập Database

```powershell
cd django_app
python manage.py migrate
```

## ✅ Bước 4: Tạo Admin Account

```powershell
python manage.py createsuperuser
# Nhập: username, email, password
```

## ✅ Bước 5: Chạy Server

```powershell
python manage.py runserver
```

## 🌐 Truy cập

|Trang|URL|
|---|---|
|**Trang chính**|http://127.0.0.1:8000/|
|**Dự đoán nhanh**|http://127.0.0.1:8000/quick-predict/|
|**Dự đoán chi tiết**|http://127.0.0.1:8000/predict/|
|**Lịch sử**|http://127.0.0.1:8000/history/|
|**Thống kê**|http://127.0.0.1:8000/statistics/|
|**Admin Panel**|http://127.0.0.1:8000/admin/|

## 🎯 Tính năng chính

✨ **Dự đoán nhanh** - Chỉ cần nhập tiêu đề  
🔧 **Dự đoán chi tiết** - Cung cấp metadata cho kết quả chính xác hơn  
📊 **Thống kê** - Xem phân tích và biểu đồ  
💾 **Lịch sử** - Lưu tất cả các bài kiểm tra  
🔐 **Admin Panel** - Quản lý dữ liệu

---

**Để dừng server: nhấn Ctrl+C**
