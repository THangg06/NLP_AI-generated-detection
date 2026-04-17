# 🚀 Django Fake News Detector - Hướng dẫn chạy

Một web application Django để phát hiện tin giả/thật sử dụng model hybrid (TF-IDF + RoBERTa + XGBoost).

## 📋 Yêu cầu

- Python 3.8+
- Virtual Environment (venv hoặc conda)
- Các packages được liệt kê trong `requirements.txt`

## 🔧 Cài đặt

### 1. Kích hoạt Virtual Environment

```bash
# Sử dụng environment hiện tại (fake-news-env)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  # (Windows)
.\fake-news-env\Scripts\Activate.ps1
```

### 2. Cài đặt Django packages

```bash
pip install -r django_app/requirements.txt
```

### 3. Chuyển đến folder Django App

```bash
cd django_app
```

### 4. Tạo database

```bash
python manage.py migrate
```

### 5. Tạo Super User (Admin)

```bash
python manage.py createsuperuser
```

Nhập username, email, password khi được yêu cầu.

### 6. Chạy server

```bash
python manage.py runserver
```

Server sẽ chạy tại `http://127.0.0.1:8000/`

## 🌐 Truy cập

- **Trang chính**: http://127.0.0.1:8000/
- **Dự đoán nhanh**: http://127.0.0.1:8000/quick-predict/
- **Dự đoán chi tiết**: http://127.0.0.1:8000/predict/
- **Lịch sử**: http://127.0.0.1:8000/history/
- **Thống kê**: http://127.0.0.1:8000/statistics/
- **Admin Panel**: http://127.0.0.1:8000/admin/

## 📁 Cấu trúc dự án

```
django_app/
├── manage.py                    # Django management script
├── requirements.txt             # Các package cần thiết
├── db.sqlite3                   # Database (tự tạo)
│
├── fake_news_web/               # Project settings
│   ├── __init__.py
│   ├── settings.py              # Django settings
│   ├── urls.py                  # URL routing
│   ├── wsgi.py                  # WSGI application
│   └── asgi.py                  # ASGI application
│
└── detector/                    # Main Django app
    ├── migrations/              # Database migrations
    ├── templates/               # HTML templates
    │   ├── base.html
    │   └── detector/
    │       ├── home.html
    │       ├── quick_predict.html
    │       ├── predict.html
    │       ├── articles_history.html
    │       ├── article_detail.html
    │       └── statistics.html
    ├── static/                  # CSS, JS, images
    │   └── css/
    │       └── style.css
    ├── __init__.py
    ├── admin.py                 # Admin configuration
    ├── apps.py                  # App configuration
    ├── forms.py                 # Django forms
    ├── models.py                # Database models
    ├── urls.py                  # App URLs
    ├── views.py                 # Views logic
    └── ml_predictor.py          # ML model integration
```

## 🤖 Tích hợp Model ML

File `detector/ml_predictor.py` tích hợp model fake news detection. Nó load:
- `fake_news_xgboost.pkl` (Hybrid model)
- `text_fake_news_xgboost.pkl` (Text-only model)
- `tfidf_vectorizer.pkl`
- `metadata_scaler.pkl`
- RoBERTa model từ Hugging Face

## 📝 Sử dụng

### Dự đoán nhanh
- Nhập tiêu đề bài viết
- Nhấn "Phân tích"
- Xem kết quả ngay

### Dự đoán chi tiết
- Nhập tiêu đề + metadata (tweet_count, retweet_count, like_count, reply_count, user_verified)
- Nhấn "Phân tích"
- Nhận dự đoán chính xác hơn

### API Endpoint
```bash
POST /api/predict/
Content-Type: application/json

{
    "title": "Tiêu đề bài viết",
    "metadata": [tweet_count, retweet_count, like_count, reply_count, user_verified]
}
```

## 🛠️ Troubleshooting

### Lỗi không tìm thấy models
Đảm bảo rằng các file `.pkl` được tạo từ training script nằm trong folder `src/`.

### Lỗi khi import torch/transformers
```bash
pip install torch transformers --upgrade
```

### Lỗi database
```bash
python manage.py makemigrations detector
python manage.py migrate
```

## 📚 Công nghệ sử dụng

- **Django 4.2**: Web framework
- **Bootstrap 5**: UI framework
- **RoBERTa**: Deep learning embeddings
- **TF-IDF**: Text feature extraction
- **XGBoost**: Classification model
- **SQLite**: Database
- **Chart.js**: Data visualization

## 👨‍💻 Phát triển thêm

### Thêm models khác
Chỉnh sửa `detector/ml_predictor.py` để tích hợp models khác.

### Customize UI
Sửa templates trong `detector/templates/detector/` và CSS trong `detector/static/css/style.css`.

### API mở rộng
Thêm endpoints mới trong `detector/urls.py` và `detector/views.py`.

## 📄 License

MIT License - Tự do sử dụng cho mục đích học tập và thương mại.

## 📞 Support

Để báo cáo lỗi hoặc đề xuất cải thiện, vui lòng liên hệ.

---

**Happy coding! 🎉**
