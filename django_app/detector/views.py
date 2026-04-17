from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from .models import NewsArticle, PredictionHistory
from .forms import NewsArticleForm, SimplePredictionForm
from .ml_predictor import get_predictor
import json


def home(request):
    """Trang chính"""
    total_articles = NewsArticle.objects.count()
    fake_count = NewsArticle.objects.filter(prediction=0).count()
    real_count = NewsArticle.objects.filter(prediction=1).count()
    
    context = {
        'total_articles': total_articles,
        'fake_count': fake_count,
        'real_count': real_count,
    }
    return render(request, 'detector/home.html', context)


def predict_page(request):
    """Trang dự đoán"""
    form = NewsArticleForm(request.POST or None)
    result = None
    
    if request.method == 'POST':
        form = NewsArticleForm(request.POST)
        if form.is_valid():
            article = form.save(commit=False)
            
            try:
                # Dự đoán chỉ dựa trên text
                predictor = get_predictor()
                result = predictor.predict(article.title)
                
                # Lưu vào database
                article.prediction = result['prediction']
                article.fake_probability = result['fake_prob']
                article.real_probability = result['real_prob']
                article.save()
                
                # Thêm lịch sử
                PredictionHistory.objects.create(article=article)
                
                messages.success(request, f"✓ Dự đoán thành công! Kết quả: {result['label']}")
                
            except Exception as e:
                messages.error(request, f"❌ Lỗi khi dự đoán: {str(e)}")
    
    articles = NewsArticle.objects.all()[:10]
    
    context = {
        'form': form,
        'result': result,
        'articles': articles
    }
    return render(request, 'detector/predict.html', context)


def quick_predict(request):
    """Trang dự đoán nhanh (chỉ text)"""
    form = SimplePredictionForm(request.POST or None)
    result = None
    
    if request.method == 'POST':
        form = SimplePredictionForm(request.POST)
        if form.is_valid():
            title = form.cleaned_data['title']
            
            try:
                predictor = get_predictor()
                
                # Dự đoán chỉ với text
                result = predictor.predict(title)
                
                # Lưu vào database
                article = NewsArticle(title=title)
                article.prediction = result['prediction']
                article.fake_probability = result['fake_prob']
                article.real_probability = result['real_prob']
                article.save()
                
                PredictionHistory.objects.create(article=article)
                
                messages.success(request, f"✓ Dự đoán thành công!")
                
            except Exception as e:
                messages.error(request, f"❌ Lỗi khi dự đoán: {str(e)}")
    
    context = {
        'form': form,
        'result': result
    }
    return render(request, 'detector/quick_predict.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    """API endpoint cho dự đoán"""
    try:
        data = json.loads(request.body)
        title = data.get('title')
        
        if not title:
            return JsonResponse({'error': 'Title is required'}, status=400)
        
        predictor = get_predictor()
        result = predictor.predict(title)
        
        return JsonResponse(result)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def articles_history(request):
    """Lịch sử các bài viết đã kiểm tra"""
    articles = NewsArticle.objects.all()
    
    # Filter
    prediction_filter = request.GET.get('prediction')
    if prediction_filter and prediction_filter in ['0', '1']:
        articles = articles.filter(prediction=int(prediction_filter))
    
    # Pagination
    from django.core.paginator import Paginator
    paginator = Paginator(articles, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'articles': page_obj.object_list,
        'prediction_filter': prediction_filter
    }
    return render(request, 'detector/articles_history.html', context)


def article_detail(request, pk):
    """Chi tiết bài viết"""
    article = get_object_or_404(NewsArticle, pk=pk)
    history = article.history.all()
    
    context = {
        'article': article,
        'history': history
    }
    return render(request, 'detector/article_detail.html', context)


def statistics(request):
    """Thống kê"""
    total_articles = NewsArticle.objects.count()
    fake_count = NewsArticle.objects.filter(prediction=0).count()
    real_count = NewsArticle.objects.filter(prediction=1).count()
    
    # Tính độ chính xác trung bình
    from django.db.models import Avg
    fake_avg_confidence = NewsArticle.objects.filter(
        prediction=0
    ).aggregate(avg=Avg('fake_probability'))['avg'] or 0
    
    real_avg_confidence = NewsArticle.objects.filter(
        prediction=1
    ).aggregate(avg=Avg('real_probability'))['avg'] or 0
    
    context = {
        'total_articles': total_articles,
        'fake_count': fake_count,
        'real_count': real_count,
        'fake_percentage': (fake_count / total_articles * 100) if total_articles > 0 else 0,
        'real_percentage': (real_count / total_articles * 100) if total_articles > 0 else 0,
        'fake_avg_confidence': round(fake_avg_confidence * 100, 2),
        'real_avg_confidence': round(real_avg_confidence * 100, 2),
    }
    return render(request, 'detector/statistics.html', context)
