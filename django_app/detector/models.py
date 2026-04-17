from django.db import models
from django.utils import timezone

class NewsArticle(models.Model):
    PREDICTION_CHOICES = [
        (0, 'Tin giả'),
        (1, 'Tin thật'),
    ]
    
    title = models.TextField('Tiêu đề', max_length=1000)
    
    # Metadata fields
    tweet_count = models.IntegerField('Số lượt tweet', default=0)
    retweet_count = models.IntegerField('Số lượt retweet', default=0)
    like_count = models.IntegerField('Số lượt like', default=0)
    reply_count = models.IntegerField('Số lượt reply', default=0)
    user_verified = models.BooleanField('Người dùng được xác minh', default=False)
    
    # Prediction results
    prediction = models.IntegerField('Kết quả dự đoán', choices=PREDICTION_CHOICES, null=True, blank=True)
    fake_probability = models.FloatField('Xác suất tin giả', default=0, help_text='Giá trị từ 0-1')
    real_probability = models.FloatField('Xác suất tin thật', default=0, help_text='Giá trị từ 0-1')
    
    # Timestamps
    created_at = models.DateTimeField('Ngày tạo', auto_now_add=True)
    updated_at = models.DateTimeField('Cập nhật', auto_now=True)
    
    class Meta:
        verbose_name = 'Bài viết tin tức'
        verbose_name_plural = 'Các bài viết tin tức'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title[:50]}... ({self.get_prediction_display()})"
    
    def get_prediction_label(self):
        return dict(self.PREDICTION_CHOICES).get(self.prediction, 'Chưa phân loại')
    
    def get_confidence_score(self):
        if self.prediction == 0:
            return round(self.fake_probability * 100, 2)
        return round(self.real_probability * 100, 2)


class PredictionHistory(models.Model):
    article = models.ForeignKey(NewsArticle, on_delete=models.CASCADE, related_name='history')
    prediction_date = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Lịch sử dự đoán'
        verbose_name_plural = 'Lịch sử dự đoán'
        ordering = ['-prediction_date']
