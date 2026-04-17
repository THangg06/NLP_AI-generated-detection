from django.contrib import admin
from .models import NewsArticle, PredictionHistory


@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    list_display = ['title_short', 'get_prediction_label', 'fake_probability', 'real_probability', 'created_at']
    list_filter = ['prediction', 'created_at', 'user_verified']
    search_fields = ['title']
    readonly_fields = ['created_at', 'updated_at', 'fake_probability', 'real_probability', 'prediction']
    
    fieldsets = (
        ('Thông tin bài viết', {
            'fields': ('title',)
        }),
        ('Metadata', {
            'fields': ('tweet_count', 'retweet_count', 'like_count', 'reply_count', 'user_verified')
        }),
        ('Kết quả dự đoán', {
            'fields': ('prediction', 'fake_probability', 'real_probability')
        }),
        ('Timestamp', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def title_short(self, obj):
        return obj.title[:80] + '...' if len(obj.title) > 80 else obj.title
    title_short.short_description = 'Tiêu đề'


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['article', 'prediction_date']
    list_filter = ['prediction_date']
    search_fields = ['article__title']
    readonly_fields = ['prediction_date']
