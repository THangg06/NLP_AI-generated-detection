from django import forms
from .models import NewsArticle


class NewsArticleForm(forms.ModelForm):
    class Meta:
        model = NewsArticle
        fields = ['title']
        widgets = {
            'title': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Nhập tiêu đề bài viết tin tức...',
                'required': True
            }),
        }
        labels = {
            'title': 'Tiêu đề bài viết',
        }


class SimplePredictionForm(forms.Form):
    """Form đơn giản chỉ nhập text"""
    title = forms.CharField(
        label='Tiêu đề bài viết',
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': 'Nhập tiêu đề bài viết để kiểm tra...'
        })
    )
