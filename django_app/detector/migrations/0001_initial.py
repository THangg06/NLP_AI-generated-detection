# Generated migration file for detector models

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='NewsArticle',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.TextField(max_length=1000, verbose_name='Tiêu đề')),
                ('tweet_count', models.IntegerField(default=0, verbose_name='Số lượt tweet')),
                ('retweet_count', models.IntegerField(default=0, verbose_name='Số lượt retweet')),
                ('like_count', models.IntegerField(default=0, verbose_name='Số lượt like')),
                ('reply_count', models.IntegerField(default=0, verbose_name='Số lượt reply')),
                ('user_verified', models.BooleanField(default=False, verbose_name='Người dùng được xác minh')),
                ('prediction', models.IntegerField(blank=True, choices=[(0, 'Tin giả'), (1, 'Tin thật')], null=True, verbose_name='Kết quả dự đoán')),
                ('fake_probability', models.FloatField(default=0, help_text='Giá trị từ 0-1', verbose_name='Xác suất tin giả')),
                ('real_probability', models.FloatField(default=0, help_text='Giá trị từ 0-1', verbose_name='Xác suất tin thật')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='Cập nhật')),
            ],
            options={
                'verbose_name': 'Bài viết tin tức',
                'verbose_name_plural': 'Các bài viết tin tức',
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='PredictionHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('prediction_date', models.DateTimeField(auto_now_add=True)),
                ('article', models.ForeignKey(on_delete=models.CASCADE, related_name='history', to='detector.newsarticle')),
            ],
            options={
                'verbose_name': 'Lịch sử dự đoán',
                'verbose_name_plural': 'Lịch sử dự đoán',
                'ordering': ['-prediction_date'],
            },
        ),
    ]
