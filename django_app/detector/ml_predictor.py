import os
import sys
import json
from pathlib import Path
from django.conf import settings

# Thêm path để import từ src folder
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

class FakeNewsPredictor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return

        self._deps_loaded = False
        self._np = None
        self._joblib = None
        self._torch = None
        self._RobertaTokenizer = None
        self._RobertaModel = None
        self._clean_text_for_tfidf = None
        self._clean_text_for_roberta = None
        self._combine_features = None
        self._extra_features = None

        self.device = None
        self.models_ready = False
        self._load_models()
        self._initialized = True

    def _load_dependencies(self):
        """Load optional ML dependencies without crashing Django startup."""
        if self._deps_loaded:
            return

        import numpy as np
        import joblib
        import torch
        from transformers import RobertaTokenizer, RobertaModel
        from preprocessing import clean_text_for_tfidf, clean_text_for_roberta
        from feature_combiner import combine_features, extra_features

        self._np = np
        self._joblib = joblib
        self._torch = torch
        self._RobertaTokenizer = RobertaTokenizer
        self._RobertaModel = RobertaModel
        self._clean_text_for_tfidf = clean_text_for_tfidf
        self._clean_text_for_roberta = clean_text_for_roberta
        self._combine_features = combine_features
        self._extra_features = extra_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._deps_loaded = True
    
    def _load_models(self):
        """Load tất cả models cần thiết"""
        # Đường dẫn đến models (root folder của project)
        # settings.BASE_DIR = django_app folder
        # Lên 1 cấp để tới level chứa src/ folder
        base_path = Path(settings.BASE_DIR).parent
        
        print(f"[DEBUG] Base path: {base_path}")
        print(f"[DEBUG] Base path exists: {base_path.exists()}")
        
        try:
            self._load_dependencies()

            # Load XGBoost model
            fake_model_path = base_path / "fake_news_xgboost.pkl"
            tfidf_path = base_path / "tfidf_vectorizer.pkl"
            
            print(f"[DEBUG] Loading: {fake_model_path}")
            print(f"[DEBUG] File exists: {fake_model_path.exists()}")
            
            self.clf_model = self._joblib.load(str(fake_model_path))
            
            # Load TF-IDF vectorizer
            self.tfidf_vectorizer = self._joblib.load(str(tfidf_path))

            threshold_path = base_path / "rumor_threshold.json"
            self.rumor_threshold = 0.5
            self.best_iteration = None
            if threshold_path.exists():
                with open(threshold_path, "r", encoding="utf-8") as f:
                    threshold_payload = json.load(f)
                    self.rumor_threshold = float(threshold_payload.get("rumor_threshold", 0.5))
                    best_iter_val = threshold_payload.get("best_iteration")
                    if best_iter_val is not None:
                        self.best_iteration = int(best_iter_val)
            
            # Load RoBERTa
            self.tokenizer = self._RobertaTokenizer.from_pretrained("roberta-base")
            self.roberta_model = self._RobertaModel.from_pretrained("roberta-base")
            self.roberta_model.to(self.device)
            self.roberta_model.eval()
            
            self.models_ready = True
            print("✓ Tất cả models đã được load thành công")
            
        except Exception as e:
            self.models_ready = False
            print(f"⚠ Cảnh báo: Models chưa sẵn sàng.")
            print(f"   Chi tiết lỗi: {str(e)}")
            import traceback
            traceback.print_exc()
            print("   Django vẫn có thể chạy, nhưng dự đoán sẽ không hoạt động cho đến khi models được cấu hình.")
    
    def get_roberta_embedding(self, text):
        """Lấy RoBERTa embedding cho text (dùng [CLS] token)"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self._torch.no_grad():
            outputs = self.roberta_model(**inputs)
        
        # ✅ Use [CLS] token (index 0) for classification
        embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding.cpu().numpy().flatten()
    
    def predict(self, title):
        """
        Dự đoán rumor/truth dựa trên text
        
        Args:
            title (str): Tiêu đề bài viết
        
        Returns:
            dict: {
                'prediction': 0 hoặc 1 (Django convention: 0=Fake, 1=Real),
                'label': 'Fake' hoặc 'Real',
                'truth_prob': float,
                'rumor_prob': float,
                'confidence': float
            }
        """
        if not self.models_ready:
            raise RuntimeError("Models chưa được tải. Vui lòng chạy training script trước.")
            
        try:
            # Clean text separately for TF-IDF and RoBERTa
            clean_tfidf = self._clean_text_for_tfidf(title)      # Aggressive
            clean_roberta = self._clean_text_for_roberta(title)  # Minimal
            
            # TF-IDF features (aggressive cleaning)
            tfidf_features = self.tfidf_vectorizer.transform([clean_tfidf]).toarray()
            
            # RoBERTa embeddings (minimal cleaning)
            roberta_features = self._np.array([self.get_roberta_embedding(clean_roberta)])

            # Handcrafted content features
            import pandas as pd
            content_df = pd.DataFrame({'content': [title]})
            engineered = self._extra_features(content_df)
            
            # Combine text features only
            X = self._combine_features(tfidf_features, roberta_features, engineered_features=engineered)
            
            # Predict
            if self.best_iteration is not None and self.best_iteration >= 0:
                probs = self.clf_model.predict_proba(X, iteration_range=(0, self.best_iteration + 1))[0]
            else:
                probs = self.clf_model.predict_proba(X)[0]
            # Model convention: class 1 = rumor, class 0 = truth
            pred_model = int(probs[1] >= self.rumor_threshold)
            
            truth_prob = float(probs[0])
            rumor_prob = float(probs[1])

            # Django/database convention: 0 = fake (rumor), 1 = real (truth)
            pred_db = 0 if pred_model == 1 else 1

            label = "Fake" if pred_db == 0 else "Real"
            confidence = (rumor_prob if pred_db == 0 else truth_prob) * 100
            
            return {
                'prediction': int(pred_db),
                'label': label,
                'truth_prob': round(truth_prob, 4),
                'rumor_prob': round(rumor_prob, 4),
                # Backward-compatible aliases used by existing Django views/models.
                'real_prob': round(truth_prob, 4),
                'fake_prob': round(rumor_prob, 4),
                'confidence': round(confidence, 2)
            }
            
        except Exception as e:
            print(f"❌ Lỗi khi dự đoán: {e}")
            import traceback
            traceback.print_exc()
            raise


# Singleton instance
predictor = None

def get_predictor():
    """Lấy singleton predictor instance"""
    global predictor
    if predictor is None:
        predictor = FakeNewsPredictor()
    return predictor
