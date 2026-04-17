import pandas as pd
import joblib

print("="*70)
print("MODEL DIAGNOSTIC - QUICK CHECK")
print("="*70)

# Check training data size
print("\n[1] TRAINING DATA SIZE")
print("-" * 70)
df_train = pd.read_csv("data/train.csv")
print(f"Training data: {len(df_train)} samples")
print(f"Label distribution:\n{df_train['label'].value_counts()}\n")

# Check test data size
print("[2] TEST DATA SIZE")
print("-" * 70)
df_test = pd.read_csv("data/test.csv")
print(f"Test data: {len(df_test)} samples")
if 'label' in df_test.columns:
    print(f"Label distribution:\n{df_test['label'].value_counts()}\n")

# Check model
print("[3] MODEL INFO")
print("-" * 70)
clf = joblib.load("fake_news_xgboost.pkl")
print(f"Model type: {type(clf)}")
print(f"Number of features expected: {clf.n_features_in_}")
print(f"Booster: {clf.get_booster()}")

# ===== KEY FINDINGS =====
print("\n[4] KEY FINDINGS")
print("-" * 70)
print(f"⚠️  TEST SIZE ALERT: Only {len(df_test)} samples for testing!")
print(f"    - This is too small for reliable accuracy assessment")
print(f"    - Statistical noise is very high with n=20")
print(f"\n⚠️  CLASS IMBALANCE: Check if training data is balanced")
print(f"\n💡 NEXT STEPS:")
print(f"    1. Increase test set size (at least 100-200 samples)")
print(f"    2. Use cross-validation instead of single split")
print(f"    3. Check if model memorized training data (overfit)")
print(f"    4. Visualize misclassified samples to find patterns")

print("\n" + "="*70)
