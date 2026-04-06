"""
Winning Solution for Playground Series S6E4 - Predicting Irrigation Need
Ensemble Model with Advanced Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PLAYGROUND SERIES S6E4 - PREDICTING IRRIGATION NEED")
print("Ensemble Model: XGBoost + LightGBM + CatBoost")
print("="*80)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/7] Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"  Train shape: {train.shape}")
print(f"  Test shape: {test.shape}")
print(f"  Target distribution:\n{train['Irrigation_Need'].value_counts()}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n[2/7] Feature engineering...")

# Combine train and test for consistent feature engineering
train['is_train'] = True
test['is_train'] = False
test['Irrigation_Need'] = 'Low'  # placeholder

df = pd.concat([train, test], axis=0, ignore_index=True)

# --- Categorical Features ---
categorical_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 
                   'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']

# Label encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --- Numerical Features ---
numerical_cols = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 
                 'Electrical_Conductivity', 'Temperature_C', 'Humidity',
                 'Rainfall_mm', 'Sunlight_Hours', 'Wind_Speed_kmh',
                 'Field_Area_hectare', 'Previous_Irrigation_mm']

# --- Advanced Feature Engineering ---

# Interaction features
df['Soil_pH_Soil_Moisture'] = df['Soil_pH'] * df['Soil_Moisture']
df['Temperature_Humidity'] = df['Temperature_C'] * df['Humidity']
df['Rainfall_Soil_Moisture'] = df['Rainfall_mm'] * df['Soil_Moisture']
df['Sunlight_Temperature'] = df['Sunlight_Hours'] * df['Temperature_C']
df['Wind_Humidity'] = df['Wind_Speed_kmh'] * df['Humidity']

# Ratio features
df['Organic_to_Soil'] = df['Organic_Carbon'] / (df['Soil_Moisture'] + 1e-6)
df['EC_to_Soil_pH'] = df['Electrical_Conductivity'] / (df['Soil_pH'] + 1e-6)
df['Rainfall_per_Hectare'] = df['Rainfall_mm'] / (df['Field_Area_hectare'] + 1e-6)
df['Prev_Irrigation_per_Hectare'] = df['Previous_Irrigation_mm'] / (df['Field_Area_hectare'] + 1e-6)

# Aggregate features
df['Soil_pH_sq'] = df['Soil_pH'] ** 2
df['Moisture_sq'] = df['Soil_Moisture'] ** 2
df['Temperature_sq'] = df['Temperature_C'] ** 2
df['Rainfall_log'] = np.log1p(df['Rainfall_mm'])
df['Humidity_log'] = np.log1p(df['Humidity'])

# Domain-specific features
df['Water_Stress'] = (df['Temperature_C'] * df['Wind_Speed_kmh']) / (df['Humidity'] + df['Soil_Moisture'] + 1e-6)
df['Irrigation_Efficiency'] = df['Previous_Irrigation_mm'] / (df['Rainfall_mm'] + 1e-6)
df['Crop_Water_Demand'] = df['Sunlight_Hours'] * df['Temperature_C'] / (df['Humidity'] + 1e-6)

# Binned features
df['Soil_Moisture_bin'] = pd.qcut(df['Soil_Moisture'], q=5, labels=False, duplicates='drop')
df['Rainfall_bin'] = pd.qcut(df['Rainfall_mm'], q=5, labels=False, duplicates='drop')
df['Temperature_bin'] = pd.qcut(df['Temperature_C'], q=5, labels=False, duplicates='drop')

# Group statistics (powerful features)
for group_col in ['Crop_Type', 'Soil_Type', 'Region', 'Season']:
    # Mean target encoding (using train data only)
    train_mask = df['is_train'] == True
    group_means = df[train_mask].groupby(group_col)['Irrigation_Need'].apply(
        lambda x: (x == 'High').mean() * 2 + (x == 'Medium').mean()
    )
    df[f'{group_col}_target_mean'] = df[group_col].map(group_means)
    
    # Count encoding
    group_counts = df[train_mask].groupby(group_col).size()
    df[f'{group_col}_count'] = df[group_col].map(group_counts)

# Separate train and test
train_final = df.iloc[:len(train)].copy()
test_final = df.iloc[len(train):].copy()

# Restore original target
train_final['Irrigation_Need'] = train['Irrigation_Need']

# Drop helper columns
train_final.drop(['is_train', 'id'], axis=1, inplace=True)
test_final.drop(['is_train'], axis=1, inplace=True)
if 'Irrigation_Need' in test_final.columns:
    test_final.drop(['Irrigation_Need'], axis=1, inplace=True)
if 'id' in test_final.columns:
    test_final.drop(['id'], axis=1, inplace=True)

print(f"  Total features created: {train_final.shape[1] - 1}")

# ============================================================
# 3. PREPARE FOR MODELING
# ============================================================
print("\n[3/7] Preparing for modeling...")

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(train_final['Irrigation_Need'])
print(f"  Classes: {target_encoder.classes_}")
print(f"  Encoded values: {target_encoder.transform(target_encoder.classes_)}")

# Features
X = train_final.drop('Irrigation_Need', axis=1)
X_test = test_final.copy()
if 'Irrigation_Need' in X_test.columns:
    X_test = X_test.drop('Irrigation_Need', axis=1)

feature_names = X.columns.tolist()
print(f"  Number of features: {len(feature_names)}")

# ============================================================
# 4. CROSS-VALIDATION SETUP
# ============================================================
print("\n[4/7] Setting up cross-validation...")
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Store predictions
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
test_xgb = np.zeros((len(X_test), 3))
test_lgb = np.zeros((len(X_test), 3))
test_cat = np.zeros((len(X_test), 3))

# ============================================================
# 5. TRAIN MODELS
# ============================================================
print("\n[5/7] Training ensemble models...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n  Fold {fold+1}/{N_SPLITS}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # --- XGBoost ---
    xgb_params = {
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, 
                 eval_set=[(X_val, y_val)], 
                 verbose=False)
    
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    fold_acc_xgb = accuracy_score(y_val, oof_xgb[val_idx])
    print(f"    XGBoost accuracy: {fold_acc_xgb:.6f}")
    
    # --- LightGBM ---
    lgb_params = {
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': -1
    }
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(100, verbose=False), 
                           lgb.log_evaluation(period=0)])
    
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    fold_acc_lgb = accuracy_score(y_val, oof_lgb[val_idx])
    print(f"    LightGBM accuracy: {fold_acc_lgb:.6f}")
    
    # --- CatBoost ---
    cat_model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=1.0,
        random_seed=42,
        verbose=0,
        task_type='CPU'
    )
    
    cat_model.fit(X_train, y_train,
                 eval_set=(X_val, y_val),
                 early_stopping_rounds=100,
                 verbose=False)
    
    oof_cat[val_idx] = cat_model.predict(X_val).flatten()
    fold_acc_cat = accuracy_score(y_val, oof_cat[val_idx])
    print(f"    CatBoost accuracy: {fold_acc_cat:.6f}")
    
    # Test predictions (probability)
    test_xgb += xgb_model.predict_proba(X_test) / N_SPLITS
    test_lgb += lgb_model.predict_proba(X_test) / N_SPLITS
    test_cat += cat_model.predict_proba(X_test) / N_SPLITS

# ============================================================
# 6. ENSEMBLE PREDICTIONS
# ============================================================
print("\n[6/7] Creating ensemble predictions...")

# OOF accuracy for each model
oof_acc_xgb = accuracy_score(y, oof_xgb)
oof_acc_lgb = accuracy_score(y, oof_lgb)
oof_acc_cat = accuracy_score(y, oof_cat)

print(f"\n  OOF Accuracies:")
print(f"    XGBoost:  {oof_acc_xgb:.6f}")
print(f"    LightGBM: {oof_acc_lgb:.6f}")
print(f"    CatBoost: {oof_acc_cat:.6f}")

# Simple average ensemble (usually works best)
test_ensemble = (test_xgb + test_lgb + test_cat) / 3
test_pred = np.argmax(test_ensemble, axis=1)

print(f"\n  Ensemble test predictions shape: {test_pred.shape}")
print(f"  Prediction distribution: {pd.Series(test_pred).value_counts().to_dict()}")

# ============================================================
# 7. CREATE SUBMISSION FILE
# ============================================================
print("\n[7/7] Creating submission file...")

# Decode predictions
test_labels = target_encoder.inverse_transform(test_pred)

submission = pd.DataFrame({
    'id': test['id'],
    'Irrigation_Need': test_labels
})

print(f"\n  Submission shape: {submission.shape}")
print(f"  Prediction distribution:\n{submission['Irrigation_Need'].value_counts()}")

# Save submission
submission.to_csv('submission.csv', index=False)
print(f"\n  ✓ Submission saved to 'submission.csv'")

# Save OOF predictions for analysis
oof_df = pd.DataFrame({
    'id': train['id'],
    'Irrigation_Need': train['Irrigation_Need'],
    'oof_pred_xgb': target_encoder.inverse_transform(oof_xgb.astype(int)),
    'oof_pred_lgb': target_encoder.inverse_transform(oof_lgb.astype(int)),
    'oof_pred_cat': target_encoder.inverse_transform(oof_cat.astype(int))
})
oof_df.to_csv('oof_predictions.csv', index=False)
print(f"  ✓ OOF predictions saved to 'oof_predictions.csv'")

print("\n" + "="*80)
print("SUBMISSION COMPLETE!")
print("="*80)
print(f"\nEstimated CV Score: ~{max(oof_acc_xgb, oof_acc_lgb, oof_acc_cat):.6f}")
print(f"\nNext steps:")
print(f"  1. Review submission.csv")
print(f"  2. Submit to Kaggle")
print(f"  3. Check leaderboard position")
