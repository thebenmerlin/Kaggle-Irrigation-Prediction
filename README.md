# 🌾 Predicting Irrigation Need

**Kaggle Competition: Playground Series S6E4**

Predicting whether crops need irrigation based on soil, weather, and field characteristics.

## 📁 Notebook Structure

| Folder | Approach |
|--------|----------|
| **Attempt 1** | Baseline ensemble (XGBoost + LightGBM) |
| **Attempt 2** | Enhanced ensemble with balanced class weights |
| **Attempt 3** | Advanced models with memory management |
| - `Agri III.ipynb` | 2-level stacking ensemble with meta-learner, smoothed target encoding, enhanced feature engineering |
| - `Hagridculture.ipynb` | 3-model ensemble with aggressive memory management, threshold optimization |

## 🚀 Key Techniques

- **Feature Engineering:** 60+ features including interactions, ratios, domain-specific indices (Water Stress, Crop Water Demand, Evapotranspiration)
- **Ensemble Methods:** XGBoost + LightGBM + CatBoost
- **Cross-Validation:** 5-fold stratified CV
- **Memory Management:** Explicit cleanup after each fold to handle 630K+ rows on Kaggle
- **Target Encoding:** Smoothed categorical encoding with high/low probability features
- **Post-processing:** Threshold optimization for balanced accuracy

## 📊 Expected Performance

- **Metric:** Balanced Accuracy
- **Target:** ~0.97+

## 📋 Usage

1. Upload any `.ipynb` to Kaggle
2. Run all cells
3. Submit the generated `submission.csv`

**Note:** `Agri III.ipynb` is the strongest model. Both notebooks include memory safety fixes to prevent kernel restarts on large datasets.
