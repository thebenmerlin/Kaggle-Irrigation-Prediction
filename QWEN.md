# Predicting-Irrigation-Need

## Project Overview

This is a **Kaggle Playground Series (Season 6, Episode 4)** competition project focused on predicting whether crops need irrigation based on soil, weather, and field characteristics. The dataset contains **630,000 training rows** and **270,000 test rows** with 20 features including soil properties, weather conditions, crop types, and irrigation history.

**Target Variable:** `Irrigation_Need` (categorical: Low, Medium, High)
**Evaluation Metric:** Balanced Accuracy
**Target Performance:** ~0.97+

## Directory Structure

```
Predicting-Irrigation-Need/
├── train.csv                  # Training data (630K rows, 21 columns)
├── test.csv                   # Test data (270K rows, 20 columns)
├── solution.py                # Main ensemble solution script (XGB + LGB + CAT)
├── s6e4-the-most-detailed-eda-100.ipynb  # Comprehensive EDA notebook
├── MEMORY_SAFETY_NOTES.md     # Memory management documentation
├── README.md                  # Project overview
├── Attempt 1/                 # Baseline ensemble (XGBoost + LightGBM)
├── Attempt 2/                 # Enhanced ensemble with balanced class weights
├── Attempt 3/                 # Advanced models with memory management
│   ├── Agri III.ipynb         # 2-level stacking with meta-learner
│   └── Hagridculture.ipynb    # 3-model ensemble with aggressive memory management
├── Attempt 4/                 # Improved approaches
│   ├── Agri IV - The Apex.ipynb
│   ├── agree-iv-latest.ipynb
│   ├── FIXES_APPLIED.md
│   └── QUICK_START.md
├── Attempt 5/                 # Ensemble voting strategies
│   ├── kaggle-grandmaster-ensemble.ipynb
│   └── s6e4-ensemble-voting-strategy-multi-submission.ipynb
├── Attempt 6/                 # Latest approaches
│   ├── ensemble_optimizer.py
│   ├── notebook910e95bae7.ipynb
│   ├── ps-s6e4-eos-voting-2-1-2.ipynb
│   └── s6e4-ensemble-voting-transfer-0-981-lb.ipynb
├── Attempt 6 Submissions.csv/ # Submission results
│   ├── results/
│   └── results-2/
├── catboost_info/             # CatBoost training logs and artifacts
└── venv/                      # Python virtual environment
```

## Key Features

### Feature Engineering (60+ features)
- **Interaction features:** Soil_pH × Soil_Moisture, Temperature × Humidity, etc.
- **Ratio features:** Organic_to_Soil, EC_to_Soil_pH, Rainfall_per_Hectare
- **Domain-specific features:** Water Stress, Crop Water Demand, Evapotranspiration indices
- **Binned features:** Soil Moisture, Rainfall, Temperature quantile bins
- **Group statistics:** Target encoding by Crop_Type, Soil_Type, Region, Season

### Modeling Approaches
- **Ensemble Methods:** XGBoost + LightGBM + CatBoost
- **Cross-Validation:** 5-fold stratified CV
- **Stacking:** 2-level stacking with meta-learner (Attempt 3+)
- **Voting Strategies:** EOS voting, multi-submission ensembles (Attempt 5-6)
- **Memory Management:** Explicit `gc.collect()` after each fold to handle 630K rows on Kaggle's 16GB RAM limit

## Building and Running

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn scipy
```

### Running the Solution
```bash
# Run the main solution script
python solution.py

# Or upload any .ipynb to Kaggle and run all cells
```

### Expected Output
- `submission.csv` - Kaggle submission file
- `oof_predictions.csv` - Out-of-fold predictions for analysis

### Expected Timeline (on Kaggle)
| Step | Time |
|------|------|
| Feature Engineering | ~30-45 sec |
| Each Fold (1-5) | ~3-4 min |
| Ensemble Eval | ~15-30 sec |
| Threshold Optimization | ~30-60 sec |
| **Total** | **~18-22 minutes** |

## Development Conventions

### Memory Management Pattern
All notebooks follow explicit memory cleanup to prevent kernel restarts:
```python
# After each model training:
del model, X_train, X_val, y_train, y_val
gc.collect()
```

### Model Parameters (Default)
- **XGBoost:** max_depth=8, learning_rate=0.05, n_estimators=1000
- **LightGBM:** max_depth=8, learning_rate=0.05, n_estimators=1000
- **CatBoost:** depth=8, learning_rate=0.05, iterations=1000

### Feature Columns
**Categorical:** Soil_Type, Crop_Type, Crop_Growth_Stage, Season, Irrigation_Type, Water_Source, Mulching_Used, Region

**Numerical:** Soil_pH, Soil_Moisture, Organic_Carbon, Electrical_Conductivity, Temperature_C, Humidity, Rainfall_mm, Sunlight_Hours, Wind_Speed_kmh, Field_Area_hectare, Previous_Irrigation_mm

## Key Techniques

1. **Smoothed Target Encoding:** Categorical encoding with high/low probability features
2. **Threshold Optimization:** Post-processing for balanced accuracy (with 90s time limit)
3. **Probability Averaging:** Simple average of XGB/LGB/CAT probabilities (usually best)
4. **Early Stopping:** 100 rounds for LGB and CatBoost
5. **Class Weights:** Balanced class weights for handling class imbalance

## Verification Checklist

Before submitting to Kaggle:
- [ ] All cells execute without errors
- [ ] "✓ Memory freed" appears 15 times (5 folds × 3 models)
- [ ] "Fold X complete" appears 5 times
- [ ] "ALL FOLDS COMPLETE" final message appears
- [ ] submission.csv is generated
- [ ] No kernel restarts in the log

## Notes

- **Agri III.ipynb** (Attempt 3) is documented as the strongest model
- All notebooks include memory safety fixes to prevent kernel restarts on large datasets
- The `solution.py` file is a standalone script that can be run locally or adapted for Kaggle
- CatBoost training logs are preserved in `catboost_info/` for debugging
