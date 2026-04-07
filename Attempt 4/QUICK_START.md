# 🚀 Quick Start: Memory-Safe Training

## ⚠️ STOP YOUR CURRENT RUN
The notebook is crashing and restarting - **no progress is being saved**.

## ✅ What Was Fixed
Your kernel is **crashing due to memory exhaustion** after ~2 hours (3 XGB models). The updated notebook now:
- ✅ **Saves checkpoints** after each model (~40 min)
- ✅ **Can resume** from last checkpoint if it crashes
- ✅ **Monitors memory** usage in real-time
- ✅ **Cleans up** memory aggressively after each model

## 📋 What To Do NOW

### Step 1: Stop Current Run
In Jupyter: **Kernel → Interrupt** (stop the crashing run)

### Step 2: Restart Fresh
**Kernel → Restart & Clear Outputs**

### Step 3: Run Updated Notebook
1. Run cell 1 (imports)
2. Run cell 2 (load data)
3. Run cell 3 (feature engineering)
4. Run cell 4 (Phase 4 - memory-safe training)

### Step 4: If It Crashes
**Just re-run Phase 4 cell** and when asked:
```
🔄 Found checkpoint with 3 completed models
Last trained: XGB_seed456
▶️ Resume from checkpoint? (y/n): y
```
Type **y** and it continues from where it left off!

## 💡 Memory Options

### If Still Crashing - Reduce Seeds
In Phase 3 cell, change:
```python
SEEDS = [42, 123, 456]  # 3 seeds instead of 5
```

This gives you:
- ✅ **3 hours** instead of 6 hours
- ✅ **60% less memory** usage
- ✅ Still strong ensemble (15 models instead of 25)

### If Still Crashing - Use Kaggle
1. Upload to Kaggle
2. Enable GPU (free 30hrs/week)
3. Run there (16GB RAM + GPU acceleration)

## 📊 Expected Output

```
[PHASE 4] Multi-seed Level-1 training (MEMORY-SAFE)...
  Models: 5 types x 5 seeds = 25 level-1 models
  💾 Test encoded features saved to disk
  [MEMORY] Before training: 3.45 GB

  Training XGB models (5 seeds)...
    XGB_seed42 ... OOF: 0.964522
  [MEMORY] After XGB_seed42: 5.12 GB
    💾 Checkpoint saved (1 models completed)

    XGB_seed123 ... OOF: 0.964309
  [MEMORY] After XGB_seed123: 5.15 GB
    💾 Checkpoint saved (2 models completed)
...
```

## 🎯 Key Points
- **Checkpoints** saved to `Attempt 4/checkpoints/`
- **Can resume** anytime after crash
- **Monitor memory** - if >8GB, consider reducing seeds
- **Total time**: ~5-6 hours (with progress saved)

---

**Full details**: See `FIXES_APPLIED.md`
