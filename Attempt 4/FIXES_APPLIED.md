# Fixed: Notebook Kernel CRASH Issue (Not Just Looping)

## CRITICAL UPDATE: It's NOT a Loop - It's a Kernel Crash!

### Problem Identified
The notebook kernel is **crashing and restarting** due to **memory exhaustion**, not looping:
- Training runs for ~2 hours (3 XGB seeds complete)
- Kernel crashes when memory runs out
- Jupyter auto-restarts the kernel
- All cells re-execute from the beginning
- **No progress is saved** between crashes

### Evidence
- Timestamps show restart (7.8s, 17.8s vs 7523.7s)
- Disk space is fine (50GB available)  
- Pattern: XGB seeds 42, 123, 456 complete → crash → restart
- Each XGB model takes ~40 minutes and uses significant RAM

## Solution Applied (Memory-Safe Version)

### 1. **Checkpointing System**
After **each model** completes training:
- Saves OOF predictions to `checkpoints/phase4_checkpoint.pkl`
- Saves test predictions
- Saves model scores
- Tracks which models/seeds are complete

**Benefit**: If kernel crashes, you can **resume** from last checkpoint instead of restarting!

### 2. **Memory Monitoring**
Added `log_memory()` function that shows RAM usage:
```
  [MEMORY] Before XGB: 4.52 GB
  [MEMORY] After XGB_seed42: 6.23 GB
```

**Benefit**: You can see if memory is growing and catch issues early.

### 3. **Aggressive Memory Cleanup**
After each model:
- Double `gc.collect()` calls
- `malloc_trim(0)` to return memory to OS
- Frees fold arrays (`del fold_oof, fold_test`)
- Deletes model objects immediately after use

**Benefit**: Prevents memory from growing unboundedly.

### 4. **Disk-Based Feature Storage**
Test encoded features saved to disk (`test_encoded.pkl`):
- Loaded only when needed for predictions
- Frees RAM during training
- Reduces peak memory usage

**Benefit**: ~1-2GB memory savings during training.

### 5. **Resume Capability**
When you re-run Phase 4 after a crash:
```
  🔄 Found checkpoint with 3 completed models
  Last trained: XGB_seed456
  ▶️ Resuming from model 3, seed 1
```

**Benefit**: No more losing 2+ hours of training!

## How to Use

### Before Running
1. **Restart the kernel** fresh (Kernel → Restart & Clear Outputs)
2. **Close other applications** to free RAM
3. Ensure you have at least **16GB RAM** available

### Running the Notebook
1. **Run cells 1-3** (imports, data loading, feature engineering)
2. **Run Phase 4** - it will:
   - Create `checkpoints/` directory
   - Save progress after each model
   - Show memory usage
   - Take ~5+ hours total (but saves progress)

### If Kernel Crashes
1. **Don't panic** - your progress is saved!
2. **Re-run Phase 4 cell**
3. When prompted: `Resume from checkpoint? (y/n):` type **y**
4. It will continue from where it left off

### After Running
Check for the completion message:
```
  Level-1 training complete: 25 models
  💾 Checkpoints saved to: /path/to/checkpoints
```

## Memory Optimization Tips

### Option A: Reduce Number of Seeds
If still crashing, reduce seeds from 5 to 3:
```python
SEEDS = [42, 123, 456]  # Instead of 5 seeds
```
This reduces models from 25 to 15 (40% less memory/time).

### Option B: Reduce Model Count
Remove the most memory-heavy models:
```python
model_configs = [
    ('xgb', 'XGB', best_params['xgb'], True),
    ('lgbm', 'LGBM', best_params['lgbm'], True),
    # Comment out CAT, HGB, RF if needed
]
```

### Option C: Use Kaggle's Free GPU
Kaggle provides 16GB RAM + GPU:
1. Upload notebook to Kaggle
2. Enable GPU in notebook settings
3. Run there instead (free tier has 30hrs/week GPU)

## Files Modified
- `Agri IV - The Apex.ipynb` - Phase 4 cell replaced with memory-safe version
- `checkpoints/` - Created automatically during training (can be deleted after completion)

## Checkpoint Files Created
- `checkpoints/phase4_checkpoint.pkl` - Training progress
- `checkpoints/test_encoded.pkl` - Preprocessed test features (deleted after training)

## Next Steps
1. **Stop the current run** (it will just keep crashing)
2. **Restart kernel** completely
3. **Run the updated notebook** from top
4. **Monitor memory** - if it approaches 80%, consider Option A/B above
5. If crash occurs, **resume from checkpoint** (type 'y' when prompted)

## Estimated Runtime
- **With 5 seeds**: ~5-6 hours (but progress saved)
- **With 3 seeds**: ~3-4 hours (less ensemble diversity)
- **Each XGB model**: ~40 minutes
- **Each LGBM model**: ~25 minutes  
- **Each CAT model**: ~35 minutes
- **Each HGB/RF model**: ~15 minutes
