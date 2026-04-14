#!/usr/bin/env python3
"""
Ensemble Optimizer for Kaggle Playground Series S6E4
Goal: Achieve a score above 0.98215 by refining the ensemble strategy.
"""

import pandas as pd
import numpy as np
from collections import Counter

# Paths to submissions
data_paths = {
    'sub_098200': '/kaggle/input/datasets/gajananbarve/submission-098200/submission.csv',
    'sub_098150': '/kaggle/input/datasets/gajananbarve/submission-098150/submission.csv',
    'sub_098114': '/kaggle/input/datasets/gajananbarve/submission-098114/submission.csv',
}

# Load submissions
def load_submissions(paths):
    submissions = {}
    for name, path in paths.items():
        try:
            submissions[name] = pd.read_csv(path)
            print(f"Loaded {name} with shape {submissions[name].shape}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    return submissions

# Weighted voting with higher scores having more influence
def weighted_vote(submissions, weights=None):
    if not weights:
        weights = {'sub_098200': 0.9, 'sub_098150': 0.07, 'sub_098114': 0.03}
    
    dfs = list(submissions.values())
    final_preds = []
    
    for idx in range(len(dfs[0])):
        preds = [df.loc[idx, 'Irrigation_Need'] for df in dfs]
        weighted_preds = []
        for pred, df in zip(preds, submissions.keys()):
            weighted_preds.extend([pred] * int(weights[df] * 100))
        
        most_common = Counter(weighted_preds).most_common(1)[0][0]
        final_preds.append(most_common)
    
    return final_preds

# Conditional transfer: Use highest-scoring model for uncertain cases
def conditional_transfer(submissions, final_preds, threshold=0.3):
    # Identify uncertain predictions (e.g., where models disagree)
    uncertain_idx = []
    for idx in range(len(final_preds)):
        preds = [df.loc[idx, 'Irrigation_Need'] for df in submissions.values()]
        if len(set(preds)) > 1:  # Disagreement
            uncertain_idx.append(idx)
    
    # Replace uncertain predictions with the highest-scoring model
    highest_score_sub = max(submissions.keys(), key=lambda x: float(x.split('_')[1]))
    for idx in uncertain_idx:
        final_preds[idx] = submissions[highest_score_sub].loc[idx, 'Irrigation_Need']
    
    return final_preds

# Fallback mechanism: Default to highest-scoring model if all disagree
def fallback_to_highest(submissions, final_preds):
    highest_score_sub = max(submissions.keys(), key=lambda x: float(x.split('_')[1]))
    for idx in range(len(final_preds)):
        preds = [df.loc[idx, 'Irrigation_Need'] for df in submissions.values()]
        if len(set(preds)) == len(submissions):  # All models disagree
            final_preds[idx] = submissions[highest_score_sub].loc[idx, 'Irrigation_Need']
    
    return final_preds

# Generate submission file
def generate_submission(final_preds, output_path='submission.csv'):
    sub = pd.read_csv('/kaggle/input/competitions/playground-series-s6e4/sample_submission.csv')
    sub['Irrigation_Need'] = final_preds
    sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == '__main__':
    # Load submissions
    submissions = load_submissions(data_paths)
    
    # Apply weighted voting
    final_preds = weighted_vote(submissions)
    
    # Apply conditional transfer
    final_preds = conditional_transfer(submissions, final_preds)
    
    # Apply fallback mechanism
    final_preds = fallback_to_highest(submissions, final_preds)
    
    # Generate submission
    generate_submission(final_preds)
