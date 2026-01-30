# Model Training Guide

This document explains how the CourtVision shot prediction model is trained, how features are weighted, and how hyperparameters are tuned.

## Overview

CourtVision uses an **XGBoost gradient boosting classifier** to predict the probability of a made shot. The model learns from historical NBA play-by-play data and outputs a probability between 0 and 1.

## Training Pipeline

### Phase 1: Player Skill Ratings

Before training the main model, we calculate **Bayesian-smoothed field goal percentages** for each player:

```
smoothed_fg = (made_shots + prior_weight × league_avg) / (total_shots + prior_weight)
```

**Parameters:**
- `prior_weight = 50`: Equivalent to 50 shots of prior knowledge
- `league_avg`: Calculated from the full dataset (~45%)

This smoothing prevents overfitting to players with few shots. A rookie with 2/3 shooting won't get a 66.7% rating but rather it will be pulled toward the league average.

### Phase 2: Feature Extraction

Features are extracted in three categories:

#### Base Features (14)
| Feature | Type | Description |
|---------|------|-------------|
| `shot_distance` | Continuous | Distance from basket in feet |
| `period` | Ordinal | Game period (1-4, 5+ for OT) |
| `seconds_remaining` | Continuous | Seconds left in period (0-720) |
| `is_three` | Binary | Shot from beyond 3PT line (>23 ft) |
| `is_dunk` | Binary | Dunk attempt |
| `is_layup` | Binary | Layup attempt |
| `is_hook` | Binary | Hook shot |
| `is_tip` | Binary | Tip-in attempt |
| `is_fadeaway` | Binary | Fadeaway shot |
| `is_bank` | Binary | Bank shot |
| `is_alley_oop` | Binary | Alley-oop attempt |
| `is_home` | Binary | Home team shot |
| `is_pullup` | Binary | Pull-up jumper |
| `player_skill_rating` | Continuous | Bayesian-smoothed FG% (0.3-0.6) |

#### CDN Enrichment Features (12)
| Feature | Type | Coverage | Description |
|---------|------|----------|-------------|
| `is_fastbreak` | Binary | ~10% | Fastbreak opportunity |
| `is_second_chance` | Binary | ~12% | Offensive rebound situation |
| `is_from_turnover` | Binary | ~14% | Shot after forcing turnover |
| `is_points_in_paint` | Binary | ~48% | Shot in the paint |
| `desc_pullup` | Binary | ~12% | Pull-up shot descriptor |
| `desc_driving` | Binary | ~10% | Driving to basket |
| `desc_step_back` | Binary | ~4% | Step-back move |
| `desc_fadeaway` | Binary | ~3% | Fadeaway move |
| `desc_running` | Binary | ~5% | Running shot |
| `desc_floating` | Binary | ~6% | Floater |
| `desc_turnaround` | Binary | ~3% | Turnaround shot |
| `desc_cutting` | Binary | ~4% | Cutting to basket |

#### Player Physical Features (5)
| Feature | Type | Description |
|---------|------|-------------|
| `height_inches` | Continuous | Player height (66-90 inches) |
| `weight` | Continuous | Player weight (160-300 lbs) |
| `is_guard` | Binary | Guard position |
| `is_forward` | Binary | Forward position |
| `is_center` | Binary | Center position |

#### Injury Context Features (5)
| Feature | Type | Description |
|---------|------|-------------|
| `is_returning_from_injury` | Binary | Was "Out" within last 14 days |
| `days_since_injury` | Continuous | Days since last "Out" status (capped at 365) |
| `had_leg_injury` | Binary | Last injury was leg-related |
| `had_shooting_injury` | Binary | Last injury was arm/hand related |
| `is_playing_hurt` | Binary | Listed as Probable/Questionable on game day |

### Phase 3: Model Training

#### Train/Test Split
- **80% training**, **20% test** (held out for evaluation)
- Stratified split to maintain class balance
- Random seed: 42 for reproducibility

#### XGBoost Configuration

```python
XGBClassifier(
    n_estimators=100,        # Max trees (with early stopping)
    max_depth=6,             # Tree depth limit
    learning_rate=0.1,       # Step size shrinkage
    subsample=0.8,           # Row sampling per tree
    colsample_bytree=0.8,    # Feature sampling per tree
    scale_pos_weight=...,    # Auto-calculated for class balance
    early_stopping_rounds=10 # Stop if no improvement
)
```

## Hyperparameter Tuning

### Current Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 100 | Capped with early stopping |
| `max_depth` | 6 | Prevents overfitting; captures feature interactions |
| `learning_rate` | 0.1 | Standard; lower values need more trees |
| `subsample` | 0.8 | Reduces variance, prevents overfitting |
| `colsample_bytree` | 0.8 | Feature bagging for regularization |
| `early_stopping_rounds` | 10 | Stops training when validation loss plateaus |

### Class Imbalance Handling

The dataset has slightly more misses than makes (~55% miss, ~45% make). We use `scale_pos_weight`:

```python
scale_pos_weight = count(misses) / count(makes)
```

This tells XGBoost to weight made shots more heavily during training.

### Tuning Process

To tune hyperparameters, you can use grid search or Bayesian optimization:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.7, 0.8, 0.9],
}

grid_search = GridSearchCV(
    XGBClassifier(early_stopping_rounds=10),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
```

## Feature Importance

XGBoost automatically calculates feature importance based on:
1. **Gain**: Average improvement in loss when feature is used
2. **Cover**: Average number of samples affected
3. **Frequency**: How often the feature is used in splits

Top features (typical ranking):
1. `shot_distance` - Most predictive single feature
2. `player_skill_rating` - Captures shooter ability
3. `is_dunk` / `is_layup` - High-percentage shots
4. `is_three` - Three-point attempts
5. `period` / `seconds_remaining` - Game context

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | % correct predictions | >62% |
| **Log Loss** | Cross-entropy loss | <0.67 |
| **Brier Score** | Calibration measure | <0.24 |
| **ROC AUC** | Discrimination ability | >0.63 |

### Interpreting Results

- **Accuracy** is simple but not ideal for probability predictions
- **Log Loss** penalizes confident wrong predictions heavily
- **Brier Score** measures how well probabilities are calibrated
- **ROC AUC** measures ability to rank shots by probability

## Retraining

To retrain the model with new data or features:

```bash
python src/retrain.py
```

This will:
1. Load player skill ratings (or recalculate)
2. Load CDN enrichment data (if available)
3. Load player metadata (if available)
4. Extract all features
5. Train XGBoost with early stopping
6. Save artifacts: `model.pkl`, `player_skills.pkl`, `feature_names.pkl`

## Model Artifacts

| File | Purpose | Size |
|------|---------|------|
| `model.pkl` | Trained XGBoost model | ~2 MB |
| `player_skills.pkl` | Player FG% lookup | ~200 KB |
| `feature_names.pkl` | Feature ordering | <1 KB |
| `player_physical.pkl` | Height/weight/position | ~100 KB |

## Performance History

| Version | Features | Accuracy | ROC AUC |
|---------|----------|----------|---------|
| v1: Logistic Regression | 1 (distance) | ~60% | ~0.60 |
| v2: XGBoost Base | 14 | 62.67% | 0.6330 |
| v3: XGBoost + CDN | 26 | 62.70% | 0.6348 |
| v4: XGBoost + Physical + Injury | 36 | - | 0.6672 |

## Future Improvements

### Short-term
- [ ] Cross-validation for more robust estimates
- [ ] Hyperparameter tuning with Optuna
- [ ] Feature selection to remove low-importance features

### Long-term
- [ ] Deep learning for sequence modeling (shot attempts in context)
- [ ] Defender distance (if data becomes available)
- [ ] Real-time model updates during games
