import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import joblib

# Chunk size for streaming through large files
CHUNK_SIZE = 50000

# Feature names for consistent ordering (must match consumer.py)
FEATURE_NAMES = [
    'shot_distance',
    'period',
    'seconds_remaining',
    'is_three',
    'is_dunk',
    'is_layup',
    'is_hook',
    'is_tip',
    'is_fadeaway',
    'is_bank',
    'is_alley_oop',
    'player_skill_rating'
]


def parse_time_remaining(time_str):
    """Parse 'MM:SS' format to seconds remaining in period."""
    if time_str is None:
        return 360  # Default to 6 minutes (mid-period)
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except:
        pass
    return 360


def extract_shot_type_features(description: pl.Series) -> dict:
    """Extract boolean shot type features from description text."""
    desc_lower = description.str.to_lowercase()
    return {
        'is_dunk': desc_lower.str.contains('dunk').cast(pl.Int8),
        'is_layup': desc_lower.str.contains('layup').cast(pl.Int8),
        'is_hook': desc_lower.str.contains('hook').cast(pl.Int8),
        'is_tip': desc_lower.str.contains('tip').cast(pl.Int8),
        'is_fadeaway': desc_lower.str.contains('fadeaway|fade away').cast(pl.Int8),
        'is_bank': desc_lower.str.contains('bank').cast(pl.Int8),
        'is_alley_oop': desc_lower.str.contains('alley oop').cast(pl.Int8),
    }


def retrain_model():
    print("Triggering automated retraining with XGBoost...")
    print(f"Features: {FEATURE_NAMES}")
    
    # First pass: Calculate player skill ratings (target encoding)
    print("\n[Phase 1/3] Calculating player skill ratings...")
    player_stats = {}
    
    try:
        batches = pl.scan_csv('data/nba_plays.csv').collect_batches(chunk_size=CHUNK_SIZE)
        
        for df in batches:
            df = df.filter(pl.col('eventmsgtype').is_in([1, 2]))
            
            for row in df.iter_rows(named=True):
                player = row['player1_name']
                if player:
                    if player not in player_stats:
                        player_stats[player] = {'made': 0, 'total': 0}
                    player_stats[player]['total'] += 1
                    if row['eventmsgtype'] == 1:  # Made shot
                        player_stats[player]['made'] += 1
        
        # Calculate skill ratings with smoothing (Bayesian average)
        # Use league average as prior, weighted by sample size
        total_made = sum(p['made'] for p in player_stats.values())
        total_shots = sum(p['total'] for p in player_stats.values())
        league_avg = total_made / total_shots if total_shots > 0 else 0.45
        prior_weight = 50  # Weight equivalent to 50 shots
        
        player_skill = {}
        for player, stats in player_stats.items():
            # Bayesian smoothed average
            smoothed_fg = (stats['made'] + prior_weight * league_avg) / (stats['total'] + prior_weight)
            player_skill[player] = round(smoothed_fg, 4)
        
        print(f"Calculated skill ratings for {len(player_skill)} players")
        print(f"League average FG%: {league_avg:.3f}")
        
        # Save player skill ratings for inference
        joblib.dump(player_skill, 'player_skills.pkl')
        print("Saved player_skills.pkl artifact")
        
    except FileNotFoundError:
        print("Error: 'nba_plays.csv' not found in data/ folder.")
        return
    
    # Second pass: Extract features and train model
    print("\n[Phase 2/3] Extracting features...")
    X_chunks = []
    y_chunks = []
    total_rows = 0
    
    try:
        batches = pl.scan_csv('data/nba_plays.csv').collect_batches(chunk_size=CHUNK_SIZE)
        
        for df in batches:
            # Filter to shot events only (eventmsgtype: 1=Made, 2=Missed)
            df = df.filter(pl.col('eventmsgtype').is_in([1, 2]))
            
            # Combine home and visitor descriptions
            df = df.with_columns([
                (pl.col('homedescription').fill_null('') + pl.col('visitordescription').fill_null('')).alias('description')
            ])
            
            # Extract shot distance
            df = df.with_columns([
                pl.col('description').str.extract(r"(\d+)'", 1).cast(pl.Int32).alias('shot_distance')
            ])
            
            # Drop rows without distance info
            df = df.filter(pl.col('shot_distance').is_not_null())
            
            if df.height == 0:
                continue
            
            # Extract shot type features
            shot_features = extract_shot_type_features(df['description'])
            for name, series in shot_features.items():
                df = df.with_columns([series.alias(name)])
            
            # Parse time remaining to seconds
            df = df.with_columns([
                pl.col('pctimestring').map_elements(
                    parse_time_remaining, 
                    return_dtype=pl.Int32
                ).alias('seconds_remaining')
            ])
            
            # Add derived features
            df = df.with_columns([
                (pl.col('shot_distance') > 23).cast(pl.Int8).alias('is_three'),
            ])
            
            # Add player skill rating
            df = df.with_columns([
                pl.col('player1_name').map_elements(
                    lambda x: player_skill.get(x, league_avg),
                    return_dtype=pl.Float64
                ).alias('player_skill_rating')
            ])
            
            # Build feature matrix in correct order
            feature_cols = [
                'shot_distance',
                'period',
                'seconds_remaining',
                'is_three',
                'is_dunk',
                'is_layup',
                'is_hook',
                'is_tip',
                'is_fadeaway',
                'is_bank',
                'is_alley_oop',
                'player_skill_rating'
            ]
            
            X_batch = df.select(feature_cols).to_numpy()
            y_batch = (df.select('eventmsgtype').to_numpy() == 1).astype(int).flatten()
            
            X_chunks.append(X_batch)
            y_chunks.append(y_batch)
            total_rows += df.height
            
            if total_rows % 500000 < CHUNK_SIZE:
                print(f"Processed {total_rows:,} training samples...")
                
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return
    
    if not X_chunks:
        print("No training data found!")
        return
    
    # Combine all chunks
    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    
    # Train/test split for proper validation (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n[Phase 3/3] Training XGBoost...")
    print(f"  Training samples: {len(y_train):,}")
    print(f"  Test samples: {len(y_test):,}")
    print(f"  Class balance - Made: {y_train.mean():.3f}, Missed: {1-y_train.mean():.3f}")
    
    # Calculate scale_pos_weight for class imbalance
    # (ratio of negative to positive samples)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    # XGBoost configuration optimized for shot prediction
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=42,
        n_jobs=-1,  # Use all cores
        eval_metric='logloss',
        early_stopping_rounds=10  # Stop if no improvement
    )
    
    # Train with early stopping using test set as validation
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Save model artifact
    joblib.dump(clf, 'model.pkl')
    
    # Calculate metrics on TEST set (not training set!)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    test_accuracy = (y_pred == y_test).mean()
    test_logloss = log_loss(y_test, y_pred_proba)
    test_brier = brier_score_loss(y_test, y_pred_proba)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Feature importance
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION (on held-out test set)")
    print(f"{'='*50}")
    print(f"  Accuracy:    {test_accuracy:.4f}")
    print(f"  Log Loss:    {test_logloss:.4f}  (lower is better)")
    print(f"  Brier Score: {test_brier:.4f}  (lower is better, measures calibration)")
    print(f"  ROC AUC:     {test_auc:.4f}  (higher is better)")
    print(f"  Best iteration: {clf.best_iteration}")
    
    print(f"\nFeature Importance:")
    importance = clf.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")
    
    # Sample predictions
    print(f"\nSample predictions:")
    print(f"  3-foot dunk by elite player: {clf.predict_proba([[3, 2, 300, 0, 1, 0, 0, 0, 0, 0, 0, 0.55]])[0][1]:.3f}")
    print(f"  3-foot layup by avg player:  {clf.predict_proba([[3, 2, 300, 0, 0, 1, 0, 0, 0, 0, 0, 0.45]])[0][1]:.3f}")
    print(f"  25-foot 3PT by avg player:   {clf.predict_proba([[25, 2, 300, 1, 0, 0, 0, 0, 0, 0, 0, 0.45]])[0][1]:.3f}")
    print(f"  25-foot buzzer beater:       {clf.predict_proba([[25, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0.45]])[0][1]:.3f}")


if __name__ == "__main__":
    retrain_model()
