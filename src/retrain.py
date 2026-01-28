import polars as pl
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import joblib

# Chunk size for streaming through large files
CHUNK_SIZE = 50000

# CDN enriched data path
ENRICHED_DATA_PATH = Path('data/enriched_shots.parquet')

# Feature names for consistent ordering (must match consumer.py)
# Base features from CSV
BASE_FEATURE_NAMES = [
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
    'is_home',
    'is_pullup',
    'player_skill_rating'
]

# Additional features from CDN enrichment
CDN_FEATURE_NAMES = [
    'is_fastbreak',
    'is_second_chance',
    'is_from_turnover',
    'is_points_in_paint',
    # Descriptor one-hot encoded
    'desc_pullup',
    'desc_driving',
    'desc_step_back',
    'desc_fadeaway',
    'desc_running',
    'desc_floating',
    'desc_turnaround',
    'desc_cutting',
]

# Combined features (used when enriched data available)
FEATURE_NAMES = BASE_FEATURE_NAMES  # Will be updated if CDN data available


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
        'is_pullup': desc_lower.str.contains('pull-up|pullup|pull up').cast(pl.Int8),
    }


def load_cdn_enrichment() -> pl.DataFrame | None:
    """Load CDN enriched data if available."""
    if not ENRICHED_DATA_PATH.exists():
        return None
    
    print(f"  Loading CDN enriched data from {ENRICHED_DATA_PATH}...")
    df = pl.read_parquet(ENRICHED_DATA_PATH)
    
    # One-hot encode descriptor
    descriptor_map = {
        'pullup': 'desc_pullup',
        'driving': 'desc_driving',
        'step back': 'desc_step_back',
        'fadeaway': 'desc_fadeaway',
        'turnaround fadeaway': 'desc_fadeaway',
        'running': 'desc_running',
        'floating': 'desc_floating',
        'driving floating': 'desc_floating',
        'turnaround': 'desc_turnaround',
        'cutting': 'desc_cutting',
    }
    
    for desc_col in ['desc_pullup', 'desc_driving', 'desc_step_back', 'desc_fadeaway',
                      'desc_running', 'desc_floating', 'desc_turnaround', 'desc_cutting']:
        df = df.with_columns(pl.lit(0).alias(desc_col).cast(pl.Int8))
    
    # Map descriptors to columns
    for desc_value, col_name in descriptor_map.items():
        df = df.with_columns(
            pl.when(pl.col('descriptor') == desc_value)
            .then(pl.lit(1))
            .otherwise(pl.col(col_name))
            .alias(col_name)
            .cast(pl.Int8)
        )
    
    # Convert boolean qualifiers to int
    for col in ['is_fastbreak', 'is_second_chance', 'is_from_turnover', 'is_points_in_paint']:
        df = df.with_columns(pl.col(col).cast(pl.Int8))
    
    print(f"  Loaded {len(df):,} enriched shots")
    return df


def retrain_model():
    global FEATURE_NAMES
    
    print("Triggering automated retraining with XGBoost...")
    
    # Check for CDN enriched data
    cdn_data = load_cdn_enrichment()
    use_cdn_features = cdn_data is not None
    
    if use_cdn_features:
        FEATURE_NAMES = BASE_FEATURE_NAMES + CDN_FEATURE_NAMES
        print(f"Using CDN-enriched features ({len(FEATURE_NAMES)} total)")
    else:
        FEATURE_NAMES = BASE_FEATURE_NAMES
        print("CDN data not found, using base features only")
    
    print(f"Features: {FEATURE_NAMES}")
    
    # First pass: Calculate player skill ratings
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
                    if row['eventmsgtype'] == 1:
                        player_stats[player]['made'] += 1
        
        total_made = sum(p['made'] for p in player_stats.values())
        total_shots = sum(p['total'] for p in player_stats.values())
        league_avg = total_made / total_shots if total_shots > 0 else 0.45
        prior_weight = 50
        
        player_skill = {}
        for player, stats in player_stats.items():
            smoothed_fg = (stats['made'] + prior_weight * league_avg) / (stats['total'] + prior_weight)
            player_skill[player] = round(smoothed_fg, 4)
        
        print(f"Calculated skill ratings for {len(player_skill)} players")
        print(f"League average FG%: {league_avg:.3f}")
        
        joblib.dump(player_skill, 'player_skills.pkl')
        print("Saved player_skills.pkl artifact")
        
    except FileNotFoundError:
        print("Error: 'nba_plays.csv' not found in data/ folder.")
        return
    
    # Second pass: Extract features
    print("\n[Phase 2/3] Extracting features...")
    X_chunks = []
    y_chunks = []
    total_rows = 0
    
    # Build CDN lookup
    cdn_lookup = {}
    if use_cdn_features and cdn_data is not None:
        print("  Building CDN feature lookup...")
        cdn_feature_cols = CDN_FEATURE_NAMES
        for row in cdn_data.iter_rows(named=True):
            key = (row['game_id'], row['player_id'], row['period'])
            if key not in cdn_lookup:
                cdn_lookup[key] = []
            cdn_lookup[key].append({col: row[col] for col in cdn_feature_cols})
        print(f"  Built lookup with {len(cdn_lookup):,} unique (game, player, period) combinations")
    
    try:
        batches = pl.scan_csv('data/nba_plays.csv').collect_batches(chunk_size=CHUNK_SIZE)
        
        for df in batches:
            df = df.filter(pl.col('eventmsgtype').is_in([1, 2]))
            
            df = df.with_columns([
                (pl.col('homedescription').fill_null('') + pl.col('visitordescription').fill_null('')).alias('description')
            ])
            
            df = df.with_columns([
                pl.col('description').str.extract(r"(\d+)'", 1).cast(pl.Int32).alias('shot_distance')
            ])
            
            df = df.filter(pl.col('shot_distance').is_not_null())
            
            if df.height == 0:
                continue
            
            shot_features = extract_shot_type_features(df['description'])
            for name, series in shot_features.items():
                df = df.with_columns([series.alias(name)])
            
            df = df.with_columns([
                pl.col('pctimestring').map_elements(
                    parse_time_remaining, 
                    return_dtype=pl.Int32
                ).alias('seconds_remaining')
            ])
            
            df = df.with_columns([
                (pl.col('shot_distance') > 23).cast(pl.Int8).alias('is_three'),
            ])
            
            df = df.with_columns([
                (pl.col('homedescription').is_not_null() & (pl.col('homedescription') != '')).cast(pl.Int8).alias('is_home')
            ])
            
            df = df.with_columns([
                pl.col('player1_name').map_elements(
                    lambda x: player_skill.get(x, league_avg),
                    return_dtype=pl.Float64
                ).alias('player_skill_rating')
            ])
            
            base_feature_cols = [
                'shot_distance', 'period', 'seconds_remaining', 'is_three',
                'is_dunk', 'is_layup', 'is_hook', 'is_tip', 'is_fadeaway',
                'is_bank', 'is_alley_oop', 'is_home', 'is_pullup', 'player_skill_rating'
            ]
            
            X_base = df.select(base_feature_cols).to_numpy()
            y_batch = (df.select('eventmsgtype').to_numpy() == 1).astype(int).flatten()
            
            if use_cdn_features:
                cdn_features = np.zeros((df.height, len(CDN_FEATURE_NAMES)), dtype=np.float32)
                
                for i, row in enumerate(df.iter_rows(named=True)):
                    game_id = row.get('game_id')
                    player_id = row.get('player1_id')
                    period = row.get('period')
                    
                    key = (game_id, player_id, period)
                    if key in cdn_lookup and cdn_lookup[key]:
                        cdn_row = cdn_lookup[key].pop(0)
                        for j, col in enumerate(CDN_FEATURE_NAMES):
                            cdn_features[i, j] = cdn_row.get(col, 0)
                
                X_batch = np.hstack([X_base, cdn_features])
            else:
                X_batch = X_base
            
            X_chunks.append(X_batch)
            y_chunks.append(y_batch)
            total_rows += df.height
            
            if total_rows % 500000 < CHUNK_SIZE:
                print(f"Processed {total_rows:,} training samples...")
                
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not X_chunks:
        print("No training data found!")
        return
    
    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n[Phase 3/3] Training XGBoost...")
    print(f"  Training samples: {len(y_train):,}")
    print(f"  Test samples: {len(y_test):,}")
    print(f"  Feature count: {X_train.shape[1]}")
    print(f"  Class balance - Made: {y_train.mean():.3f}, Missed: {1-y_train.mean():.3f}")
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=10
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    joblib.dump(clf, 'model.pkl')
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    test_accuracy = (y_pred == y_test).mean()
    test_logloss = log_loss(y_test, y_pred_proba)
    test_brier = brier_score_loss(y_test, y_pred_proba)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION (on held-out test set)")
    print(f"{'='*50}")
    print(f"  Accuracy:    {test_accuracy:.4f}")
    print(f"  Log Loss:    {test_logloss:.4f}  (lower is better)")
    print(f"  Brier Score: {test_brier:.4f}  (lower is better)")
    print(f"  ROC AUC:     {test_auc:.4f}  (higher is better)")
    print(f"  Best iteration: {clf.best_iteration}")
    
    print(f"\nFeature Importance (top 15):")
    importance = clf.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])[:15]:
        print(f"  {name}: {imp:.4f}")
    
    print(f"\nSample predictions:")
    
    def make_sample(base_feats, cdn_feats=None):
        if use_cdn_features and cdn_feats:
            return [base_feats + cdn_feats]
        return [base_feats + [0] * len(CDN_FEATURE_NAMES)] if use_cdn_features else [base_feats]
    
    base1 = [3, 2, 300, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0.55]
    cdn1 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    print(f"  3-foot home dunk by elite player:  {clf.predict_proba(make_sample(base1, cdn1))[0][1]:.3f}")
    
    base2 = [3, 2, 300, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.45]
    cdn2 = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    print(f"  3-foot away driving layup:         {clf.predict_proba(make_sample(base2, cdn2))[0][1]:.3f}")
    
    base3 = [3, 2, 300, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0.45]
    cdn3 = [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    print(f"  3-foot fastbreak layup (turnover): {clf.predict_proba(make_sample(base3, cdn3))[0][1]:.3f}")
    
    base4 = [25, 2, 300, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.45]
    cdn4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(f"  25-foot home catch-and-shoot 3PT:  {clf.predict_proba(make_sample(base4, cdn4))[0][1]:.3f}")
    
    base5 = [25, 4, 30, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.50]
    cdn5 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    print(f"  25-foot step-back 3PT (clutch):    {clf.predict_proba(make_sample(base5, cdn5))[0][1]:.3f}")
    
    joblib.dump(FEATURE_NAMES, 'feature_names.pkl')
    print(f"\nSaved model.pkl, player_skills.pkl, feature_names.pkl")


if __name__ == "__main__":
    retrain_model()
