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

# Injury-enriched shots path (includes CDN + injury features)
INJURY_ENRICHED_PATH = Path('data/shots_with_injury.parquet')

# Player metadata path
PLAYER_METADATA_PATH = Path('data/player_metadata.parquet')

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

# Player physical features
PLAYER_PHYSICAL_FEATURE_NAMES = [
    'height_inches',
    'weight',
    'is_guard',
    'is_forward',
    'is_center',
]

# Injury context features
INJURY_FEATURE_NAMES = [
    'is_returning_from_injury',
    'days_since_injury',
    'had_leg_injury',
    'had_shooting_injury',
    'is_playing_hurt',
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
        'turnaround fadeaway': 'desc_fadeaway',  # Group with fadeaway
        'running': 'desc_running',
        'floating': 'desc_floating',
        'driving floating': 'desc_floating',  # Group with floating
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


def load_player_metadata() -> dict | None:
    """Load player physical metadata (height, weight, position) if available."""
    if not PLAYER_METADATA_PATH.exists():
        return None
    
    print(f"  Loading player metadata from {PLAYER_METADATA_PATH}...")
    df = pl.read_parquet(PLAYER_METADATA_PATH)
    
    # Calculate average values for missing data
    avg_height = df.filter(pl.col('height_inches').is_not_null())['height_inches'].mean()
    avg_weight = df.filter(pl.col('weight').is_not_null())['weight'].mean()
    
    # Build lookup by player_id
    player_lookup = {}
    for row in df.iter_rows(named=True):
        player_id = row['player_id']
        player_lookup[player_id] = {
            'height_inches': row['height_inches'] if row['height_inches'] else avg_height,
            'weight': row['weight'] if row['weight'] else avg_weight,
            'is_guard': int(row['is_guard']) if row['is_guard'] else 0,
            'is_forward': int(row['is_forward']) if row['is_forward'] else 0,
            'is_center': int(row['is_center']) if row['is_center'] else 0,
        }
    
    print(f"  Loaded metadata for {len(player_lookup):,} players")
    print(f"  Avg height: {avg_height:.1f} inches, Avg weight: {avg_weight:.1f} lbs")
    return player_lookup, avg_height, avg_weight


def load_injury_enriched_data() -> pl.DataFrame | None:
    """Load injury-enriched shots data if available."""
    if not INJURY_ENRICHED_PATH.exists():
        return None
    
    print(f"  Loading injury-enriched data from {INJURY_ENRICHED_PATH}...")
    df = pl.read_parquet(INJURY_ENRICHED_PATH)
    
    # Ensure injury features are int type
    for col in INJURY_FEATURE_NAMES:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int32))
    
    print(f"  Loaded {len(df):,} injury-enriched shots")
    
    # Show injury feature stats
    returning = df.filter(pl.col('is_returning_from_injury') == 1).height
    playing_hurt = df.filter(pl.col('is_playing_hurt') == 1).height
    print(f"  Returning from injury: {returning:,} ({100*returning/len(df):.1f}%)")
    print(f"  Playing hurt: {playing_hurt:,} ({100*playing_hurt/len(df):.1f}%)")
    
    return df


def train_from_enriched_data(
    enriched_df: pl.DataFrame,
    player_skill: dict,
    player_lookup: dict,
    avg_height: float,
    avg_weight: float,
    use_player_features: bool,
    feature_names: list,
):
    """
    Train model directly from pre-enriched parquet data (fast path).
    
    This is used when injury-enriched data is available, which already
    has CDN and injury features pre-joined.
    """
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    
    print(f"  Processing {len(enriched_df):,} enriched shots...")
    
    # First, calculate player skill ratings from this data
    print("  Calculating player skill ratings...")
    player_stats = {}
    for row in enriched_df.iter_rows(named=True):
        player = row.get('player_name', '')
        if player:
            if player not in player_stats:
                player_stats[player] = {'made': 0, 'total': 0}
            player_stats[player]['total'] += 1
            if row.get('shot_made') == 1:
                player_stats[player]['made'] += 1
    
    total_made = sum(p['made'] for p in player_stats.values())
    total_shots = sum(p['total'] for p in player_stats.values())
    league_avg = total_made / total_shots if total_shots > 0 else 0.45
    prior_weight = 50
    
    player_skill_local = {}
    for player, stats in player_stats.items():
        smoothed_fg = (stats['made'] + prior_weight * league_avg) / (stats['total'] + prior_weight)
        player_skill_local[player] = round(smoothed_fg, 4)
    
    print(f"  Calculated skill ratings for {len(player_skill_local)} players")
    print(f"  League average FG%: {league_avg:.3f}")
    
    # Save player skill ratings
    joblib.dump(player_skill_local, 'player_skills.pkl')
    
    # Build feature matrix
    print("  Building feature matrix...")
    
    # Map columns from enriched data to feature names
    feature_vectors = []
    labels = []
    
    for row in enriched_df.iter_rows(named=True):
        features = []
        
        # Base features
        features.append(row.get('shot_distance', 0) or 0)
        features.append(row.get('period', 1) or 1)
        
        # Parse clock to seconds
        clock = row.get('clock', 'PT6M00.00S')
        if clock and isinstance(clock, str) and clock.startswith('PT'):
            try:
                # Format: PT11M48.00S
                import re
                match = re.match(r'PT(\d+)M([\d.]+)S', clock)
                if match:
                    mins, secs = match.groups()
                    seconds_remaining = int(mins) * 60 + int(float(secs))
                else:
                    seconds_remaining = 360
            except:
                seconds_remaining = 360
        else:
            seconds_remaining = 360
        features.append(seconds_remaining)
        
        # Shot type features
        dist = row.get('shot_distance', 0) or 0
        features.append(1 if dist > 23 else 0)  # is_three
        
        sub_type = (row.get('sub_type') or '').lower()
        desc = (row.get('descriptor') or '').lower()
        
        features.append(1 if 'dunk' in sub_type else 0)  # is_dunk
        features.append(1 if 'layup' in sub_type else 0)  # is_layup
        features.append(1 if 'hook' in sub_type else 0)  # is_hook
        features.append(1 if 'tip' in sub_type or 'tip' in desc else 0)  # is_tip
        features.append(1 if 'fadeaway' in desc else 0)  # is_fadeaway
        features.append(1 if 'bank' in sub_type else 0)  # is_bank
        features.append(1 if 'alley' in sub_type else 0)  # is_alley_oop
        features.append(0)  # is_home - not easily available, default to 0
        features.append(1 if 'pullup' in desc else 0)  # is_pullup
        
        # Player skill rating
        player = row.get('player_name', '')
        features.append(player_skill_local.get(player, league_avg))
        
        # CDN features
        features.append(1 if row.get('is_fastbreak') else 0)
        features.append(1 if row.get('is_second_chance') else 0)
        features.append(1 if row.get('is_from_turnover') else 0)
        features.append(1 if row.get('is_points_in_paint') else 0)
        features.append(1 if 'pullup' in desc else 0)  # desc_pullup
        features.append(1 if 'driving' in desc else 0)  # desc_driving
        features.append(1 if 'step back' in desc else 0)  # desc_step_back
        features.append(1 if 'fadeaway' in desc else 0)  # desc_fadeaway
        features.append(1 if 'running' in desc else 0)  # desc_running
        features.append(1 if 'floating' in desc else 0)  # desc_floating
        features.append(1 if 'turnaround' in desc else 0)  # desc_turnaround
        features.append(1 if 'cutting' in desc else 0)  # desc_cutting
        
        # Player physical features
        if use_player_features:
            player_id = row.get('player_id')
            if player_id and player_id in player_lookup:
                p = player_lookup[player_id]
                features.append(p['height_inches'])
                features.append(p['weight'])
                features.append(p['is_guard'])
                features.append(p['is_forward'])
                features.append(p['is_center'])
            else:
                features.append(avg_height)
                features.append(avg_weight)
                features.append(0)
                features.append(0)
                features.append(0)
        
        # Injury features
        features.append(row.get('is_returning_from_injury', 0) or 0)
        features.append(row.get('days_since_injury', 365) or 365)
        features.append(row.get('had_leg_injury', 0) or 0)
        features.append(row.get('had_shooting_injury', 0) or 0)
        features.append(row.get('is_playing_hurt', 0) or 0)
        
        feature_vectors.append(features)
        labels.append(row.get('shot_made', 0))
    
    X = np.array(feature_vectors, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Expected features: {len(feature_names)}, Got: {X.shape[1]}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n[Training XGBoost with {len(feature_names)} features]")
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=10,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    logloss = log_loss(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n=== Model Performance ===")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    
    # Feature importance
    print(f"\n=== Top 10 Feature Importances ===")
    importances = list(zip(feature_names, model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.4f}")
    
    # Save model
    joblib.dump(model, 'shot_model.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    print(f"\nSaved model to shot_model.pkl ({len(feature_names)} features)")
    
    return model


def retrain_model():
    global FEATURE_NAMES  # May be updated if CDN data available
    
    print("Triggering automated retraining with XGBoost...")
    
    # Check for injury-enriched data first (includes CDN features)
    injury_data = load_injury_enriched_data()
    use_injury_features = injury_data is not None
    
    # Fall back to CDN-only enriched data if no injury data
    cdn_data = None
    if not use_injury_features:
        cdn_data = load_cdn_enrichment()
    use_cdn_features = cdn_data is not None or use_injury_features
    
    # Check for player physical metadata
    player_meta_result = load_player_metadata()
    use_player_features = player_meta_result is not None
    if use_player_features:
        player_lookup, avg_height, avg_weight = player_meta_result
    else:
        player_lookup, avg_height, avg_weight = {}, 78.0, 215.0  # Default averages
    
    # Build feature list
    FEATURE_NAMES = BASE_FEATURE_NAMES.copy()
    if use_cdn_features:
        FEATURE_NAMES.extend(CDN_FEATURE_NAMES)
    if use_player_features:
        FEATURE_NAMES.extend(PLAYER_PHYSICAL_FEATURE_NAMES)
    if use_injury_features:
        FEATURE_NAMES.extend(INJURY_FEATURE_NAMES)
    
    print(f"Using {len(FEATURE_NAMES)} features:")
    print(f"  Base: {len(BASE_FEATURE_NAMES)}")
    if use_cdn_features:
        print(f"  CDN: {len(CDN_FEATURE_NAMES)}")
    if use_player_features:
        print(f"  Player Physical: {len(PLAYER_PHYSICAL_FEATURE_NAMES)}")
    if use_injury_features:
        print(f"  Injury: {len(INJURY_FEATURE_NAMES)}")
    print(f"Features: {FEATURE_NAMES}")

    # Fast path: If we have injury-enriched data, train directly from it
    if use_injury_features and injury_data is not None:
        print("\n[Fast Path] Training from injury-enriched data...")
        return train_from_enriched_data(injury_data, {}, player_lookup, 
                                         avg_height, avg_weight, use_player_features, FEATURE_NAMES)
    
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
    
    # If using CDN data, create a lookup by (game_id, player_id, period)
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
            
            # Add is_home: shot taken by home team (homedescription is not empty)
            df = df.with_columns([
                (pl.col('homedescription').is_not_null() & (pl.col('homedescription') != '')).cast(pl.Int8).alias('is_home')
            ])
            
            # Add player skill rating
            df = df.with_columns([
                pl.col('player1_name').map_elements(
                    lambda x: player_skill.get(x, league_avg),
                    return_dtype=pl.Float64
                ).alias('player_skill_rating')
            ])
            
            # Build base feature matrix
            base_feature_cols = [
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
            
            X_base = df.select(base_feature_cols).to_numpy()
            y_batch = (df.select('eventmsgtype').to_numpy() == 1).astype(int).flatten()
            
            if use_cdn_features:
                # Add CDN features - for each row, look up matching CDN data
                cdn_features = np.zeros((df.height, len(CDN_FEATURE_NAMES)), dtype=np.float32)
                
                for i, row in enumerate(df.iter_rows(named=True)):
                    game_id = row.get('game_id')
                    player_id = row.get('player1_id')
                    period = row.get('period')
                    
                    key = (game_id, player_id, period)
                    if key in cdn_lookup and cdn_lookup[key]:
                        # Take the first matching shot (simplified - could improve matching)
                        cdn_row = cdn_lookup[key].pop(0) if cdn_lookup[key] else {}
                        for j, col in enumerate(CDN_FEATURE_NAMES):
                            cdn_features[i, j] = cdn_row.get(col, 0)
                
                X_batch = np.hstack([X_base, cdn_features])
            else:
                X_batch = X_base
            
            # Add player physical features if available
            if use_player_features:
                physical_features = np.zeros((df.height, len(PLAYER_PHYSICAL_FEATURE_NAMES)), dtype=np.float32)
                
                for i, row in enumerate(df.iter_rows(named=True)):
                    player_id = row.get('player1_id')
                    if player_id and player_id in player_lookup:
                        p = player_lookup[player_id]
                        physical_features[i, 0] = p['height_inches']
                        physical_features[i, 1] = p['weight']
                        physical_features[i, 2] = p['is_guard']
                        physical_features[i, 3] = p['is_forward']
                        physical_features[i, 4] = p['is_center']
                    else:
                        # Use averages for unknown players
                        physical_features[i, 0] = avg_height
                        physical_features[i, 1] = avg_weight
                        # Position defaults to 0
                
                X_batch = np.hstack([X_batch, physical_features])
            
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
    print(f"  Feature count: {X_train.shape[1]}")
    print(f"  Class balance - Made: {y_train.mean():.3f}, Missed: {1-y_train.mean():.3f}")
    
    # Calculate scale_pos_weight for class imbalance
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
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=10
    )
    
    # Train with early stopping using test set as validation
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Save model artifact
    joblib.dump(clf, 'model.pkl')
    
    # Calculate metrics on TEST set
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
    
    print(f"\nFeature Importance (top 15):")
    importance = clf.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])[:15]:
        print(f"  {name}: {imp:.4f}")
    
    # Sample predictions
    print(f"\nSample predictions:")
    n_features = len(FEATURE_NAMES)
    
    def make_sample(base_feats, cdn_feats=None, player_feats=None):
        result = base_feats.copy()
        if use_cdn_features:
            result.extend(cdn_feats if cdn_feats else [0] * len(CDN_FEATURE_NAMES))
        if use_player_features:
            # Default: 78 inches (6'6"), 215 lbs, guard
            result.extend(player_feats if player_feats else [78, 215, 1, 0, 0])
        return [result]
    
    # 3-foot home dunk by tall center (7'0", 265 lbs)
    base1 = [3, 2, 300, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0.55]
    cdn1 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # points in paint
    player1 = [84, 265, 0, 0, 1]  # 7'0" center
    print(f"  3-foot dunk by 7'0\" center:        {clf.predict_proba(make_sample(base1, cdn1, player1))[0][1]:.3f}")
    
    # 3-foot away layup by small guard (6'0", 175 lbs)
    base2 = [3, 2, 300, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.45]
    cdn2 = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  # points in paint, driving
    player2 = [72, 175, 1, 0, 0]  # 6'0" guard
    print(f"  3-foot driving layup by 6'0\" guard: {clf.predict_proba(make_sample(base2, cdn2, player2))[0][1]:.3f}")
    
    # Fastbreak layup by forward (6'8", 225 lbs)
    base3 = [3, 2, 300, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0.45]
    cdn3 = [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]  # fastbreak, from turnover, paint, running
    player3 = [80, 225, 0, 1, 0]  # 6'8" forward
    print(f"  Fastbreak layup by 6'8\" forward:   {clf.predict_proba(make_sample(base3, cdn3, player3))[0][1]:.3f}")
    
    # 25-foot 3PT by tall guard (6'6", 200 lbs)
    base4 = [25, 2, 300, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.45]
    cdn4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # catch and shoot
    player4 = [78, 200, 1, 0, 0]  # 6'6" guard
    print(f"  25-foot 3PT by 6'6\" guard:         {clf.predict_proba(make_sample(base4, cdn4, player4))[0][1]:.3f}")
    
    # Step-back three by 6'3" guard in clutch
    base5 = [25, 4, 30, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.50]
    cdn5 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # step back
    player5 = [75, 185, 1, 0, 0]  # 6'3" guard
    print(f"  Step-back 3PT by 6'3\" guard:       {clf.predict_proba(make_sample(base5, cdn5, player5))[0][1]:.3f}")
    
    # Save artifacts for inference
    joblib.dump(FEATURE_NAMES, 'feature_names.pkl')
    if use_player_features:
        joblib.dump((player_lookup, avg_height, avg_weight), 'player_physical.pkl')
        print(f"\nSaved model.pkl, player_skills.pkl, feature_names.pkl, player_physical.pkl")
    else:
        print(f"\nSaved model.pkl, player_skills.pkl, feature_names.pkl")


if __name__ == "__main__":
    retrain_model()
