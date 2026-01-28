import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Chunk size for streaming through large files
CHUNK_SIZE = 50000

def retrain_model():
    print("Triggering automated retraining...")
    
    # Collect training data in chunks to handle large files
    X_chunks = []
    y_chunks = []
    total_rows = 0
    
    try:
        # Use Polars lazy scanning with batched collection for memory efficiency
        batches = pl.scan_csv('data/nba_plays.csv').collect_batches(chunk_size=CHUNK_SIZE)
        
        for df in batches:
            
            # Filter to shot events only (eventmsgtype: 1=Made, 2=Missed)
            df = df.filter(pl.col('eventmsgtype').is_in([1, 2]))
            
            # Combine home and visitor descriptions to get shot info
            df = df.with_columns([
                (pl.col('homedescription').fill_null('') + pl.col('visitordescription').fill_null('')).alias('description')
            ])
            
            # Extract shot distance using regex
            df = df.with_columns([
                pl.col('description').str.extract(r"(\d+)'", 1).cast(pl.Int32).alias('shot_distance')
            ])
            
            # Drop rows without distance info
            df = df.filter(pl.col('shot_distance').is_not_null())
            
            if df.height > 0:
                # Extract features and labels
                X_chunks.append(df.select('shot_distance').to_numpy())
                y_chunks.append((df.select('eventmsgtype').to_numpy() == 1).astype(int).flatten())
                total_rows += df.height
                print(f"Processed {total_rows} training samples...")
                
    except FileNotFoundError:
        print("Error: 'nba_plays.csv' not found in data/ folder.")
        return
    
    if not X_chunks:
        print("No training data found!")
        return
    
    # Combine all chunks
    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    
    print(f"Training on {len(y)} samples...")
    
    # Training
    clf = LogisticRegression()
    clf.fit(X, y)
    
    # Save Artifact
    joblib.dump(clf, 'model.pkl')
    print(f"Model Retrained. Accuracy: {clf.score(X, y):.4f}")

if __name__ == "__main__":
    retrain_model()
