import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def retrain_model():
    print("Triggering automated retraining...")
    # In production, this would pull from the DB. Here we use the CSV.
    df = pd.read_csv('data/nba_plays.csv')
    df = df[df['EVENT_TYPE'].isin(['Made Shot', 'Missed Shot'])]
    
    # Feature Engineering
    X = df[['SHOT_DISTANCE']].fillna(0)
    y = df['EVENT_TYPE'].apply(lambda x: 1 if 'Made' in x else 0)
    
    # Training
    clf = LogisticRegression()
    clf.fit(X, y)
    
    # Save Artifact
    joblib.dump(clf, 'model.pkl')
    print(f"Model Retrained. Accuracy: {clf.score(X, y):.4f}")

if __name__ == "__main__":
    retrain_model()
