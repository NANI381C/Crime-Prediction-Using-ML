# src/classifier_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Try to import xgboost; fallback to RandomForest
try:
    from xgboost import XGBClassifier
    CLASSIFIER = "xgb"
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    CLASSIFIER = "rf"

def train_classifier(df_raw, model_out_path="models/crime_classifier.joblib"):
    """
    df_raw: original raw dataframe (before monthly aggregation), must include:
            'Crime Description', 'Weapon Used', 'Victim Age', 'Victim Gender', 'City', 'Crime Domain'
    Returns: metrics dict and saved model path
    """
    df = df_raw.copy().dropna(subset=['Crime Description','Crime Domain','City'])
    # Basic encoding of textual features - keep simple bag-of-words for description using hashing (or label encode frequent categories)
    # For speed, we do simple label-encodings for categorical fields (not ideal, but works)
    encoders = {}
    for c in ['Crime Description','Weapon Used','Victim Gender','City']:
        le = LabelEncoder()
        df[c+'_enc'] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    # Target
    le_target = LabelEncoder()
    df['target_enc'] = le_target.fit_transform(df['Crime Domain'].astype(str))
    encoders['target'] = le_target

    # Features: Crime Description enc, Weapon enc, Age (numeric), Gender enc, City enc
    X = df[['Crime Description_enc','Weapon Used_enc','Victim Age','Victim Gender_enc','City_enc']].fillna(0)
    y = df['target_enc']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if CLASSIFIER == "xgb":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100)
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    # Save model and encoders
    joblib.dump(model, model_out_path)
    joblib.dump(encoders, model_out_path + ".encoders.joblib")

    return {"accuracy":acc, "report":report, "model_path":model_out_path}
