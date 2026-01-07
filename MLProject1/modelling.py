import mlflow
import dagshub
import dagshub.auth
import pandas as pd
import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()

token = os.environ.get("DAGSHUB_TOKEN")
if token:
    print("DagsHub Token terdeteksi, melakukan autentikasi otomatis...")
    dagshub.auth.add_app_token(token)

dagshub.init(repo_owner="Miftahrachmatullah", 
             repo_name="submission-mlops", 
             mlflow=True)

print("Loading data...")
try:
    df = pd.read_csv("data_clean.csv")
except FileNotFoundError:
    df = pd.read_csv("Submission_CI_Repo/data_clean.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Mulai Training dengan n_estimators={args.n_estimators}...")

with mlflow.start_run():
    mlflow.sklearn.autolog(disable=True)

    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Akurasi: {acc}")

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("accuracy", acc)
    
    mlflow.sklearn.log_model(model, "model")
    
    if os.path.exists("churn_preprocessing"):
        mlflow.log_artifacts("churn_preprocessing", artifact_path="preprocessing")
    elif os.path.exists("Submission_CI_Repo/churn_preprocessing"):
         mlflow.log_artifacts("Submission_CI_Repo/churn_preprocessing", artifact_path="preprocessing")

print("Training Selesai.")