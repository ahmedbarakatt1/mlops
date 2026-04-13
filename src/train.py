import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading Breast Cancer dataset...")
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101
)

# train the model and log parameters, metrics, and the model itself to MLflow
print("Starting MLflow run...")
mlflow.set_experiment("personal-mlops-project")

with mlflow.start_run() as run:
    # Save the Run ID for the automated pipeline
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # Train
    model_variant = os.getenv("MODEL_VARIANT", "weak").strip().lower()
    if model_variant == "weak":
        model = DummyClassifier(strategy="most_frequent")
        mlflow.log_param("model_variant", "weak")
        mlflow.log_param("model_type", "DummyClassifier")
    else:
        n_estimators = 150
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=101
        )
        mlflow.log_param("model_variant", "strong")
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_param("n_estimators", n_estimators)

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Log parameters and metrics to MLflow
    mlflow.log_param("test_size", 0.25)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model itself as an artifact
    mlflow.sklearn.log_model(model, "model")
    print("Logged to MLflow successfully.")

# export the run id
with open("model_info.txt", "w") as f:
    f.write(run_id)

print(f"Saved Run ID '{run_id}' to model_info.txt")
print("Training complete!")
