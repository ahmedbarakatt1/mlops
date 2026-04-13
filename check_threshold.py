import sys
import mlflow

THRESHOLD = 0.90  # Increased threshold for the new model dataset!

try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Checking Run ID: {run_id}")
except FileNotFoundError:
    print("ERROR: model_info.txt not found. Did the validate job run?")
    sys.exit(1)

client = mlflow.tracking.MlflowClient()

try:
    run = client.get_run(run_id)
except Exception as e:
    print(f"ERROR: Could not fetch run from MLflow. Details: {e}")
    sys.exit(1)

accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("ERROR: 'accuracy' metric not found in this MLflow run.")
    sys.exit(1)

print(f"Model accuracy from MLflow: {accuracy:.4f}")
print(f"Required threshold:         {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}.")
    print("Blocking deployment. The model is not good enough for production.")
    sys.exit(1)
else:
    print(f"PASSED: Accuracy {accuracy:.4f} meets the threshold {THRESHOLD}.")
    print("Model approved for deployment!")
    sys.exit(0)
