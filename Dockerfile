FROM python:3.10-slim

# Use efficient layering strategy
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Accept RUN_ID as argument
ARG RUN_ID
ENV MLFLOW_RUN_ID=${RUN_ID}

# Copy the rest of the project
COPY . .

# Simulate downloading the model to local directory
RUN echo "Simulating model download for Run ID: ${RUN_ID}" && \
    mkdir -p /opt/model && \
    echo "Model downloaded from MLflow run: ${RUN_ID}" > /opt/model/model_info.txt

# Command to run training script (or serve command in a real app)
CMD ["python", "src/train.py"]