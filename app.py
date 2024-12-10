
import joblib
from flask import Flask, jsonify, request
import datetime
import random
from prometheus_client import Counter, Histogram, Gauge, generate_latest, multiprocess, CollectorRegistry, make_wsgi_app
from prometheus_client.exposition import start_http_server
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import time

app = Flask(__name__)

# Prometheus Metrics
registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    "request_count",
    "Total number of requests",
    ["method", "endpoint", "status"],
    registry=registry,
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken for predictions",
    registry=registry,
)
ERROR_COUNT = Counter(
    "error_count",
    "Number of prediction errors",
    ["error_type"],
    registry=registry,
)
MODEL_PREDICTIONS = Counter(
    "model_predictions",
    "Count of model predictions by class",
    ["class"],
    registry=registry,
)
HEALTH_STATUS = Gauge(
    "health_status", "Current health status of the application", registry=registry
)

def make_prediction(data):
    model = joblib.load("src/artifacts/models/random_forest.joblib")
    prediction = model.predict(data)
    prediction_label = "Approved" if prediction == 1 else "Rejected"

    return jsonify({
        "status": "success",
        "prediction": prediction_label,
        "raw_output": int(prediction)
    })

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.json
        prediction = make_prediction(data)
        MODEL_PREDICTIONS.labels(prediction).inc()
        REQUEST_COUNT.labels(request.method, "/predict", "200").inc()
        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        ERROR_COUNT.labels(str(e)).inc()
        REQUEST_COUNT.labels(request.method, "/predict", "500").inc()
        return jsonify({"error": "Prediction failed"}), 500
    finally:
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

def check_model_loaded():
    try:
        model = joblib.load("src/artifacts/models/random_forest.joblib")
        return True
    except Exception as e:
        print(f"Model loading error: {e}")
        return False
    
@app.route('/health', methods=['GET'])
def health_check():
    model_status = check_model_loaded()
    
    # Prepare the response
    response = {
        "status": "healthy" if model_status else "unhealthy",
        "checks": {
            "model_loaded": model_status,
            "api_status": "ok",
            "model_version": "1.0.0",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
    }
    
    return jsonify(response)

@app.route('/metrics')
def metrics():
    return generate_latest(registry), 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
