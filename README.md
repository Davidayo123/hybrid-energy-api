# Real-Time Hybrid Energy Forecaster API ⚡

This repository contains the deployment architecture for a predictive machine learning API designed for real-time microgrid energy forecasting. It serves as the "Predictive Brain" for downstream industrial plant controllers.

## 🧠 Model Architecture
This system does not rely on a single algorithm. It utilizes a **Sequential & Parallel Hybrid Framework** featuring:
* **Deep Learning:** A Temporal Convolutional Network with Gated Recurrent Units (TE-GRU) for complex pattern recognition.
* **Gradient Boosting:** A LightGBM model for high-accuracy tabular baseline predictions.
* **Thermodynamic Control Loop:** A custom-built, dual-state Metropolis-Hastings algorithm with recency-weighted exponential decay. It dynamically adjusts model weights and residual bias over a rolling 120-hour memory buffer to adapt to volatile grid conditions.
* **Optimization:** Hyperparameters were mapped using a 7-dimensional Bayesian optimization space (Optuna) to find the absolute mathematical ceiling of the dataset.

## 🚀 API Framework
The model is wrapped in a high-performance **FastAPI** server, allowing hardware controllers (like ESP32s or Raspberry Pis) to ping the endpoint via standard HTTP POST requests from anywhere in the world.

---

## 📡 API Contract (How to use this API)

**Endpoint:** `POST /predict_load`

### 1. The Request (Input)
The downstream controller must send a JSON payload containing the current hour's weather/time features, alongside a strict 120-hour historical memory buffer for the Markov Chain to analyze.

```json
{
  "current_features": [[0.82, 25.4, 1.2, 0.5, ...]], 
  "history_true": [4.21, 4.15, 4.50, ...], 
  "history_features": [
    [0.70, 24.1, ...], 
    [0.75, 24.5, ...]
  ]
}
