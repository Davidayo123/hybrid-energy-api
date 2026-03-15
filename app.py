from fastapi import FastAPI, request, jsonify
from local_inference_wrapper import LocalEdgeForecaster

app = FastAPI()

print("Starting Local Edge API and loading AI models...")
ai_brain = LocalEdgeForecaster()
print("AI API is live and listening on port 5000!")

@app.route('/predict', methods=['POST'])
def predict_energy():
    try:
        incoming_data = request.get_json()
        results = ai_brain.build_features_and_predict(incoming_data)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # host='127.0.0.1' ensures it only runs offline on the local hardware
    app.run(host='127.0.0.1', port=5000, debug=False)
