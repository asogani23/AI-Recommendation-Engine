
# Load the trained model
model = RecommendationModel()
model.load_state_dict(torch.load("recommendation_model.pt", map_location=torch.device("cpu"), weights_only=True))
model.eval()

# Initialize Flask app
app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_id = data.get("user_id")
    # Generate dummy recommendations
    input_tensor = torch.randn(1, 10)
    recommendation = model(input_tensor).detach().numpy().tolist()
    return jsonify({"user_id": user_id, "recommendations": recommendation})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

