from flask import Flask, request, jsonify
import torch
import torch.nn as nn

app = Flask(__name__)

# Load the recommendation model
class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        return (user_vec * item_vec).sum(1)

model = RecommendationModel(100, 50)
model.load_state_dict(torch.load("recommendation_model.pt"))
model.eval()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user = torch.tensor([data['user_id']], dtype=torch.int64)
    item = torch.tensor([data['item_id']], dtype=torch.int64)
    with torch.no_grad():
        score = model(user, item).item()
    return jsonify({"recommendation_score": score}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

