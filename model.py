import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Dummy dataset: Replace with your own user-item interactions
num_users, num_items = 100, 50
user_item_matrix = np.random.randint(0, 2, size=(num_users, num_items))

class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        return (user_vec * item_vec).sum(1)

# Initialize the model, optimizer, and loss function
model = RecommendationModel(num_users, num_items)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(10):
    for user in range(num_users):
        for item in range(num_items):
            rating = torch.tensor(user_item_matrix[user, item], dtype=torch.float32)
            user_tensor = torch.tensor([user], dtype=torch.int64)
            item_tensor = torch.tensor([item], dtype=torch.int64)

            pred = model(user_tensor, item_tensor)
            loss = loss_fn(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Save the trained model
torch.save(model.state_dict(), "recommendation_model.pt")

