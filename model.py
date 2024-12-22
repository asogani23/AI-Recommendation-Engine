import torch
import torch.nn as nn
import torch.optim as optim

# Define the model architecture (replace with your actual model)
class RecommendationModel(nn.Module):
    def __init__(self):
        super(RecommendationModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Example: Input size 10, output size 1

    def forward(self, x):
            return self.fc(x)

# Initialize the model, loss function, and optimizer
model = RecommendationModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Example training loop (replace with your actual training data)
epochs = 5
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    # Dummy input (10 features) and target
    inputs = torch.randn(1, 10)  # Batch size of 1, 10 features
    target = torch.randn(1)      # Batch size of 1
    output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "recommendation_model.pt")
print("Model saved as recommendation_model.pt")


