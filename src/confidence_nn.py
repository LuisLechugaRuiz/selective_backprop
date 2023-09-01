import torch
import torch.nn as nn
import torch.optim as optim


class CustomLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.9, beta=0.1, gamma=0.1):
        super(CustomLinearLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))
        self.confidence = torch.ones(output_dim, input_dim) * 0.5
        self.counter = torch.zeros(output_dim, input_dim)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        return torch.matmul(x, self.weights.t()) + self.bias

    def update_confidence_and_counter(self, grad, correct_predictions, total_predictions):
        self.counter += correct_predictions / total_predictions
        normalized_counter = self.counter / total_predictions
        confidence_update = torch.abs(grad) * (1 - self.gamma * normalized_counter)
        self.confidence = self.alpha * self.confidence + (1 - self.alpha) * confidence_update


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.layer1 = CustomLinearLayer(2, 4)
        self.layer2 = CustomLinearLayer(4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Dummy data and labels
data = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
labels = torch.tensor([[1.0], [1.0], [0.0], [0.0]])

# Initialize the model, loss, and optimizer
model = CustomNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()

    # Update the confidence and counter for each layer
    correct_predictions = (outputs.round() == labels).sum().item()
    for layer in model.children():
        layer.update_confidence_and_counter(layer.weights.grad, correct_predictions, len(data))

    # Perform weight updates considering confidence and counter
    for layer in model.children():
        modified_grad = layer.weights.grad * (1 - layer.confidence) * (1 - 0.1 * (layer.counter / epoch if epoch > 0 else 1))
        layer.weights.data -= 0.1 * modified_grad

    print(f"Epoch {epoch}, Loss: {loss.item()}")
