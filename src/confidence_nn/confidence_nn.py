import torch
import torch.nn as nn
import torch.optim as optim


class ConfidenceLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.9, beta=0.1, gamma=0.1, delta=1.0, device='cpu'):
        super(ConfidenceLinearLayer, self).__init__()
        self.alpha = alpha  # Decay term for confidence
        self.beta = beta    # Influence of the frequency
        self.gamma = gamma  # Decay term for confidence update
        self.delta = delta  # Decay rate for frequency normalization
        self.device = device
        self.init_parameters(input_dim, output_dim)

    def init_parameters(self, input_dim, output_dim):
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim).to(self.device))
        self.bias = nn.Parameter(torch.zeros(output_dim).to(self.device))
        self.confidence = torch.ones(output_dim, input_dim).to(self.device) * 0.5
        self.frequency = torch.zeros(output_dim, input_dim).to(self.device)
        self.moving_average = torch.zeros_like(self.confidence).to(self.device)

    def forward(self, x):
        return torch.matmul(x, self.weights.t()) + self.bias

    def custom_backward(self, grad, lr):
        normalized_frequency = self.frequency / (self.frequency + self.delta)
        modified_factor = (1 - self.confidence) * (1 - self.beta * normalized_frequency)
        modified_grad = grad * modified_factor
        self.weights.data -= lr * modified_grad

    def update_frequency_and_confidence(self, grad):
        self.frequency += 1
        normalized_frequency = self.frequency / (self.frequency + self.delta)

        # Moving average of gradient directions
        direction_avg = self.alpha * self.moving_average + (1 - self.alpha) * torch.sign(grad)
        self.moving_average = direction_avg

        # Calculate stability
        stability = torch.abs(direction_avg)

        # Calculate the absolute gradient (already have this as |grad|)
        abs_gradient = torch.abs(grad)

        # Combine stability and magnitude to form new confidence measure
        confidence_update = stability * abs_gradient * (1 - self.gamma * normalized_frequency)

        self.confidence = self.alpha * self.confidence + (1 - self.alpha) * confidence_update
        self.confidence = torch.sigmoid(self.confidence)  # Non-linearity
        torch.clamp_(self.confidence, 0, 1)  # optional: to keep it within a range

        # Debug
        # print("Confidence shape:", self.confidence.shape, "Confidence values:", self.confidence)


class ConfidenceNet(nn.Module):
    def __init__(self):
        super(ConfidenceNet, self).__init__()
        self.layer1 = ConfidenceLinearLayer(20, 40)
        self.layer2 = ConfidenceLinearLayer(40, 1)
        self.losses = []

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def train_confidence_model(model, data, labels, epochs=100, lr=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, labels = data.to(device), labels.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # Note the retain_graph=True
        for layer in model.children():
            layer.update_frequency_and_confidence(layer.weights.grad)
            layer.custom_backward(layer.weights.grad, lr)
        # print(f"Epoch {epoch}, Loss: {loss.item()}")
        model.losses.append(loss.item())
    return model.losses
