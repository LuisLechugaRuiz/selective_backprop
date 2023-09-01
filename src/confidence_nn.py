import torch
import torch.nn as nn
import torch.optim as optim


class CustomLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.9, beta=0.1, gamma=0.1, delta=1.0, device='cpu'):
        super(CustomLinearLayer, self).__init__()
        self.alpha = alpha  # Decay term for confidence
        self.beta = beta    # Influence of the frequency
        self.gamma = gamma  # Decay term for confidence update
        self.delta = delta  # Decay rate for frequency normalization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO: Centralize this
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


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.layer1 = CustomLinearLayer(2, 4)
        self.layer2 = CustomLinearLayer(4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# For standard backpropagation -> TODO: Remove
class StandardNet(nn.Module):
    def __init__(self):
        super(StandardNet, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# TODO: Move to another file and remove custom_training (is just for testing).
def train_model(model, data, labels, custom_training=False, epochs=100, lr=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, labels = data.to(device), labels.to(device)
    losses = []

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        if custom_training:
            loss.backward(retain_graph=True)  # Note the retain_graph=True
            for layer in model.children():
                layer.update_frequency_and_confidence(layer.weights.grad)
                layer.custom_backward(layer.weights.grad, lr)
        else:
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        losses.append(loss.item())
    return losses


if __name__ == "__main__":
    data = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([[1.0], [1.0], [0.0], [0.0]], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training using CustomNet
    print("Training using CustomNet:")
    custom_model = CustomNet().to(device)
    custom_losses = train_model(custom_model, data, labels, custom_training=True)

    # Training using StandardNet
    print("\nTraining using StandardNet:")
    standard_model = StandardNet().to(device)
    standard_losses = train_model(standard_model, data, labels)

    # Comparing losses
    print("\nComparing losses:")
    for epoch, (custom_loss, standard_loss) in enumerate(zip(custom_losses, standard_losses)):
        print(f"Epoch {epoch}, Custom Loss: {custom_loss}, Standard Loss: {standard_loss}")
