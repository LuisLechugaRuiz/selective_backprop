import torch
import torch.nn as nn
import torch.optim as optim


class SynapticIntelligence(nn.Module):
    def __init__(self, device):
        super(SynapticIntelligence, self).__init__()
        self.fc1 = nn.Linear(20, 40)
        self.fc2 = nn.Linear(40, 1)

        self.omega_fc1 = torch.zeros_like(self.fc1.weight.data).to(device)
        self.prev_weights_fc1 = torch.zeros_like(self.fc1.weight.data).to(device)

        self.omega_fc2 = torch.zeros_like(self.fc2.weight.data).to(device)
        self.prev_weights_fc2 = torch.zeros_like(self.fc2.weight.data).to(device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def update_importance(self):
        for param, omega, prev_weights in zip(
            [self.fc1.weight, self.fc2.weight], 
            [self.omega_fc1, self.omega_fc2], 
            [self.prev_weights_fc1, self.prev_weights_fc2]
        ):
            # print("Param:", param)
            delta_theta = param.data - prev_weights
            # print("Param data:", param.data)
            omega += delta_theta * param.grad.data

    def update_loss(self, base_loss):
        si_loss = 0
        for param, omega, prev_weights in zip(
            [self.fc1.weight, self.fc2.weight], 
            [self.omega_fc1, self.omega_fc2], 
            [self.prev_weights_fc1, self.prev_weights_fc2]
        ):
            si_loss += (omega * (param - prev_weights) ** 2).sum()
        return base_loss + 0.1 * si_loss  # 0.1 is the regularization strength


def train_si_model(model, data, labels, epochs=100, lr=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, labels = data.to(device), labels.to(device)
    losses = []

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        # print(f"Epoch {epoch}, Real loss: {loss.item()}")

        # Add SI regularization to loss
        total_loss = model.update_loss(loss)

        # Backward and optimize
        total_loss.backward()
        optimizer.step()

        # Update Ω (importance)
        model.update_importance()

        # print(f"Epoch {epoch}, Loss: {total_loss.item()}")
        losses.append(total_loss.item())
    return losses
