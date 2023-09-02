import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from confidence_nn.confidence_nn import ConfidenceNet, train_confidence_model
from synaptic_intelligence_nn.synaptic_intelligence import SynapticIntelligence, train_si_model


# For standard backpropagation
class StandardNet(nn.Module):
    def __init__(self):
        super(StandardNet, self).__init__()
        self.layer1 = nn.Linear(20, 40)
        self.layer2 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# Function to evaluate the model
def evaluate(model, data, labels):
    with torch.no_grad():
        outputs = model(data)
        criterion = nn.MSELoss()
        loss = criterion(outputs, labels)
    return loss.item()


def generate_dataset(device):
    # Generate data for Task A with 20 features
    data_A, labels_A = make_classification(n_features=20, n_redundant=0, n_informative=20, n_clusters_per_class=1, n_classes=2, n_samples=200)
    scaler_A = StandardScaler()
    data_A = scaler_A.fit_transform(data_A)
    labels_A = labels_A.reshape(-1, 1)
    data_A = torch.tensor(data_A, dtype=torch.float32).to(device)
    labels_A = torch.tensor(labels_A, dtype=torch.float32).to(device)

    # Generate data for Task B using sklearn's make_moons function with noise
    X, y = make_moons(n_samples=200, noise=0.1)
    scaler_B = StandardScaler()
    X = scaler_B.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Adding 18 random features to make it 20-dimensional
    X_train = np.hstack((X_train, np.random.rand(X_train.shape[0], 18)))
    data_B = torch.tensor(X_train, dtype=torch.float32).to(device)
    labels_B = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)

    return data_A, labels_A, data_B, labels_B


def train_standard_model(data, labels, epochs=100, lr=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StandardNet().to(device)
    data, labels = data.to(device), labels.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # print(f"Epoch {epoch}, Loss: {loss.item()}")
    return model


def train_custom_model(model_name, data, labels, epochs=100, lr=0.1):
    if model_name == "confidence":
        custom_model = ConfidenceNet(device).to(device)
        train_confidence_model(custom_model, data, labels, epochs, lr)
    elif model_name == "synaptic":
        custom_model = SynapticIntelligence(device).to(device)
        train_si_model(custom_model, data, labels, epochs, lr)
    return custom_model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_A, labels_A, data_B, labels_B = generate_dataset(device)

    # Initialize models
    model_name = "synaptic" # TODO: Don't hardcode.

    # Train and test on Task A
    print("... Training on Task A ...")
    print("---- Custom model ----")
    custom_model = train_custom_model(model_name, data_A, labels_A)
    print("---- Standard model ----")
    standard_model = train_standard_model(data_A, labels_A)
    performance_A1_custom = evaluate(custom_model, data_A, labels_A)
    performance_A1_standard = evaluate(standard_model, data_A, labels_A)

    # Save the models
    torch.save(custom_model.state_dict(), 'custom_model_task_A.pth')
    torch.save(standard_model.state_dict(), 'standard_model_task_A.pth')

    # Train and test on Task B
    print("... Training on Task B ...")
    print("---- Custom model ----")
    custom_model = train_custom_model(model_name, data_A, labels_A)
    print("---- Standard model ----")
    standard_model = train_standard_model(data_B, labels_B)
    performance_B_custom = evaluate(custom_model, data_B, labels_B)
    performance_B_standard = evaluate(standard_model, data_B, labels_B)

    # Load the saved models (Optional: to see the effect of forgetting)
    # custom_model.load_state_dict(torch.load('custom_model_task_A.pth'))
    # standard_model.load_state_dict(torch.load('standard_model_task_A.pth'))

    # Test on Task A again
    print("... Retesting on Task A ...")
    performance_A2_custom = evaluate(custom_model, data_A, labels_A)
    performance_A2_standard = evaluate(standard_model, data_A, labels_A)

    print("---- Results ----")
    print("---- Custom model ----")
    print(f"Task A (1st time): {performance_A1_custom}")
    print(f"Task B: {performance_B_custom}")
    print(f"Task A (2nd time): {performance_A2_custom}")
    print("---- Standard model ----")
    print(f"Task A (1st time): {performance_A1_standard}")
    print(f"Task B: {performance_B_standard}")
    print(f"Task A (2nd time): {performance_A2_standard}")
