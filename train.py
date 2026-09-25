import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import DigitCNN

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Dataset
train_dataset = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Loading model
model = DigitCNN().to(device)

# loss & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
epochs = 5

for epoch in range(epochs):
    running_loss = 0

    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = loss_fn(logits, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch + 1}/{epochs}]Loss: {average_loss:.4f}")

# Evaluation

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)


accuracy = correct / total

print(f"Test accuracy: {accuracy:.2%}")

# Save model

torch.save(model.state_dict(), "digit_model.pth")

print("Model saved to digit_model.pth")
