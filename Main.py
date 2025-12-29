import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# example_image = Image.open("number3_square.png").convert("L")

# --- Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),                # convert images to tensors (0–1)
    transforms.Normalize((0.5,), (0.5,))  # normalize to range [-1, 1]
])

img_tensor = transform(image1).unsqueeze(0)

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# --- CNN model ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # input channel 1 (grayscale), output 16 filters
        self.pool = nn.MaxPool2d(2, 2)                # downsample from 28x28 to 14x14
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 32 filters
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)                 # 10 classes (digits 0–9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)             # output layer (no softmax)
        return x

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model.load_state_dict(torch.load("first_model.pth"))
# model.eval()

# --- Training ---
for epoch in range(15):  # 15 full passes through the dataset
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "first_model.pth")

# --- Testing ---
output = model(img_tensor)
prediction = torch.argmax(output, dim=1)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
