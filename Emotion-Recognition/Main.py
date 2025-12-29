import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# --- CNN Model ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
        

# --- Dataset ---
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.ImageFolder(root="images/train", transform=transform)
test_data = datasets.ImageFolder(root="images/validation", transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# --- Model Setup ---
model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#model.load_state_dict(torch.load("first_model.pth"))
#model.eval()

# --- Training---

print("Starting training...")
for epoch in range(5):
    batch_count = 0
    for images, labels in train_loader:
        if batch_count > 1200:
            break
        batch_count += 1
        
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")


# --- Evaluation ---
correct = [0] * 7
total = [0] * 7
batch_index = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        for i in range(len(predicted)):
            correct[predicted[i]] += (predicted[i] == labels[i]).item()
            total[labels[i]] += 1

print("Correct predictions per class:", correct)
print("Total samples per class:", total)

# torch.save(model.state_dict(), "first_model.pth")
