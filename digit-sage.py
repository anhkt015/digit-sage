
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# 1. Tải dữ liệu MNIST
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 2. Xây dựng mô hình CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

# 3. Huấn luyện mô hình
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for images, labels in trainloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Đọc ảnh
img_path = '/content/R.jpg'  # Corrected: Added quotes around the file path
if os.path.exists(img_path): # Added: Check if the file exists
  img = Image.open(img_path).convert('L')  # chuyển sang ảnh đen trắng

  # Resize về 28x28 và chuyển thành tensor
  transform = transforms.Compose([
      transforms.Resize((28, 28)),
      transforms.ToTensor()
  ])
  img_tensor = transform(img).unsqueeze(0)  # thêm batch dimension

  # Dự đoán
  model.eval()
  output = model(img_tensor)
  predicted = torch.argmax(output, 1)
  print(f"👉 Dự đoán: Số {predicted.item()}")

  # Hiển thị ảnh
  plt.imshow(img, cmap='gray')
  plt.title(f"Dự đoán: {predicted.item()}")
  plt.axis('off')
  plt.show()
else:
  print(f"Error: Image file not found at {img_path}") # Added: Error message if file not found