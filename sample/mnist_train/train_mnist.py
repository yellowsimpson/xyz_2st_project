# train_mnist.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) 하이퍼파라미터
batch_size = 64
epochs = 3
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 2) 데이터셋 (MNIST 자동 다운로드)
transform = transforms.Compose([
    transforms.ToTensor(),                 # [0, 255] -> [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균/표준편차 정규화 (관례적인 값)
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3) 간단한 CNN 모델 정의
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),   # 28x28 -> 26x26
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),  # 26x26 -> 24x24
            nn.ReLU(),
            nn.MaxPool2d(2),          # 24x24 -> 12x12
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)        # 0~9
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 4) 학습 루프
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

    # 간단한 테스트 정확도 확인
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = correct / total * 100
    print(f"    Test Accuracy: {acc:.2f}%")

# 5) 모델 저장
save_path = "./mnist_cnn.pth"
torch.save(model.state_dict(), save_path)
print("모델 저장 완료:", save_path)
