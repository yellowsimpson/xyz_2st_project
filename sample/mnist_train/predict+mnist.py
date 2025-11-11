# predict_mnist.py
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# ─────────────────────────
# 1) 모델 구조는 train과 동일해야 함
# ─────────────────────────
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MNIST_CNN().to(device)
model.load_state_dict(torch.load("./mnist_cnn.pth", map_location=device))
model.eval()
print("MNIST 모델 로드 완료")

# MNIST와 동일한 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ─────────────────────────
# 2) 카메라에서 손글씨 숫자 인식
# ─────────────────────────
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        break

    # 임시로: 화면 중앙 근처에 숫자를 쓴다고 가정하고 중앙 부분만 crop
    h, w, _ = frame.shape
    size = min(h, w) // 2
    cx, cy = w // 2, h // 2
    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = cx + size // 2
    y2 = cy + size // 2

    digit_roi = frame[y1:y2, x1:x2]

    # 흑백 변환
    gray = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)

    # 28x28로 리사이즈
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # MNIST는 배경 검정, 숫자 흰색 → 반대면 반전
    # (필요하면 아래 한 줄 켜기)
    # resized = cv2.bitwise_not(resized)

    # 텐서 변환 + 정규화
    img_tensor = transform(resized).unsqueeze(0).to(device)  # shape: [1, 1, 28, 28]

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    # 화면 표시
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, f"Pred: {pred}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("MNIST Digit Cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
