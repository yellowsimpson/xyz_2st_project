import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# ===== 1) MNIST 모델 정의 (train 때와 동일) =====
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    size = min(h, w) // 2
    cx, cy = w // 2, h // 2
    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = cx + size // 2
    y2 = cy + size // 2

    roi = frame[y1:y2, x1:x2]

    # ---- (1) 전처리: 흑백 + 이진화 ----
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 배경 흰색, 글자 검정이면 반전하면 더 잘 됨
    # gray = cv2.bitwise_not(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise 제거용 morphology (옵션)
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    # ---- (2) contour로 숫자 후보 찾기 ----
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_boxes = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = w_box * h_box
        # 너무 작거나 넓은 건 버리기 (필요시 값 조정)
        if area < 100 or h_box < 10:
            continue
        digit_boxes.append((x, y, w_box, h_box))

    # 왼쪽 → 오른쪽 정렬
    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

    pred_str = ""

    for (dx, dy, dw, dh) in digit_boxes:
        digit_img = th[dy:dy+dh, dx:dx+dw]

        # MNIST input: 28x28
        digit_resized = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)

        # 흑백(0~255) → MNIST 텐서
        tensor = transform(digit_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            pred = out.argmax(dim=1).item()
            pred_str += str(pred)

        # ROI 안에 digit box 그리기
        cv2.rectangle(roi, (dx, dy), (dx+dw, dy+dh), (0, 0, 255), 1)
        cv2.putText(roi, str(pred), (dx, dy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # 전체 프레임에 결과 출력
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, f"Pred: {pred_str}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Multi-digit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
