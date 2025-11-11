import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# 1) YOLO 모델 불러오기
#  - car_detect.pt 모델에 "번호판" 클래스가 있다면 그걸 쓰고,
#  - 아직은 그냥 모든 박스에서 테스트해도 됨.
model = YOLO("/home/deepet/VSCode/xyz_2st_project/sample/weight/car_detect.pt")

# 2) OCR 전처리 (숫자에 맞게)
def preprocess_ocr_roi(img):
    # 그레이 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 노이즈 감소
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Otsu 이진화
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 필요하면 반전 (배경/글씨 색에 따라 조정)
    # th = cv2.bitwise_not(th)
    # OCR 잘 되도록 확대
    th_big = cv2.resize(th, None, fx=2.0, fy=2.0,
                        interpolation=cv2.INTER_LINEAR)
    return th_big

# 3) Tesseract로 숫자 OCR
def ocr_digits(img):
    custom_config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(
        img,
        lang="eng",
        config=custom_config
    )
    return text.strip()

# 4) 비디오/카메라
cap = cv2.VideoCapture(0)

# 번호판 클래스 id가 따로 있다면 여기 설정 (예: 0, 1 등)
# 없으면 None으로 두고 모든 박스에 대해 OCR 시도
PLATE_CLASS_ID = None   # 예: 번호판 class가 2번이면 2로 바꾸기

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    # 5) YOLO 예측
    results = model(frame, imgsz=640)[0]

    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            # 좌표와 클래스, conf 추출
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())

            # 번호판 클래스만 쓰고 싶으면 여기서 필터
            if PLATE_CLASS_ID is not None and cls_id != PLATE_CLASS_ID:
                continue

            # 너무 작은 박스는 스킵
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            # ROI 추출
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # OCR 전처리
            th_big = preprocess_ocr_roi(roi)

            # 숫자 OCR
            text = ocr_digits(th_big)
            print("인식된 텍스트:", text)

            # 시각화
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

    cv2.imshow("YOLO + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
