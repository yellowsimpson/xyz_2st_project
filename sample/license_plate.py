import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# tesseract 경로가 필요하면 설정 (우분투 기본)
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# 1) YOLO 모델 불러오기
# 만약 번호판 전용으로 학습된 모델(.pt)이 있다면 그 경로를 넣으세요.
# 아니면 yolov8n.pt로 일반 탐지 후 'license_plate' 클래스로 학습시켜야 합니다.
model = YOLO("/home/deepet/VSCode/xyz_1st_project/xyz_1st_project/weight/car_detect.pt")   # 일반 모델. 번호판 전용 모델이면 그 파일을 사용.

# 2) 전처리용 헬퍼
def preprocess_plate(img):
    # 그레이 -> 대비 향상 -> 이진화 -> resize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE로 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # 이진화
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 필요시 morphology
    kernel = np.ones((2,2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    # OCR에 맞게 적당한 크기로 리사이즈
    h, w = th.shape
    scale = 2.0 if max(h,w) < 200 else 1.0
    th = cv2.resize(th, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    return th

# 3) OCR 헬퍼 (Tesseract 사용)
def ocr_tesseract(img):
    # img는 전처리된 그레이/이진 이미지
    custom_config = "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ가나다라마바사아자차카타파하" 
    # PSM 7 한 줄 텍스트 가정. whitelist는 필요시 조절.
    text = pytesseract.image_to_string(img, lang="kor+eng", config=custom_config)
    return text.strip()

# 4) 비디오/카메라 읽기 (파일도 가능)
cap = cv2.VideoCapture(0)  # 웹캠 대신 비디오 파일 경로도 가능

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 5) YOLO 예측 (frame 단위)
    results = model(frame, imgsz=640)[0]  # 결과 객체

    # results.boxes 또는 results.boxes.xyxy 로 bbox 접근 (ultralytics 버전에 따라 차이)
    if hasattr(results, "boxes"):
        boxes = results.boxes
        for box in boxes:
            # box.xyxy 에서 좌표 추출
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy[0].numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0]) if hasattr(box, 'conf') else (box.conf if hasattr(box, 'conf') else 0)
            # 클래스 검사: 번호판 클래스라면 (만약 여러분이 모델을 번호판으로 학습)
            cls = int(box.cls[0]) if hasattr(box, 'cls') else None

            # 임시: 모든 검출 박스에 대해 OCR 시도 (실제론 cls 체크)
            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            pre = preprocess_plate(plate_img)
            text = ocr_tesseract(pre)
            # 시각화
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("ANPR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
