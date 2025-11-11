import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import time

# 1) YOLO 모델 불러오기
model = YOLO("/home/deepet/VSCode/xyz_2st_project/sample/weight/Vehicle_number.pt")

# 2) OCR 전처리 (숫자에 맞게)
def preprocess_ocr_roi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # th = cv2.bitwise_not(th)  # 필요하면 반전
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

cap = cv2.VideoCapture('/dev/video4')

# ✅ bbox 잠금 관련 변수
active_bbox = None          # (x1, y1, x2, y2)
lock_until = 0.0            # 이 시간까지 bbox 고정
LOCK_DURATION = 5.0        # 초 단위

# 번호판 클래스만 쓰고 싶으면 여기 세팅 (없으면 None)
PLATE_CLASS_ID = None   # 예: 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    now = time.time()

    # 10초가 지났으면 bbox 잠금 해제
    if now > lock_until:
        active_bbox = None

    # 항상 YOLO는 돌리되, bbox는 "잠금이 비어 있을 때만" 갱신
    results = model(frame, imgsz=640)[0]

    if active_bbox is None:
        # 새로 잠글 bbox를 찾는다
        if hasattr(results, "boxes") and results.boxes is not None:
            best_box = None
            best_conf = 0.0

            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())

                # 번호판 class 필터
                if PLATE_CLASS_ID is not None and cls_id != PLATE_CLASS_ID:
                    continue

                # 너무 작은 박스는 무시
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                if conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)

            # 가장 좋은 박스를 잠금
            if best_box is not None:
                active_bbox = best_box
                lock_until = now + LOCK_DURATION
                # print(f"bbox 잠금! {LOCK_DURATION}초 동안 유지: {active_bbox}")

    text = ""

    # 🔒 active_bbox가 있을 때만 OCR 수행
    if active_bbox is not None:
        x1, y1, x2, y2 = active_bbox

        # 프레임 경계 체크
        h, w, _ = frame.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        roi = frame[y1:y2, x1:x2]
        if roi.size != 0:
            th_big = preprocess_ocr_roi(roi)
            text = ocr_digits(th_big)
            print("인식된 텍스트:", text)

            # bbox + 텍스트 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
    else:
        # 잠금된 bbox가 없는 상태라는 표시 (옵션)
        cv2.putText(frame, "No locked target",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

    # 남은 잠금 시간 표시 (옵션)
    if lock_until > now:
        remain = int(lock_until - now)
        cv2.putText(frame, f"Lock: {remain}s",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2)

    cv2.imshow("YOLO + OCR (Locked 10s)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
