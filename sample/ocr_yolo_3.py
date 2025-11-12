import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import time
import os

# 1) YOLO 모델 불러오기
model = YOLO("/home/deepet/VSCode/xyz_2st_project/sample/weight/Vehicle_number.pt")

# 2) OCR 전처리 (숫자에 맞게)
def preprocess_ocr_roi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    th_big = cv2.resize(th, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    return th_big

# 3) Tesseract로 숫자 OCR
def ocr_digits(img):
    # 숫자만
    custom_config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(img, lang="eng", config=custom_config)
    return text.strip()

cap = cv2.VideoCapture('/dev/video4')

# ✅ bbox 잠금 관련 변수
active_bbox = None          # (x1, y1, x2, y2)
lock_until = 0.0            # 이 시간까지 bbox 고정

# ✅ 5초 주기로 새로 캡처 & OCR
CAPTURE_INTERVAL = 5.0
LOCK_DURATION = CAPTURE_INTERVAL  # 잠금 시간과 캡처 주기를 동일하게
last_capture_time = 0.0

# 번호판 클래스만 쓰고 싶으면 여기 세팅 (없으면 None)
PLATE_CLASS_ID = None       # 예: 2

plate_text = ""             # 현재 표시 중인 OCR 결과

# 결과 저장 폴더(선택)
os.makedirs("plates", exist_ok=True)

def find_best_box(results, plate_class_id=None, min_w=20, min_h=20):
    """가장 신뢰도 높은 박스 선택 (필요 시 클래스 필터/최소 크기 필터 적용)"""
    best_box = None
    best_conf = 0.0
    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())

            if plate_class_id is not None and cls_id != plate_class_id:
                continue

            if (x2 - x1) < min_w or (y2 - y1) < min_h:
                continue

            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, x2, y2)
    return best_box

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    now = time.time()

    # 🔁 5초마다: 잠금이 끝났으면 다시 탐지 → 즉시 캡처 & OCR
    if now >= lock_until:
        # 새 라운드: YOLO로 최선의 박스 다시 선택
        results = model(frame, imgsz=640)[0]
        best_box = find_best_box(results, plate_class_id=PLATE_CLASS_ID)

        if best_box is not None:
            active_bbox = best_box
            lock_until = now + LOCK_DURATION  # 다음 5초 동안 고정

            # 즉시 ROI 캡처 + OCR
            x1, y1, x2, y2 = active_bbox
            h, w, _ = frame.shape
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            roi = frame[y1:y2, x1:x2]
            if roi.size != 0:
                # 파일로도 저장(원하면)
                ts = int(now)
                save_path = f"plates/captured_plate_{ts}.jpg"
                cv2.imwrite(save_path, roi)
                print(f"번호판 이미지 저장: {save_path}")

                th_big = preprocess_ocr_roi(roi)
                plate_text = ocr_digits(th_big)
                last_capture_time = now
                print(f"[{time.strftime('%H:%M:%S')}] OCR 결과: {plate_text if plate_text else '(빈 문자열)'}")
        else:
            # 탐지 실패 시 다음 주기에 재시도
            active_bbox = None
            lock_until = now + 1.0  # 너무 자주 도는 걸 방지(1초 뒤 재시도)

    # ==== 화면 표시 부분 ====
    if active_bbox is not None:
        x1, y1, x2, y2 = active_bbox
        h, w, _ = frame.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 최신 OCR 결과 표시
        if plate_text:
            cv2.putText(frame, plate_text, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No target", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 남은 잠금 시간/다음 캡처까지 남은 시간 표시
    remain_lock = max(0, int(lock_until - now))
    remain_capture = max(0, int(CAPTURE_INTERVAL - (now - last_capture_time))) if last_capture_time > 0 else "init"
    cv2.putText(frame, f"Lock: {remain_lock}s | Next capture in: {remain_capture}s",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO + OCR (Capture every 5s)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
