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
    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    th_big = cv2.resize(
        th, None,
        fx=2.0, fy=2.0,
        interpolation=cv2.INTER_LINEAR
    )
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
LOCK_DURATION = 5.0         # 초 단위

# 번호판 클래스만 쓰고 싶으면 여기 세팅 (없으면 None)
PLATE_CLASS_ID = None       # 예: 2

# ★ 한 번만 캡쳐 & 인식하기 위한 변수
plate_captured = False      # 이미 캡쳐했는지 여부
plate_text = ""             # 캡쳐된 이미지에서 읽은 차량 번호

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    now = time.time()

    # 잠금 해제 (옵션)
    if now > lock_until:
        active_bbox = None

    # ★ 이미 번호판을 한 번 캡쳐했다면,
    #   더 이상 YOLO 돌리지 않고 그냥 화면에만 표시해도 된다.
    #   (원하면 계속 YOLO 돌려도 되지만, 질문 의도상 한 번만 인식)
    if not plate_captured:
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

                # 가장 좋은 박스를 잠금 + ★캡쳐
                if best_box is not None:
                    active_bbox = best_box
                    lock_until = now + LOCK_DURATION

                    # ★ 여기서 한 번만 ROI 캡쳐 + OCR 실행
                    x1, y1, x2, y2 = active_bbox

                    h, w, _ = frame.shape
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h))

                    roi = frame[y1:y2, x1:x2]

                    if roi.size != 0:
                        # ★ 캡쳐 이미지 파일로 저장 (원하면)
                        cv2.imwrite("captured_plate.jpg", roi)
                        print("번호판 이미지 저장: captured_plate.jpg")

                        # ★ 저장된(혹은 메모리 상의) 이미지에서 한 번만 OCR
                        th_big = preprocess_ocr_roi(roi)
                        plate_text = ocr_digits(th_big)
                        print("최초 인식된 차량 번호:", plate_text)

                        plate_captured = True  # 이후에는 다시 OCR 안 함

                        # 번호 한 번만 읽고 카메라 종료하고 싶으면:
                        # break

    # ==== 화면 표시 부분 ====
    if active_bbox is not None:
        x1, y1, x2, y2 = active_bbox
        h, w, _ = frame.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)

        # ★ plate_captured 후에는 계속 같은 번호만 표시
        if plate_text:
            cv2.putText(frame, plate_text,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
    else:
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

    cv2.imshow("YOLO + OCR (Capture Once)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
