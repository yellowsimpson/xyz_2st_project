import cv2
import pytesseract

# ✅ 1) 이미지 로드 (임시로 파일에서 읽기)
img = cv2.imread("/home/deepet/VSCode/xyz_2st_project/sample/img/text_img4.png")
if img is None:
    raise FileNotFoundError("이미지를 불러올 수 없습니다. 경로를 확인하세요.")

frame = img.copy()

# ===== ROI: 중앙 사각형만 사용 (나중에 YOLO bbox로 교체) =====
h, w, _ = frame.shape
roi_size = int(min(h, w) * 0.75)

cx, cy = w // 2, h // 2
x1 = cx - roi_size // 2
y1 = cy - roi_size // 2
x2 = cx + roi_size // 2
y2 = cy + roi_size // 2

# 이미지 경계 초과 방지
x1 = max(x1, 0)
y1 = max(y1, 0)
x2 = min(x2, w)
y2 = min(y2, h)

roi = frame[y1:y2, x1:x2]
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ===== 1) 그레이 변환 =====
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# 노이즈 제거
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# ===== 2) 이진화 + 필요하면 색 반전 =====
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 글씨/배경 반전 필요하면 아래 한 줄 활성화
# th = cv2.bitwise_not(th)

# ===== 3) OCR에 잘 보이도록 확대 =====
th_big = cv2.resize(th, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

# ===== 4) Tesseract 설정 (숫자 위주라면) =====
custom_config = "--psm 7 -c tessedit_char_whitelist=0123456789"

text = pytesseract.image_to_string(
    th_big,
    lang="eng",
    config=custom_config
).strip()

print("인식된 텍스트:", text)

# 결과를 원본 이미지 위에 표시
cv2.putText(frame, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 시각화
cv2.imshow("OCR ROI Image", frame)
cv2.imshow("Threshold", th_big)

cv2.waitKey(0)
cv2.destroyAllWindows()
