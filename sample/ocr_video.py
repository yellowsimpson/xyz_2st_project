import cv2
import pytesseract

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    # ===== ROI: 일단 중앙 사각형만 사용 (나중에 YOLO bbox로 교체) =====
    h, w, _ = frame.shape
    roi_size = min(h, w) // 2
    cx, cy = w // 2, h // 2
    x1 = cx - roi_size // 2
    y1 = cy - roi_size // 2
    x2 = cx + roi_size // 2
    y2 = cy + roi_size // 2

    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ===== 1) 그레이 변환 =====
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 노이즈 조금 제거 (필기 흔들림 줄이기)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ===== 2) 이진화 + 필요하면 색 반전 =====
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 글씨가 검은색, 배경이 흰색이면 그대로 쓰고
    # 반대로 나오면 아래 한 줄 켜서 반전해보기
    # th = cv2.bitwise_not(th)

    # ===== 3) OCR에 잘 보이도록 확대 =====
    th_big = cv2.resize(th, None, fx=2.0, fy=2.0,
                        interpolation=cv2.INTER_LINEAR)

    # ===== 4) Tesseract 설정 (숫자 위주라면) =====
    # 한 줄 텍스트 가정 + 숫자만 허용
    custom_config = "--psm 7 -c tessedit_char_whitelist=0123456789"

    text = pytesseract.image_to_string(
        th_big,
        lang="eng",          # 숫자만이면 eng로 충분. (한글 필요하면 kor+eng)
        config=custom_config
    ).strip()

    print("인식된 텍스트:", text)

    # 결과를 원본 이미지 위에 표시
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("OCR ROI Camera", frame)
    #cv2.imshow("THRESH", th_big)  # 디버깅용으로 보면 좋음

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
