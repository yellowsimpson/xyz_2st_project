import cv2
import pytesseract

# 1) 이미지 읽기
img = cv2.imread("/home/deepet/VSCode/xyz_2st_project/sample/img/text_img4.png")

cap = cv2.VideoCapture(0)  # 웹캠 대신 비디오 파일 경로도 가능

# 2) 흑백 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3) 이진화(배경/글자 분리 – OCR 인식 잘 되게)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4) OCR 실행 (한글+영어)
text = pytesseract.image_to_string(thresh, lang="kor+eng")

# print("인식된 텍스트:")
print(text)
