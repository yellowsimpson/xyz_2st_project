# image_plate_ocr.py
# 사용법:
#   python3 image_plate_ocr.py plate_test.jpg
#
# plate_test.jpg 같은 이미지에서 번호판(또는 글자)을 읽어서
# 터미널에 인식 결과를 출력하는 스크립트

import sys
import os
import cv2
import easyocr


def main():
    # 1) 이미지 경로 받기
    if len(sys.argv) < 2:
        print("[사용법] python3 image_plate_ocr.py <이미지_파일경로>")
        print("예) python3 image_plate_ocr.py plate_test.jpg")
        return

    img_path = sys.argv[1]

    if not os.path.exists(img_path):
        print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {img_path}")
        return

    # 2) 이미지 로드
    print(f"[INFO] 이미지 로딩 중: {img_path}")
    image = cv2.imread(img_path)

    if image is None:
        print("[ERROR] 이미지를 읽을 수 없습니다. 파일이 깨졌거나 형식이 이상할 수 있습니다.")
        return

    # 3) EasyOCR 로더
    print("[INFO] EasyOCR 모델 로딩 중... (처음 한 번은 시간 좀 걸릴 수 있어요)")
    reader = easyocr.Reader(['ko', 'en'])  # GPU 있으면 gpu=True 로 변경 가능

    # 4) OCR 수행
    print("[INFO] OCR 시도 중...")
    results = reader.readtext(image)

    if not results:
        print("[INFO] 문자를 인식하지 못했습니다.")
        return

    print("[INFO] 인식 결과 목록:")
    for i, (bbox, text, conf) in enumerate(results):
        print(f"  [{i}] text='{text}', conf={conf:.2f}")

    # 5) 가장 신뢰도 높은 텍스트 하나만 뽑고 싶다면:
    best_text = ""
    best_conf = 0.0
    for (bbox, text, conf) in results:
        if conf > best_conf:
            best_conf = conf
            best_text = text

    print()
    print(f"[RESULT] 최종 선택 텍스트: '{best_text}' (conf={best_conf:.2f})")


if __name__ == "__main__":
    main()
