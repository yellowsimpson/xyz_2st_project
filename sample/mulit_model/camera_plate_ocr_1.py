# camera_plate_ocr.py (깔끔 버전)

import sys
import os
import cv2
import easyocr
import time


def main():

    reader = easyocr.Reader(['ko', 'en'])  # GPU: reader = easyocr.Reader(['ko', 'en'], gpu=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다. 장치 번호(0, 1, 2...)를 바꿔보세요.")
        return

    print("[INFO] 카메라가 열렸습니다.")
    print("[INFO] 'c' 키: 현재 화면에서 번호판(문자) 인식 시도")
    print("[INFO] 'q' 키: 프로그램 종료")

    last_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽지 못했습니다.")
            break

        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 'c' to OCR, 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if last_text:
            cv2.putText(display_frame, f"Last plate: {last_text}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Camera - Plate OCR", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            print("[INFO] 현재 프레임에서 OCR 시도 중...")
            t0 = time.time()

            results = reader.readtext(frame)

            if not results:
                print("[INFO] 문자를 인식하지 못했습니다.")
                last_text = ""
            else:
                best_text = ""
                best_conf = 0.0
                for (bbox, text, conf) in results:
                    if conf > best_conf:
                        best_conf = conf
                        best_text = text

                last_text = best_text
                print(f"[RESULT] 인식된 텍스트: {best_text} (conf={best_conf:.2f})")
                print(f"[INFO] OCR 처리 시간: {time.time() - t0:.2f}초")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
