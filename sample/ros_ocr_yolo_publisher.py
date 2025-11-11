import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ===== 1) ROS2 노드 정의 =====
class YoloOcrNode(Node):
    def __init__(self):
        super().__init__("yolo_ocr_node")

        # 퍼블리셔: 인식된 텍스트를 /ocr_text 로 보냄
        self.pub_ocr = self.create_publisher(String, "/ocr_text", 10)

        # YOLO 모델 로드
        self.model = YOLO("/home/deepet/VSCode/xyz_1st_project/xyz_1st_project/weight/car_detect.pt")

        # bbox 잠금 관련 변수
        self.active_bbox = None       # (x1, y1, x2, y2)
        self.lock_until = 0.0
        self.LOCK_DURATION = 10.0     # 초 단위

        # 번호판 class id가 정해져 있으면 설정 (예: 2)
        self.PLATE_CLASS_ID = None    # 번호판 class만 쓰고 싶으면 숫자로 변경

        # 카메라
        self.cap = cv2.VideoCapture(0)

    # OCR 전처리
    def preprocess_ocr_roi(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 필요 시 반전
        # th = cv2.bitwise_not(th)
        th_big = cv2.resize(th, None, fx=2.0, fy=2.0,
                            interpolation=cv2.INTER_LINEAR)
        return th_big

    # Tesseract 숫자 인식
    def ocr_digits(self, img):
        custom_config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(
            img,
            lang="eng",
            config=custom_config
        )
        return text.strip()

    def loop(self):
        # 한 프레임 처리 루프
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("프레임을 읽을 수 없습니다. 종료합니다.")
            return False

        now = time.time()

        # 10초 지나면 bbox 잠금 해제
        if now > self.lock_until:
            self.active_bbox = None

        # YOLO 추론 (항상 돌리되, active_bbox 없을 때만 새로 잠금)
        results = self.model(frame, imgsz=640)[0]

        if self.active_bbox is None:
            best_box = None
            best_conf = 0.0

            if hasattr(results, "boxes") and results.boxes is not None:
                for box in results.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(box.conf[0].cpu().item())
                    cls_id = int(box.cls[0].cpu().item())

                    # 번호판 클래스만 쓰고 싶을 때
                    if self.PLATE_CLASS_ID is not None and cls_id != self.PLATE_CLASS_ID:
                        continue

                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue

                    if conf > best_conf:
                        best_conf = conf
                        best_box = (x1, y1, x2, y2)

            if best_box is not None:
                self.active_bbox = best_box
                self.lock_until = now + self.LOCK_DURATION
                self.get_logger().info(f"bbox 잠금: {self.active_bbox}, {self.LOCK_DURATION}초 유지")

        text = ""

        if self.active_bbox is not None:
            x1, y1, x2, y2 = self.active_bbox

            h, w, _ = frame.shape
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            roi = frame[y1:y2, x1:x2]
            if roi.size != 0:
                th_big = self.preprocess_ocr_roi(roi)
                text = self.ocr_digits(th_big)

                # ROS2로 퍼블리시
                if text != "":
                    msg = String()
                    msg.data = text
                    self.pub_ocr.publish(msg)
                    self.get_logger().info(f"OCR 텍스트 퍼블리시: {text}")

                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No locked target", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 남은 시간 표시
        if self.lock_until > now:
            remain = int(self.lock_until - now)
            cv2.putText(frame, f"Lock: {remain}s", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255), 2)

        cv2.imshow("YOLO + OCR (Locked 10s, ROS2)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        return True

    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.destroy_node()


def main():
    rclpy.init()
    node = YoloOcrNode()

    try:
        while rclpy.ok():
            if not node.loop():   # loop가 False를 리턴하면 종료
                break
            # 콜백 처리 (서비스/타이머 등 있을 경우)
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
