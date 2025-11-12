import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class YoloOcrNode(Node):
    def __init__(self):
        super().__init__("yolo_ocr_node")

        # ===== ROS2 퍼블리셔 =====
        self.pub_ocr = self.create_publisher(String, "/ocr_text", 10)

        # ===== YOLO 모델 로드 =====
        self.model = YOLO("/home/deepet/VSCode/xyz_2st_project/sample/weight/Vehicle_number.pt")

        # ===== 카메라 =====
        self.cap = cv2.VideoCapture('/dev/video4')

        # ===== OCR/잠금 관련 변수 =====
        self.active_bbox = None
        self.lock_until = 0.0
        self.LOCK_DURATION = 5.0  # 초 단위

        self.PLATE_CLASS_ID = None  # 특정 클래스만 인식할 때 설정
        self.plate_captured = False
        self.plate_text = ""

    # ===== OCR 전처리 =====
    def preprocess_ocr_roi(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        th_big = cv2.resize(th, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        return th_big

    # ===== 숫자 OCR =====
    def ocr_digits(self, img):
        custom_config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(img, lang="eng", config=custom_config)
        return text.strip()

    # ===== 루프 =====
    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("프레임을 읽을 수 없습니다.")
            return False

        now = time.time()

        # 잠금 해제
        if now > self.lock_until:
            self.active_bbox = None

        # 번호판 한 번만 인식
        if not self.plate_captured:
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

                    x1, y1, x2, y2 = self.active_bbox
                    h, w, _ = frame.shape
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h))

                    roi = frame[y1:y2, x1:x2]
                    if roi.size != 0:
                        cv2.imwrite("captured_plate.jpg", roi)
                        self.get_logger().info("번호판 이미지 저장: captured_plate.jpg")

                        th_big = self.preprocess_ocr_roi(roi)
                        self.plate_text = self.ocr_digits(th_big)
                        self.get_logger().info(f"최초 인식된 차량 번호: {self.plate_text}")

                        # ===== ROS2 토픽으로 퍼블리시 =====
                        if self.plate_text != "":
                            msg = String()
                            msg.data = self.plate_text
                            self.pub_ocr.publish(msg)
                            self.get_logger().info(f"OCR 텍스트 퍼블리시: {self.plate_text}")

                        self.plate_captured = True

        # ===== 시각화 =====
        if self.active_bbox is not None:
            x1, y1, x2, y2 = self.active_bbox
            h, w, _ = frame.shape
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.plate_text:
                cv2.putText(frame, self.plate_text, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No locked target", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if self.lock_until > now:
            remain = int(self.lock_until - now)
            cv2.putText(frame, f"Lock: {remain}s", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("YOLO + OCR (Capture Once, ROS2)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    # ===== 종료 처리 =====
    def clean_up(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.destroy_node()


def main():
    rclpy.init()
    node = YoloOcrNode()
    try:
        while rclpy.ok():
            if not node.loop():
                break
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        node.clean_up()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
