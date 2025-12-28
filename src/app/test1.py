import cv2

from src.camera.webcam import Webcam
from src.detection.detect import FaceDetector


def main():
    cam = Webcam(camera_id=0, width=640, height=480)
    detector = FaceDetector()

    while True:
        frame_rgb = cam.read()
        if frame_rgb is None:
            print("Failed to read from camera")
            break

        boxes = detector.detect(frame_rgb)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(
                frame_bgr,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        cv2.imshow("MediaPipe Face Detection", frame_bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()


if __name__ == "__main__":
    main()
