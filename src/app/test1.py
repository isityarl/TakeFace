import cv2
import numpy as np

from src.camera.webcam import Webcam
from src.detection.detect import FaceDetector
from src.recognition.embedder import Embedder
from src.recognition.register import preproccess


def main():
    embedder = Embedder()
    cam = Webcam(camera_id=0, width=640, height=480)
    detector = FaceDetector()

    known_embeddings = {
        'Yernar': np.load('data/embeddings/face1.npy'),
        'Nurbek': np.load('data/embeddings/face2.npy')
    }    
    threshold = 0.75

    while True:
        frame = cam.read()
        if frame is None:
            print("Failed to read from camera")
            break

        boxes = detector.detect(frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_img = frame[y1:y2, x1:x2]
            face_tensor = preproccess(face_img)
            emb = embedder.get_embedding(face_tensor)

            for user, embb in known_embeddings.items():
                sim = np.dot(emb, embb) / (np.linalg.norm(emb) * np.linalg.norm(embb))
                if sim > threshold:
                    print(f'{user} recognized, sim={sim:.2f}')

        cv2.imshow("MediaPipe Face Detection", frame_bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()   


if __name__ == "__main__":
    main()
