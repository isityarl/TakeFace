import cv2
import numpy as np
import torch
import pandas as pd

from src.camera.webcam import Webcam
from src.detection.detect import FaceDetector
from src.recognition.embedder import Embedder

def preproccess(face):
    face_img = face
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype(np.float32)
    face_img = (face_img - 127.5) / 128.0
    face_img = np.transpose(face_img, (2,0,1))

    face_tensor = torch.from_numpy(face_img)


    return face_tensor


def reg():
    embedder = Embedder()
    cam = Webcam(camera_id=0, width=640, height=480)
    detector = FaceDetector()

    embeddings = []
    while len(embeddings) < 20:
        frame = cam.read()
        if frame is None:
            print("Failed to read from camera")
            break

        boxes = detector.detect(frame)
        
        if not boxes: continue
        x1, y1, x2, y2 = boxes[0]
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face_img = frame[y1:y2, x1:x2]
        face_tensor = preproccess(face_img)
        emb = embedder.get_embedding(face_tensor)
        embeddings.append(emb)
        print(f"Embedding {len(embeddings)}/20")

    final = np.mean(embeddings, axis=0)
    np.save('data/embeddings/face2.npy', final)
    print('saved')

if __name__=='__main__':
    reg()