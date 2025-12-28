from facenet_pytorch import InceptionResnetV1
import torch

class Embedder:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def get_embedding(self, face_tensor):
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(face_tensor)
        return embedding.cpu().numpy()[0]
    
if __name__=='__main__':
    emb = Embedder()
    xa = emb.get_embedding()