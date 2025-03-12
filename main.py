import numpy as np
import pandas as pd
import datetime
import os
import faiss
import cv2
import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
import insightface
from insightface.app import FaceAnalysis
from pathlib import Path
import uvicorn
import onnxruntime

# =============== CONFIGURAÇÕES ===============
CONFIG = {
    "threshold_reconhecimento": 0.45,
    "diretorio_rostos": "img_dbv",
    "dimensao_vetor": 512,
    "escala_frame": 0.5,
    "use_gpu": False,  # Alterado para False para usar CPU
    "det_size": (640, 640),
    "model_path": os.path.join(""),
}

# =============== INICIALIZAÇÃO DO INSIGHTFACE ===============
# Configurar o ONNX Runtime para usar apenas a CPU

app = FaceAnalysis(providers=['CPUExecutionProvider'], root=CONFIG["model_path"])
app.prepare(ctx_id=-1, det_size=CONFIG["det_size"])

# =============== GERENCIADOR DE BANCO DE DADOS FACIAL ===============
class FaceDatabaseManager:
    def __init__(self):
        self.index = None
        self.labels = []
        self.carregar_banco_dados()

    def carregar_rostos(self) -> Dict:
        faces = {}
        try:
            for arquivo in os.listdir(CONFIG["diretorio_rostos"]):
                if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                    caminho = os.path.join(CONFIG["diretorio_rostos"], arquivo)
                    nome = os.path.splitext(arquivo)[0]
                    
                    img = cv2.imdecode(np.fromfile(caminho, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    
                    faces_detected = app.get(img)
                    if faces_detected:
                        faces[nome] = faces_detected[0].embedding
        except Exception as e:
            print(f"Erro ao carregar rostos: {e}")
        return faces

    def carregar_banco_dados(self):
        faces = self.carregar_rostos()
        if not faces:
            print("Nenhum rosto cadastrado encontrado")
            return

        self.labels = list(faces.keys())
        encodings = np.array(list(faces.values()), dtype="float32")
        faiss.normalize_L2(encodings)

        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(CONFIG["dimensao_vetor"]))
        self.index.add_with_ids(encodings, np.arange(len(self.labels)))

    def adicionar_rosto(self, nome: str, embedding: np.ndarray):
        embedding = np.array(embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        new_id = len(self.labels)
        self.index.add_with_ids(embedding, np.array([new_id]))
        self.labels.append(nome)

# =============== API SETUP ===============
api = FastAPI(title="Face Recognition API")
db_manager = FaceDatabaseManager()

# =============== ENDPOINTS ===============
@api.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    try:
        # Processar imagem recebida
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detectar rostos
        faces_detected = app.get(img)
        if not faces_detected:
            raise HTTPException(status_code=400, detail="Nenhum rosto detectado na imagem")
        
        # Processar primeiro rosto encontrado
        face = faces_detected[0]
        embedding = face.embedding
        
        # Reconhecer rosto
        embedding = np.array(embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        distances, indices = db_manager.index.search(embedding, 3)
        if distances[0][0] > CONFIG["threshold_reconhecimento"]:
            if (distances[0][0] - distances[0][1]) > 0.05:
                nome = db_manager.labels[indices[0][0]]
                return JSONResponse(content={
                    "nome": nome,
                    "confianca": float(distances[0][0]),
                    "status": "reconhecido"
                })
        
        return JSONResponse(content={
            "status": "desconhecido",
            "confianca": float(distances[0][0]) if len(distances) > 0 else 0.0
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/register")
async def register_face(nome: str, file: UploadFile = File(...)):
    try:
        # Validar nome
        if not nome or len(nome) < 2:
            raise HTTPException(status_code=400, detail="Nome inválido")
        
        # Processar imagem
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detectar rosto
        faces_detected = app.get(img)
        if not faces_detected:
            raise HTTPException(status_code=400, detail="Nenhum rosto detectado na imagem")
        
        # Salvar imagem no banco de dados
        face_dir = CONFIG["diretorio_rostos"]
        os.makedirs(face_dir, exist_ok=True)
        filename = f"{nome}.jpg"
        filepath = os.path.join(face_dir, filename)
        cv2.imwrite(filepath, img)
        
        # Atualizar índice FAISS
        embedding = faces_detected[0].embedding
        db_manager.adicionar_rosto(nome, embedding)
        
        return JSONResponse(content={
            "status": "sucesso",
            "mensagem": f"Rosto de {nome} cadastrado com sucesso"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/logs")
async def get_access_logs():
    try:
        if not os.path.exists("log_acesso.xlsx"):
            return JSONResponse(content={"logs": []})
        
        df = pd.read_excel("log_acesso.xlsx")
        return JSONResponse(content={
            "logs": df.to_dict(orient="records")
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)
