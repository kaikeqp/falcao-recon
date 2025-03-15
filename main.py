import numpy as np
import pandas as pd
import datetime
import os
import faiss
import cv2
import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Dict
import insightface
from insightface.app import FaceAnalysis
from pathlib import Path
import uvicorn
import onnxruntime
import sqlite3
from uuid import uuid4
from datetime import datetime

# =============== CONFIGURAÇÕES ===============
CONFIG = {
    "threshold_reconhecimento": 0.45,
    "diretorio_rostos": "img_dbv",
    "dimensao_vetor": 512,
    "escala_frame": 0.5,
    "use_gpu": False,  # Alterado para False para usar CPU
    "det_size": (640, 640),
    "model_path": "/home/ubuntu/.insightface",
}

# =============== CONFIGURAÇÕES ADICIONAIS ===============
CONFIG.update({
    "database_path": "face_recognition.db",
    "recognized_images_dir": "recognized_images"
})

# =============== INICIALIZAÇÃO DO BANCO DE DADOS ===============
def init_db():
    os.makedirs(CONFIG["recognized_images_dir"], exist_ok=True)
    conn = sqlite3.connect(CONFIG["database_path"])
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS recognition_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  nome TEXT,
                  data_hora DATETIME,
                  imagem_path TEXT,
                  confianca REAL)''')
    conn.commit()
    conn.close()

init_db()

# =============== INICIALIZAÇÃO DO INSIGHTFACE ===============
# Configurar o ONNX Runtime para usar apenas a CPU

app = FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'], root=CONFIG["model_path"])
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

                save_recognition_log(nome, distances[0][0], img)
                
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

@api.get("/registers")
async def get_access_logs(
    ano: int = None,
    mes: int = None,
    dia: int = None
):
    try:
        conn = sqlite3.connect(CONFIG["database_path"])
        c = conn.cursor()
        
        # Query base
        query = '''SELECT id, nome, 
                  strftime('%d/%m/%Y %H:%M:%S', data_hora) as data_formatada,
                  confianca 
                  FROM recognition_logs'''
        
        params = []
        conditions = []
        
        # Adicionar filtros
        if ano:
            conditions.append("strftime('%Y', data_hora) = ?")
            params.append(f"{ano:04d}")
            if mes:
                conditions.append("strftime('%m', data_hora) = ?")
                params.append(f"{mes:02d}")
                if dia:
                    conditions.append("strftime('%d', data_hora) = ?")
                    params.append(f"{dia:02d}")
        
        # Montar query final
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY id DESC"
        
        c.execute(query, params)
        
        logs = []
        for row in c.fetchall():
            logs.append({
                "id": row[0],
                "nome": row[1],
                "data": row[2],
                "confianca": f"{row[3]*100:.2f}%",
                "url_imagem": f"/images/{row[0]}"
            })
        
        conn.close()
        return JSONResponse(content={"logs": logs})
    
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Erro no banco de dados: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/export-excel")
async def export_excel(
    ano: int = None,
    mes: int = None,
    dia: int = None
):
    try:
        conn = sqlite3.connect(CONFIG["database_path"])
        
        # Construir query com filtros de data
        base_query = '''SELECT 
                        id,
                        nome,
                        strftime('%d/%m/%Y %H:%M:%S', data_hora) as data_hora,
                        confianca 
                        FROM recognition_logs'''
        
        where_clauses = []
        params = []
        
        # Adicionar filtros
        if ano:
            where_clauses.append("strftime('%Y', data_hora) = ?")
            params.append(f"{ano:04d}")
            if mes:
                where_clauses.append("strftime('%m', data_hora) = ?")
                params.append(f"{mes:02d}")
                if dia:
                    where_clauses.append("strftime('%d', data_hora) = ?")
                    params.append(f"{dia:02d}")
        
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)
        
        # Executar query
        df = pd.read_sql_query(base_query, conn, params=params if params else None)
        
        # Processar dados
        df['Confiança (%)'] = df['confianca'] * 100
        
        # Renomear colunas
        df = df.rename(columns={
            'id': 'ID',
            'nome': 'Nome',
            'data_hora': 'Data/Hora'
        })
        
        # Remover coluna original de confiança
        df = df.drop(columns=['confianca'])
        
        # Salvar Excel
        df.to_excel("logs.xlsx", index=False, engine='openpyxl')
        
        return FileResponse(
            "logs.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=f"logs_{datetime.now().strftime('%Y%m%d')}.xlsx"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Parâmetros de data inválidos")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# =============== ENDPOINT PARA OBTER IMAGEM ===============
@api.get("/images/{image_id}")
async def get_image(image_id: int):
    try:
        conn = sqlite3.connect(CONFIG["database_path"])
        c = conn.cursor()
        
        # Buscar caminho da imagem
        c.execute('''SELECT imagem_path FROM recognition_logs WHERE id = ?''', (image_id,))
        result = c.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Registro não encontrado")
        
        img_path = result[0]
        
        if not os.path.exists(img_path):
            raise HTTPException(status_code=404, detail="Imagem não encontrada")
        
        return FileResponse(img_path, media_type="image/jpeg")
    
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Erro no banco de dados: {str(e)}")
    finally:
        conn.close()

# =============== FUNÇÕES DE LOG ===============
def save_recognition_log(nome: str, confidence: float, image: np.ndarray):
    conn = sqlite3.connect(CONFIG["database_path"])
    c = conn.cursor()
    
    try:
        # Salvar imagem
        os.makedirs(CONFIG["recognized_images_dir"], exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:6]}.jpg"
        img_path = os.path.join(CONFIG["recognized_images_dir"], filename)
        cv2.imwrite(img_path, image)

        # Compactação otimizada da imagem
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 40]  # 0-100 (quanto menor, mais compactado)
        success, buf = cv2.imencode('.jpg', image, encode_param)

        if success:
            with open(img_path, 'wb') as f:
                f.write(buf.tobytes())
        else:
            raise Exception("Falha na compactação da imagem")
        
        # Redução adicional de resolução para imagens muito grandes
        height, width = image.shape[:2]
        if width > 300:
            scale = 300 / width
            resized = cv2.resize(image, (int(width*scale), int(height*scale)))
            success, buf = cv2.imencode('.jpg', resized, encode_param)
            if success:
                with open(img_path, 'wb') as f:
                    f.write(buf.tobytes())
        
        # Salvar no banco
        c.execute('''INSERT INTO recognition_logs 
                     (nome, data_hora, imagem_path, confianca)
                     VALUES (?, ?, ?, ?)''',
                  (nome, datetime.now(), img_path, float(confidence)))
        conn.commit()
    except Exception as e:
        print(f"Erro ao salvar log: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)
