from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io
import base64
import torch  # se seu modelo for em PyTorch
from pathlib import Path

# =====================================
# 🚀 Inicialização
# =====================================
app = FastAPI(title="IAneural API", version="1.0")

# CORS - liberar acesso do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste se quiser restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# 📦 Modelos de requisição
# =====================================
class GenerateRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None

class BatchRequest(BaseModel):
    prompts: List[str]

# =====================================
# 🧠 Carregamento do modelo
# =====================================
# ⚠️ Substitua aqui com o carregamento do seu modelo treinado
# Exemplo PyTorch:
MODEL_PATH = Path("outputs/training/model_epoch_019.pth")

def load_model():
    # Exemplo: suponha que você tenha uma classe Generator()
    # from models import Generator
    # model = Generator()
    # model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    # model.eval()
    # return model
    #
    # Por enquanto, placeholder:
    return None

model = load_model()

# =====================================
# 🎨 Função de geração de imagem
# =====================================
def generate_image_from_model(prompt: str, seed: Optional[int] = None):
    """
    Gera uma imagem usando o modelo real (substituir o placeholder).
    """
    # Se quiser consistência entre execuções
    if seed is not None:
        torch.manual_seed(seed)

    # 👉 Aqui você chama seu modelo real
    # Por exemplo:
    # img_tensor = model.generate(prompt)
    # img = tensor_to_pil(img_tensor)

    # 🔴 Placeholder temporário
    color = (255, 0, 0)
    img = Image.new("RGB", (64, 64), color=color)
    return img

# =====================================
# ⚙️ Funções auxiliares
# =====================================
def pil_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# =====================================
# 🌐 Endpoints
# =====================================
@app.get("/health")
async def health():
    return {"message": "Servidor funcionando!"}

@app.get("/model-info")
async def model_info():
    return {"conditional_mode": True, "model_loaded": model is not None}

@app.get("/sample-prompts")
async def sample_prompts():
    return {
        "prompts": [
            "a flying airplane",
            "a red car",
            "a cute dog",
            "a colorful bird"
        ]
    }

@app.post("/generate")
async def generate(req: GenerateRequest):
    img = generate_image_from_model(req.prompt, req.seed)
    encoded = pil_to_base64(img)
    return {"image": encoded, "prompt": req.prompt, "message": "Imagem gerada!"}

@app.post("/generate-batch")
async def generate_batch(req: BatchRequest):
    results = []
    for p in req.prompts:
        img = generate_image_from_model(p)
        results.append({"prompt": p, "image": pil_to_base64(img)})
    return {"images": results, "count": len(results), "message": "Batch gerado!"}

# =====================================
# ▶️ Execução local
# =====================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
