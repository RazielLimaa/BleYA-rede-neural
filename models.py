from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import torch
import io
import base64
from torchvision.transforms import Resize, ToPILImage

# Importa sua classe de modelo
from models import ConditionalGenerator

# =====================================================
# 🚀 Configuração do servidor FastAPI
# =====================================================
app = FastAPI(title="Text-to-Image GAN API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 📦 Estruturas de requisição
# =====================================================
class GenerateRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None

class BatchRequest(BaseModel):
    prompts: List[str]

# =====================================================
# 🧠 Carrega o modelo condicional treinado
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "outputs/training/model_epoch_019.pth"

def load_model():
    model = ConditionalGenerator(latent_dim=100, embedding_dim=50)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Modelo carregado de {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Erro ao carregar modelo: {e}\n→ Usando pesos aleatórios para debug.")
    model.eval()
    return model.to(device)

model = load_model()

# =====================================================
# 🔧 Funções utilitárias
# =====================================================
def tensor_to_pil(tensor: torch.Tensor, size=256) -> Image.Image:
    """
    Converte tensor [-1,1] -> PIL.Image RGB redimensionada.
    """
    tensor = tensor.clamp(-1, 1)  # Garantir limite
    tensor = (tensor + 1) / 2      # [-1,1] -> [0,1]

    # Remove batch se necessário
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Garante 3 canais
    if tensor.shape[0] != 3:
        tensor = tensor.repeat(3, 1, 1)

    # Converte para PIL e redimensiona
    pil_img = ToPILImage()(tensor)
    pil_img = pil_img.resize((size, size), Image.BICUBIC)
    return pil_img

def pil_to_base64(img: Image.Image) -> str:
    """Converte PIL -> Base64 PNG"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def text_to_embedding(prompt: str, dim=50) -> torch.Tensor:
    """Gera embedding fake para debug"""
    torch.manual_seed(abs(hash(prompt)) % (2**32))
    return torch.randn(1, dim, device=device)

# =====================================================
# 🎨 Geração de imagem
# =====================================================
def generate_image(prompt: str, seed: Optional[int] = None) -> Image.Image:
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn(1, 100, device=device)
    text_embedding = text_to_embedding(prompt)

    with torch.no_grad():
        img_tensor = model(noise, text_embedding)[0]  # (3,H,W)
        img_pil = tensor_to_pil(img_tensor, size=256)

    return img_pil

# =====================================================
# 🌐 Endpoints da API
# =====================================================
@app.get("/health")
async def health():
    return {"status": "ok", "message": "Servidor funcionando!"}

@app.get("/model-info")
async def model_info():
    return {
        "model_type": "ConditionalGenerator",
        "embedding_dim": 50,
        "latent_dim": 100,
        "device": str(device),
    }

@app.post("/generate")
async def generate(req: GenerateRequest):
    img = generate_image(req.prompt, req.seed)
    encoded = pil_to_base64(img)
    return {"image": encoded, "prompt": req.prompt}

@app.post("/generate-batch")
async def generate_batch(req: BatchRequest):
    images = []
    for prompt in req.prompts:
        img = generate_image(prompt)
        images.append({
            "prompt": prompt,
            "image": pil_to_base64(img)
        })
    return {"count": len(images), "images": images}

# =====================================================
# ▶️ Execução local
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
