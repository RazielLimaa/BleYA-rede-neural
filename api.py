from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import torch
import os
import logging
from models import Generator
from utils import generate_noise, tensor_to_base64, get_device, set_seed

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cria aplicação FastAPI
app = FastAPI(
    title="GAN Image Generator API",
    description="API para geração de imagens usando rede neural GAN treinada",
    version="1.0.0"
)

# Configuração CORS para permitir requests do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo global
generator_model = None
device = None
latent_dim = 100

class GenerateRequest(BaseModel):
    """Schema para request de geração de imagem."""
    prompt: str = ""
    seed: int = None


class GenerateResponse(BaseModel):
    """Schema para response de geração de imagem."""
    image: str
    message: str


def load_generator(checkpoint_path='checkpoints/final_model.pth'):
    """
    Carrega o modelo gerador do checkpoint.
    
    Args:
        checkpoint_path: caminho do checkpoint
    
    Returns:
        modelo gerador carregado
    """
    global generator_model, device
    
    if generator_model is not None:
        return generator_model
    
    device = get_device()
    
    # Verifica se checkpoint existe
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint não encontrado em {checkpoint_path}")
        logger.warning("Criando modelo não treinado para demonstração")
        
        # Cria modelo não treinado
        generator_model = Generator(latent_dim).to(device)
        generator_model.eval()
        return generator_model
    
    # Carrega modelo treinado
    logger.info(f"Carregando modelo de {checkpoint_path}")
    generator_model = Generator(latent_dim).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator_model.load_state_dict(checkpoint['generator_state_dict'])
    generator_model.eval()
    
    logger.info("Modelo carregado com sucesso")
    return generator_model


@app.on_event("startup")
async def startup_event():
    """Evento executado no startup da API."""
    logger.info("Iniciando API de geração de imagens...")
    
    # Tenta carregar modelo
    load_generator()
    
    logger.info("API iniciada com sucesso!")


@app.get("/")
async def root():
    """Endpoint raiz da API."""
    return {
        "message": "GAN Image Generator API",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "POST - Gera uma nova imagem",
            "/health": "GET - Verifica saúde da API",
            "/model-info": "GET - Informações do modelo"
        }
    }


@app.get("/health")
async def health_check():
    """Verifica se a API está funcionando."""
    model_loaded = generator_model is not None
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(device) if device else "not initialized"
    }


@app.get("/model-info")
async def model_info():
    """Retorna informações sobre o modelo."""
    if generator_model is None:
        load_generator()
    
    num_params = sum(p.numel() for p in generator_model.parameters())
    
    return {
        "model_type": "GAN Generator",
        "latent_dim": latent_dim,
        "output_size": "28x28",
        "num_parameters": num_params,
        "device": str(device)
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest = None):
    """
    Gera uma nova imagem usando o modelo GAN.
    
    Args:
        request: request opcional com prompt e seed
    
    Returns:
        imagem gerada em formato base64
    """
    try:
        # Carrega modelo se não estiver carregado
        if generator_model is None:
            load_generator()
        
        # Define seed se fornecida
        if request and request.seed is not None:
            set_seed(request.seed)
            logger.info(f"Usando seed: {request.seed}")
        
        # Gera ruído aleatório
        noise = generate_noise(1, latent_dim, device)
        
        # Gera imagem
        with torch.no_grad():
            fake_image = generator_model(noise)
        
        # Converte para base64
        image_base64 = tensor_to_base64(fake_image)
        
        logger.info("Imagem gerada com sucesso")
        
        return GenerateResponse(
            image=image_base64,
            message="Imagem gerada com sucesso"
        )
    
    except Exception as e:
        logger.error(f"Erro ao gerar imagem: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar imagem: {str(e)}"
        )


@app.post("/generate-batch")
async def generate_batch(num_images: int = 4):
    """
    Gera múltiplas imagens de uma vez.
    
    Args:
        num_images: número de imagens para gerar (máximo 16)
    
    Returns:
        lista de imagens em base64
    """
    try:
        if num_images > 16:
            raise HTTPException(
                status_code=400,
                detail="Número máximo de imagens é 16"
            )
        
        # Carrega modelo se não estiver carregado
        if generator_model is None:
            load_generator()
        
        # Gera múltiplas imagens
        noise = generate_noise(num_images, latent_dim, device)
        
        with torch.no_grad():
            fake_images = generator_model(noise)
        
        # Converte cada imagem para base64
        images_base64 = []
        for i in range(num_images):
            image_base64 = tensor_to_base64(fake_images[i])
            images_base64.append(image_base64)
        
        logger.info(f"{num_images} imagens geradas com sucesso")
        
        return {
            "images": images_base64,
            "count": num_images,
            "message": f"{num_images} imagens geradas com sucesso"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao gerar batch de imagens: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar imagens: {str(e)}"
        )


# Serve arquivos estáticos do frontend
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
    
    @app.get("/app")
    async def serve_app():
        """Serve o frontend HTML."""
        return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import uvicorn
    
    # Roda servidor
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
