from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
import os
import logging
from models import Generator, ConditionalGenerator
from text_encoder import TextEncoder
from utils import generate_noise, tensor_to_base64, get_device, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GAN Image Generator API",
    description="API para geração de imagens usando rede neural GAN treinada",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator_model = None
text_encoder = None
device = None
latent_dim = 100
text_embed_dim = 50
conditional_mode = False

class GenerateRequest(BaseModel):
    """Schema para request de geração de imagem."""
    prompt: Optional[str] = None
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    """Schema para response de geração de imagem."""
    image: str
    message: str
    prompt: Optional[str] = None
    mode: str


class BatchGenerateRequest(BaseModel):
    """Schema para request de geração em batch."""
    prompts: Optional[List[str]] = None
    num_images: Optional[int] = 4


def load_generator():
    """
    Carrega o modelo gerador do checkpoint.
    Tenta carregar modelo condicional primeiro, depois fallback para modelo original.
    
    Returns:
        modelo gerador carregado
    """
    global generator_model, text_encoder, device, conditional_mode
    
    if generator_model is not None:
        return generator_model
    
    device = get_device()
    
    conditional_checkpoint = 'checkpoints/conditional_final_model.pth'
    ema_checkpoint = 'checkpoints/ema_generator.pth'
    original_checkpoint = 'checkpoints/final_model.pth'
    
    if os.path.exists(conditional_checkpoint):
        logger.info(f"Carregando modelo condicional de {conditional_checkpoint}")
        try:
            generator_model = ConditionalGenerator(
                latent_dim=latent_dim, 
                embedding_dim=text_embed_dim
            ).to(device)
            
            checkpoint = torch.load(conditional_checkpoint, map_location=device)
            generator_model.load_state_dict(checkpoint['generator_state_dict'])
            generator_model.eval()
            
            logger.info("Inicializando TextEncoder...")
            text_encoder = TextEncoder(embedding_dim=text_embed_dim)
            
            conditional_mode = True
            logger.info("✓ Modo CONDICIONAL ativado (Text-to-Image)")
            logger.info("  - Modelo: ConditionalGenerator")
            logger.info("  - Saída: RGB 32x32")
            logger.info("  - TextEncoder: Carregado")
            
            return generator_model
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo condicional: {str(e)}")
            logger.info("Tentando carregar modelo original...")
    
    elif os.path.exists(ema_checkpoint):
        logger.info(f"Carregando modelo EMA de {ema_checkpoint}")
        try:
            generator_model = ConditionalGenerator(
                latent_dim=latent_dim, 
                embedding_dim=text_embed_dim
            ).to(device)
            
            checkpoint = torch.load(ema_checkpoint, map_location=device)
            generator_model.load_state_dict(checkpoint)
            generator_model.eval()
            
            logger.info("Inicializando TextEncoder...")
            text_encoder = TextEncoder(embedding_dim=text_embed_dim)
            
            conditional_mode = True
            logger.info("✓ Modo CONDICIONAL ativado (Text-to-Image) com EMA")
            logger.info("  - Modelo: ConditionalGenerator (EMA)")
            logger.info("  - Saída: RGB 32x32")
            logger.info("  - TextEncoder: Carregado")
            
            return generator_model
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo EMA: {str(e)}")
            logger.info("Tentando carregar modelo original...")
    
    if os.path.exists(original_checkpoint):
        logger.info(f"Carregando modelo original de {original_checkpoint}")
        generator_model = Generator(latent_dim).to(device)
        
        checkpoint = torch.load(original_checkpoint, map_location=device)
        generator_model.load_state_dict(checkpoint['generator_state_dict'])
        generator_model.eval()
        
        conditional_mode = False
        logger.info("✓ Modo ORIGINAL ativado (MNIST)")
        logger.info("  - Modelo: Generator")
        logger.info("  - Saída: Grayscale 28x28")
        
        return generator_model
    
    logger.warning("Nenhum checkpoint encontrado")
    logger.warning("Criando modelo não treinado para demonstração")
    
    generator_model = Generator(latent_dim).to(device)
    generator_model.eval()
    conditional_mode = False
    logger.info("✓ Modo ORIGINAL ativado (modelo não treinado)")
    
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
        "version": "2.0.0",
        "mode": "conditional" if conditional_mode else "original",
        "endpoints": {
            "/generate": "POST - Gera uma nova imagem",
            "/generate-batch": "POST - Gera múltiplas imagens",
            "/health": "GET - Verifica saúde da API",
            "/model-info": "GET - Informações do modelo",
            "/sample-prompts": "GET - Exemplos de prompts (modo condicional)"
        }
    }


@app.get("/health")
async def health_check():
    """Verifica se a API está funcionando."""
    model_loaded = generator_model is not None
    text_encoder_loaded = text_encoder is not None
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "text_encoder_loaded": text_encoder_loaded,
        "conditional_mode": conditional_mode,
        "device": str(device) if device else "not initialized"
    }


@app.get("/model-info")
async def model_info():
    """Retorna informações sobre o modelo."""
    if generator_model is None:
        load_generator()
    
    num_params = sum(p.numel() for p in generator_model.parameters())
    
    info = {
        "model_type": "Conditional GAN Generator" if conditional_mode else "GAN Generator",
        "latent_dim": latent_dim,
        "num_parameters": num_params,
        "device": str(device),
        "conditional_mode": conditional_mode,
        "text_encoder_loaded": text_encoder is not None
    }
    
    if conditional_mode:
        info["image_size"] = "32x32"
        info["image_channels"] = 3
        info["text_embed_dim"] = text_embed_dim
    else:
        info["image_size"] = "28x28"
        info["image_channels"] = 1
    
    return info


@app.get("/sample-prompts")
async def sample_prompts():
    """
    Retorna exemplos de prompts para geração de imagens.
    Útil no modo condicional para ajudar usuários a entender quais prompts funcionam.
    """
    if not conditional_mode:
        return {
            "message": "Prompts não são necessários no modo original (MNIST)",
            "mode": "original"
        }
    
    cifar10_prompts = [
        "a flying airplane in the sky",
        "a red car on the road",
        "a colorful bird with feathers",
        "a cute cat with whiskers",
        "a graceful deer in the forest",
        "a friendly dog with fur",
        "a green frog near water",
        "a strong horse running",
        "a sailing ship on the ocean",
        "a large truck on the highway"
    ]
    
    return {
        "prompts": cifar10_prompts,
        "mode": "conditional",
        "message": "Exemplos de prompts baseados nas classes CIFAR-10"
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest = GenerateRequest()):
    """
    Gera uma nova imagem usando o modelo GAN.
    
    Args:
        request: request com prompt (condicional) e seed opcional
    
    Returns:
        imagem gerada em formato base64
    """
    try:
        if generator_model is None:
            load_generator()
        
        if request.seed is not None:
            set_seed(request.seed)
            logger.info(f"Usando seed: {request.seed}")
        
        noise = generate_noise(1, latent_dim, device)
        
        if conditional_mode:
            if not request.prompt or not request.prompt.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Prompt é obrigatório no modo condicional. Use /sample-prompts para ver exemplos."
                )
            
            if text_encoder is None:
                raise HTTPException(
                    status_code=500,
                    detail="TextEncoder não está carregado. Reinicie a API."
                )
            
            prompt = request.prompt.strip()
            logger.info(f"Gerando imagem para prompt: '{prompt}'")
            
            text_embedding = text_encoder.embed_text(prompt, return_tensor=True)
            text_embedding = text_embedding.unsqueeze(0).to(device)
            
            with torch.no_grad():
                fake_image = generator_model(noise, text_embedding)
            
            image_base64 = tensor_to_base64(fake_image)
            
            logger.info(f"Imagem condicional gerada com sucesso para: '{prompt}'")
            
            return GenerateResponse(
                image=image_base64,
                message="Imagem condicional gerada com sucesso",
                prompt=prompt,
                mode="conditional"
            )
        
        else:
            with torch.no_grad():
                fake_image = generator_model(noise)
            
            image_base64 = tensor_to_base64(fake_image)
            
            logger.info("Imagem gerada com sucesso (modo original)")
            
            return GenerateResponse(
                image=image_base64,
                message="Imagem gerada com sucesso",
                prompt=None,
                mode="original"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao gerar imagem: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar imagem: {str(e)}"
        )


@app.post("/generate-batch")
async def generate_batch(request: BatchGenerateRequest = BatchGenerateRequest()):
    """
    Gera múltiplas imagens de uma vez.
    
    Args:
        request: contém lista de prompts ou número de imagens
    
    Returns:
        lista de imagens em base64
    """
    try:
        if generator_model is None:
            load_generator()
        
        if conditional_mode:
            if not request.prompts or len(request.prompts) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Lista de prompts é obrigatória no modo condicional. Use /sample-prompts para ver exemplos."
                )
            
            if text_encoder is None:
                raise HTTPException(
                    status_code=500,
                    detail="TextEncoder não está carregado. Reinicie a API."
                )
            
            num_images = len(request.prompts)
            if num_images > 16:
                raise HTTPException(
                    status_code=400,
                    detail="Número máximo de imagens é 16"
                )
            
            prompts = request.prompts
            logger.info(f"Gerando {num_images} imagens condicionais")
            
            text_embeddings = text_encoder.embed_batch(prompts, return_tensor=True)
            text_embeddings = text_embeddings.to(device)
            
            noise = generate_noise(num_images, latent_dim, device)
            
            with torch.no_grad():
                fake_images = generator_model(noise, text_embeddings)
            
            images_base64 = []
            for i in range(num_images):
                image_base64 = tensor_to_base64(fake_images[i])
                images_base64.append({
                    "image": image_base64,
                    "prompt": prompts[i]
                })
            
            logger.info(f"{num_images} imagens condicionais geradas com sucesso")
            
            return {
                "images": images_base64,
                "count": num_images,
                "mode": "conditional",
                "message": f"{num_images} imagens condicionais geradas com sucesso"
            }
        
        else:
            num_images = request.num_images or 4
            
            if num_images > 16:
                raise HTTPException(
                    status_code=400,
                    detail="Número máximo de imagens é 16"
                )
            
            noise = generate_noise(num_images, latent_dim, device)
            
            with torch.no_grad():
                fake_images = generator_model(noise)
            
            images_base64 = []
            for i in range(num_images):
                image_base64 = tensor_to_base64(fake_images[i])
                images_base64.append({
                    "image": image_base64,
                    "prompt": None
                })
            
            logger.info(f"{num_images} imagens geradas com sucesso (modo original)")
            
            return {
                "images": images_base64,
                "count": num_images,
                "mode": "original",
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
        host="127.0.0.1",
        port=5000,
        log_level="info"
    )
