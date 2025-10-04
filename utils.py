import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os
import logging
from datetime import datetime

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """
    Define seed aleatória para reprodutibilidade.
    
    Args:
        seed: valor da seed (padrão: 42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    logger.info(f"Seed definida para: {seed}")


def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch, g_loss, d_loss, filepath):
    """
    Salva checkpoint do modelo contendo todos os estados de treinamento.
    
    Args:
        generator: modelo do gerador
        discriminator: modelo do discriminador
        g_optimizer: otimizador do gerador
        d_optimizer: otimizador do discriminador
        epoch: época atual
        g_loss: perda do gerador
        d_loss: perda do discriminador
        filepath: caminho para salvar o checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_loss': g_loss,
        'd_loss': d_loss,
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint salvo em: {filepath} (Época {epoch})")


def load_checkpoint(filepath, generator, discriminator, g_optimizer=None, d_optimizer=None, device='cpu'):
    """
    Carrega checkpoint do modelo.
    
    Args:
        filepath: caminho do checkpoint
        generator: modelo do gerador
        discriminator: modelo do discriminador
        g_optimizer: otimizador do gerador (opcional)
        d_optimizer: otimizador do discriminador (opcional)
        device: dispositivo (cpu ou cuda)
    
    Returns:
        época do checkpoint carregado
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if g_optimizer is not None:
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    
    if d_optimizer is not None:
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    logger.info(f"Checkpoint carregado de: {filepath} (Época {epoch})")
    
    return epoch


def tensor_to_image(tensor):
    """
    Converte tensor PyTorch para imagem PIL.
    Assume que o tensor está no intervalo [-1, 1].
    
    Args:
        tensor: tensor com shape (1, H, W) ou (H, W)
    
    Returns:
        imagem PIL em escala de cinza
    """
    # Remove batch dimension se existir
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    
    # Desnormaliza de [-1, 1] para [0, 255]
    tensor = (tensor + 1) / 2.0  # [-1, 1] -> [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Converte para numpy e PIL
    img_array = tensor.cpu().numpy()
    img_array = (img_array * 255).astype(np.uint8)
    
    img = Image.fromarray(img_array, mode='L')
    return img


def image_to_base64(image):
    """
    Converte imagem PIL para string base64.
    
    Args:
        image: imagem PIL
    
    Returns:
        string base64 da imagem
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def tensor_to_base64(tensor):
    """
    Converte tensor PyTorch diretamente para base64.
    
    Args:
        tensor: tensor de imagem
    
    Returns:
        string base64 da imagem
    """
    image = tensor_to_image(tensor)
    return image_to_base64(image)


def save_image(tensor, filepath):
    """
    Salva tensor como imagem em arquivo.
    
    Args:
        tensor: tensor de imagem
        filepath: caminho para salvar a imagem
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    image = tensor_to_image(tensor)
    image.save(filepath)
    logger.info(f"Imagem salva em: {filepath}")


def save_image_grid(images, filepath, nrow=8):
    """
    Salva múltiplas imagens em uma grade.
    
    Args:
        images: tensor com batch de imagens (B, 1, H, W)
        filepath: caminho para salvar a grade
        nrow: número de imagens por linha
    """
    from torchvision.utils import make_grid
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Cria grade de imagens
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    
    # Converte para PIL e salva
    image = tensor_to_image(grid)
    image.save(filepath)
    logger.info(f"Grade de imagens salva em: {filepath}")


def generate_noise(batch_size, latent_dim, device='cpu'):
    """
    Gera vetor de ruído aleatório para o gerador.
    
    Args:
        batch_size: tamanho do batch
        latent_dim: dimensão do vetor latente
        device: dispositivo (cpu ou cuda) - pode ser string ou torch.device
    
    Returns:
        tensor de ruído com shape (batch_size, latent_dim)
    """
    if isinstance(device, str):
        device = torch.device(device)
    return torch.randn(batch_size, latent_dim, device=device)


def create_output_directories():
    """
    Cria diretórios necessários para salvar outputs.
    """
    directories = [
        'checkpoints',
        'outputs',
        'outputs/training',
        'outputs/generated'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Diretórios de output criados")


def get_device():
    """
    Retorna o dispositivo disponível (CUDA se disponível, senão CPU).
    
    Returns:
        dispositivo PyTorch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")
    return device


def log_training_stats(epoch, num_epochs, batch_idx, num_batches, d_loss, g_loss):
    """
    Registra estatísticas de treinamento.
    
    Args:
        epoch: época atual
        num_epochs: total de épocas
        batch_idx: índice do batch atual
        num_batches: total de batches
        d_loss: perda do discriminador
        g_loss: perda do gerador
    """
    logger.info(
        f"Época [{epoch}/{num_epochs}] "
        f"Batch [{batch_idx}/{num_batches}] "
        f"D_loss: {d_loss:.4f} "
        f"G_loss: {g_loss:.4f}"
    )


def normalize_images(images):
    """
    Normaliza imagens do intervalo [0, 1] para [-1, 1].
    
    Args:
        images: tensor de imagens no intervalo [0, 1]
    
    Returns:
        imagens normalizadas no intervalo [-1, 1]
    """
    return images * 2.0 - 1.0


def denormalize_images(images):
    """
    Desnormaliza imagens do intervalo [-1, 1] para [0, 1].
    
    Args:
        images: tensor de imagens no intervalo [-1, 1]
    
    Returns:
        imagens desnormalizadas no intervalo [0, 1]
    """
    return (images + 1.0) / 2.0


if __name__ == "__main__":
    # Teste das funções utilitárias
    print("Testando funções utilitárias...")
    
    # Testa seed
    set_seed(42)
    
    # Testa dispositivo
    device = get_device()
    
    # Testa geração de ruído
    noise = generate_noise(4, 100, device)
    print(f"Ruído gerado: shape={noise.shape}, device={noise.device}")
    
    # Testa criação de diretórios
    create_output_directories()
    
    # Testa conversão tensor -> imagem -> base64
    fake_tensor = torch.randn(1, 28, 28)
    fake_tensor = fake_tensor.clamp(-1, 1)
    
    img = tensor_to_image(fake_tensor)
    print(f"Imagem PIL criada: mode={img.mode}, size={img.size}")
    
    base64_str = image_to_base64(img)
    print(f"Base64 string gerada: {len(base64_str)} caracteres")
    
    # Testa salvamento de imagem
    save_image(fake_tensor, "outputs/test_image.png")
    
    print("\nTodas as funções utilitárias funcionando corretamente!")
