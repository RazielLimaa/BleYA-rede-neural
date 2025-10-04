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
    Suporta tanto imagens em escala de cinza quanto RGB.
    
    Args:
        tensor: tensor com shape (1, H, W) para grayscale, (3, H, W) para RGB,
                ou com batch dimension (B, C, H, W)
    
    Returns:
        imagem PIL em modo 'L' para grayscale ou 'RGB' para imagens coloridas
    """
    # Remove batch dimension se existir
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Desnormaliza de [-1, 1] para [0, 255]
    tensor = (tensor + 1) / 2.0  # [-1, 1] -> [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Converte para numpy
    img_array = tensor.cpu().numpy()
    img_array = (img_array * 255).astype(np.uint8)
    
    # Determina o modo baseado no número de canais
    if tensor.dim() == 3:
        channels = tensor.shape[0]
        if channels == 1:
            # Grayscale: remove dimensão do canal
            img_array = img_array.squeeze(0)
            img = Image.fromarray(img_array, mode='L')
        elif channels == 3:
            # RGB: transpõe de (C, H, W) para (H, W, C)
            img_array = np.transpose(img_array, (1, 2, 0))
            img = Image.fromarray(img_array, mode='RGB')
        else:
            raise ValueError(f"Número de canais não suportado: {channels}. Use 1 para grayscale ou 3 para RGB.")
    elif tensor.dim() == 2:
        # Já é 2D, assume grayscale
        img = Image.fromarray(img_array, mode='L')
    else:
        raise ValueError(f"Dimensões de tensor não suportadas: {tensor.shape}")
    
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
    Suporta imagens em escala de cinza e RGB.
    
    Args:
        images: tensor com batch de imagens (B, C, H, W) onde C=1 para grayscale ou C=3 para RGB
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
    Funciona para imagens grayscale (1 canal) e RGB (3 canais).
    
    Args:
        images: tensor de imagens no intervalo [0, 1] com shape (B, C, H, W)
    
    Returns:
        imagens normalizadas no intervalo [-1, 1]
    """
    return images * 2.0 - 1.0


def denormalize_images(images):
    """
    Desnormaliza imagens do intervalo [-1, 1] para [0, 1].
    Funciona para imagens grayscale (1 canal) e RGB (3 canais).
    
    Args:
        images: tensor de imagens no intervalo [-1, 1] com shape (B, C, H, W)
    
    Returns:
        imagens desnormalizadas no intervalo [0, 1]
    """
    return (images + 1.0) / 2.0


def get_cifar10_classes():
    """
    Retorna lista com os nomes das 10 classes do CIFAR-10.
    
    Returns:
        lista com os nomes das classes do CIFAR-10
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def get_cifar10_dataloader(batch_size=64, train=True, data_dir='./data', num_workers=0):
    """
    Carrega o dataset CIFAR-10 e retorna um DataLoader configurado.
    Dataset é baixado e armazenado em cache no diretório especificado.
    As imagens são normalizadas para o intervalo [-1, 1].
    
    Args:
        batch_size: tamanho do batch (padrão: 64)
        train: se True carrega conjunto de treino, senão conjunto de teste (padrão: True)
        data_dir: diretório para cache do dataset (padrão: './data')
        num_workers: número de workers para carregamento paralelo (padrão: 0 para CPU-only)
    
    Returns:
        DataLoader com o dataset CIFAR-10
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # Cria diretório de dados se não existir
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transformações: ToTensor converte para [0, 1], depois normalizamos para [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normaliza para [-1, 1]
    ])
    
    try:
        # Carrega dataset CIFAR-10
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )
        
        # Cria DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,  # Embaralha apenas no treino
            num_workers=num_workers,
            pin_memory=False  # CPU-only
        )
        
        dataset_type = "treino" if train else "teste"
        logger.info(f"CIFAR-10 dataset carregado: {len(dataset)} imagens ({dataset_type})")
        logger.info(f"DataLoader criado: batch_size={batch_size}, num_batches={len(dataloader)}")
        
        return dataloader
        
    except Exception as e:
        logger.error(f"Erro ao carregar CIFAR-10: {e}")
        raise


if __name__ == "__main__":
    # Teste das funções utilitárias
    print("="*60)
    print("Testando funções utilitárias...")
    print("="*60)
    
    # Testa seed
    set_seed(42)
    
    # Testa dispositivo
    device = get_device()
    
    # Testa geração de ruído
    noise = generate_noise(4, 100, device)
    print(f"\n✓ Ruído gerado: shape={noise.shape}, device={noise.device}")
    
    # Testa criação de diretórios
    create_output_directories()
    
    # Testa classes do CIFAR-10
    print("\n--- Testando CIFAR-10 ---")
    cifar_classes = get_cifar10_classes()
    print(f"✓ Classes CIFAR-10: {cifar_classes}")
    print(f"✓ Total de classes: {len(cifar_classes)}")
    
    # Testa conversão de imagens GRAYSCALE
    print("\n--- Testando imagens GRAYSCALE ---")
    fake_gray_tensor = torch.randn(1, 28, 28)
    fake_gray_tensor = fake_gray_tensor.clamp(-1, 1)
    
    img_gray = tensor_to_image(fake_gray_tensor)
    print(f"✓ Imagem PIL grayscale criada: mode={img_gray.mode}, size={img_gray.size}")
    
    base64_str_gray = image_to_base64(img_gray)
    print(f"✓ Base64 grayscale gerada: {len(base64_str_gray)} caracteres")
    
    # Testa salvamento de imagem grayscale
    save_image(fake_gray_tensor, "outputs/test_image_grayscale.png")
    
    # Testa normalização/desnormalização grayscale
    normalized_gray = normalize_images(torch.rand(1, 1, 28, 28))
    denormalized_gray = denormalize_images(normalized_gray)
    print(f"✓ Normalização grayscale: min={normalized_gray.min():.2f}, max={normalized_gray.max():.2f}")
    print(f"✓ Desnormalização grayscale: min={denormalized_gray.min():.2f}, max={denormalized_gray.max():.2f}")
    
    # Testa conversão de imagens RGB
    print("\n--- Testando imagens RGB ---")
    fake_rgb_tensor = torch.randn(3, 32, 32)
    fake_rgb_tensor = fake_rgb_tensor.clamp(-1, 1)
    
    img_rgb = tensor_to_image(fake_rgb_tensor)
    print(f"✓ Imagem PIL RGB criada: mode={img_rgb.mode}, size={img_rgb.size}")
    
    base64_str_rgb = image_to_base64(img_rgb)
    print(f"✓ Base64 RGB gerada: {len(base64_str_rgb)} caracteres")
    
    # Testa salvamento de imagem RGB
    save_image(fake_rgb_tensor, "outputs/test_image_rgb.png")
    
    # Testa normalização/desnormalização RGB
    normalized_rgb = normalize_images(torch.rand(1, 3, 32, 32))
    denormalized_rgb = denormalize_images(normalized_rgb)
    print(f"✓ Normalização RGB: min={normalized_rgb.min():.2f}, max={normalized_rgb.max():.2f}")
    print(f"✓ Desnormalização RGB: min={denormalized_rgb.min():.2f}, max={denormalized_rgb.max():.2f}")
    
    # Testa conversão com batch dimension
    print("\n--- Testando com batch dimension ---")
    fake_batch_gray = torch.randn(1, 1, 28, 28).clamp(-1, 1)
    fake_batch_rgb = torch.randn(1, 3, 32, 32).clamp(-1, 1)
    
    img_batch_gray = tensor_to_image(fake_batch_gray)
    img_batch_rgb = tensor_to_image(fake_batch_rgb)
    print(f"✓ Batch grayscale: mode={img_batch_gray.mode}, size={img_batch_gray.size}")
    print(f"✓ Batch RGB: mode={img_batch_rgb.mode}, size={img_batch_rgb.size}")
    
    # Testa grade de imagens grayscale
    print("\n--- Testando grade de imagens ---")
    batch_gray_images = torch.randn(8, 1, 28, 28).clamp(-1, 1)
    save_image_grid(batch_gray_images, "outputs/test_grid_grayscale.png", nrow=4)
    
    # Testa grade de imagens RGB
    batch_rgb_images = torch.randn(8, 3, 32, 32).clamp(-1, 1)
    save_image_grid(batch_rgb_images, "outputs/test_grid_rgb.png", nrow=4)
    
    # Teste do DataLoader CIFAR-10 (comentado por padrão para não baixar o dataset durante testes rápidos)
    print("\n--- Testando CIFAR-10 DataLoader (descomente para testar download) ---")
    print("# Para testar o download do CIFAR-10, descomente as linhas abaixo:")
    print("# dataloader = get_cifar10_dataloader(batch_size=4, train=True)")
    print("# images, labels = next(iter(dataloader))")
    print("# print(f'✓ Batch CIFAR-10: images.shape={images.shape}, labels.shape={labels.shape}')")
    
    print("\n" + "="*60)
    print("✓ Todas as funções utilitárias funcionando corretamente!")
    print("✓ Suporte completo para grayscale (1 canal) e RGB (3 canais)")
    print("="*60)
