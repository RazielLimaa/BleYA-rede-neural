import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from models import Generator, Discriminator, initialize_weights
from utils import (
    set_seed, save_checkpoint, generate_noise, save_image_grid,
    create_output_directories, get_device, log_training_stats, normalize_images
)

def train_gan(
    num_epochs=20,
    batch_size=128,
    latent_dim=100,
    lr=0.0002,
    beta1=0.5,
    sample_interval=100,
    checkpoint_interval=5,
    seed=42
):
    """
    Pipeline completo de treinamento da GAN.
    
    Args:
        num_epochs: número de épocas de treinamento
        batch_size: tamanho do batch
        latent_dim: dimensão do vetor de ruído latente
        lr: taxa de aprendizado
        beta1: parâmetro beta1 do otimizador Adam
        sample_interval: intervalo de batches para gerar amostras
        checkpoint_interval: intervalo de épocas para salvar checkpoints
        seed: seed para reprodutibilidade
    """
    
    # Configurações iniciais
    set_seed(seed)
    device = get_device()
    create_output_directories()
    
    # Carrega dataset MNIST
    print("Carregando dataset MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Dataset carregado: {len(train_dataset)} imagens")
    
    # Cria modelos
    print("\nCriando modelos...")
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Inicializa pesos
    initialize_weights(generator)
    initialize_weights(discriminator)
    
    # Otimizadores
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Função de perda
    criterion = nn.BCELoss()
    
    # Labels para treinamento
    real_label = 1.0
    fake_label = 0.0
    
    print(f"Modelos criados e movidos para {device}")
    print(f"Iniciando treinamento por {num_epochs} épocas...\n")
    
    # Loop de treinamento
    avg_d_loss = 0.0
    avg_g_loss = 0.0
    
    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        
        for batch_idx, (real_images, _) in enumerate(dataloader, 1):
            batch_size_current = real_images.size(0)
            real_images = real_images.to(device)
            
            # Normaliza imagens de [0, 1] para [-1, 1]
            real_images = normalize_images(real_images)
            
            # =====================
            # Treina Discriminador
            # =====================
            d_optimizer.zero_grad()
            
            # Treina com imagens reais
            real_labels = torch.full((batch_size_current, 1), real_label, device=device)
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            # Treina com imagens falsas
            noise = generate_noise(batch_size_current, latent_dim, device)
            fake_images = generator(noise)
            fake_labels = torch.full((batch_size_current, 1), fake_label, device=device)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            # Perda total do discriminador
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ================
            # Treina Gerador
            # ================
            g_optimizer.zero_grad()
            
            # Gera novas imagens falsas
            noise = generate_noise(batch_size_current, latent_dim, device)
            fake_images = generator(noise)
            
            # Tenta enganar o discriminador
            real_labels = torch.full((batch_size_current, 1), real_label, device=device)
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)
            
            g_loss.backward()
            g_optimizer.step()
            
            # Acumula perdas da época
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            
            # Log e salvamento de amostras
            if batch_idx % sample_interval == 0:
                log_training_stats(
                    epoch, num_epochs,
                    batch_idx, len(dataloader),
                    d_loss.item(), g_loss.item()
                )
                
                # Salva grade de imagens geradas
                with torch.no_grad():
                    generator.eval()
                    sample_noise = generate_noise(64, latent_dim, device)
                    sample_images = generator(sample_noise)
                    save_image_grid(
                        sample_images,
                        f'outputs/training/epoch_{epoch:03d}_batch_{batch_idx:04d}.png',
                        nrow=8
                    )
                    generator.train()
        
        # Estatísticas da época
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        
        print(f"\n{'='*70}")
        print(f"Época {epoch}/{num_epochs} concluída")
        print(f"Perda média do Discriminador: {avg_d_loss:.4f}")
        print(f"Perda média do Gerador: {avg_g_loss:.4f}")
        print(f"{'='*70}\n")
        
        # Salva checkpoint
        if epoch % checkpoint_interval == 0:
            checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch:03d}.pth'
            save_checkpoint(
                generator, discriminator,
                g_optimizer, d_optimizer,
                epoch, avg_g_loss, avg_d_loss,
                checkpoint_path
            )
        
        # Salva amostras finais da época
        with torch.no_grad():
            generator.eval()
            sample_noise = generate_noise(64, latent_dim, device)
            sample_images = generator(sample_noise)
            save_image_grid(
                sample_images,
                f'outputs/training/epoch_{epoch:03d}_final.png',
                nrow=8
            )
    
    # Salva modelo final
    final_checkpoint = 'checkpoints/final_model.pth'
    save_checkpoint(
        generator, discriminator,
        g_optimizer, d_optimizer,
        num_epochs, avg_g_loss, avg_d_loss,
        final_checkpoint
    )
    
    print("\n" + "="*70)
    print("Treinamento concluído!")
    print(f"Modelo final salvo em: {final_checkpoint}")
    print(f"Imagens de treinamento salvas em: outputs/training/")
    print("="*70)


def generate_samples(num_samples=100, checkpoint_path='checkpoints/final_model.pth', output_dir='outputs/generated'):
    """
    Gera amostras de imagens usando modelo treinado.
    
    Args:
        num_samples: número de imagens para gerar
        checkpoint_path: caminho do checkpoint do modelo
        output_dir: diretório para salvar as imagens geradas
    """
    set_seed()
    device = get_device()
    
    # Cria gerador
    latent_dim = 100
    generator = Generator(latent_dim).to(device)
    
    # Carrega modelo treinado
    if not os.path.exists(checkpoint_path):
        print(f"Erro: Checkpoint não encontrado em {checkpoint_path}")
        print("Execute o treinamento primeiro!")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print(f"Modelo carregado de: {checkpoint_path}")
    print(f"Gerando {num_samples} imagens...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Gera imagens individuais
    with torch.no_grad():
        for i in range(num_samples):
            noise = generate_noise(1, latent_dim, device)
            fake_image = generator(noise)
            
            from utils import save_image
            save_image(fake_image, f'{output_dir}/generated_{i:04d}.png')
        
        # Gera grade de amostras
        sample_noise = generate_noise(64, latent_dim, device)
        sample_images = generator(sample_noise)
        save_image_grid(
            sample_images,
            f'{output_dir}/sample_grid.png',
            nrow=8
        )
    
    print(f"\n{num_samples} imagens geradas e salvas em: {output_dir}/")
    print(f"Grade de amostras salva em: {output_dir}/sample_grid.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Treinamento de GAN para geração de imagens')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'],
                        help='Modo: train (treinar modelo) ou generate (gerar amostras)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Número de épocas de treinamento')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Tamanho do batch')
    parser.add_argument('--samples', type=int, default=100,
                        help='Número de amostras para gerar')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pth',
                        help='Caminho do checkpoint')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Modo: Treinamento")
        train_gan(
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        print("Modo: Geração de amostras")
        generate_samples(
            num_samples=args.samples,
            checkpoint_path=args.checkpoint
        )
