import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import copy
from models import Generator, Discriminator, ConditionalGenerator, ConditionalDiscriminator, initialize_weights
from text_encoder import TextEncoder
from utils import (
    set_seed, save_checkpoint, generate_noise, save_image_grid,
    create_output_directories, get_device, log_training_stats, normalize_images,
    get_cifar10_dataloader
)


class EMA:
    """
    Exponential Moving Average for model weights.
    Maintains a shadow copy of model parameters with exponential decay.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def compute_gradient_penalty(discriminator, real_images, text_embeddings, device):
    """
    Compute R1 gradient penalty for discriminator.
    R1 penalty: ||∇D(x)||^2 where x is real data
    """
    real_images_copy = real_images.detach().clone()
    real_images_copy.requires_grad_(True)
    
    real_pred = discriminator(real_images_copy, text_embeddings)
    
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_images_copy,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
    
    return grad_penalty


def extract_discriminator_features(discriminator, images, text_embeddings):
    """
    Extract intermediate features from discriminator for feature matching.
    Returns features before the final classification layer.
    """
    h = discriminator.conv_first(images)
    h = discriminator.res_block1(h)
    h = discriminator.res_block2(h)
    h = discriminator.attention(h)
    h = discriminator.res_block3(h)
    h = discriminator.res_block4(h)
    h = discriminator.lrelu(h)
    h = discriminator.global_pool(h)
    features = h.view(h.size(0), -1)
    return features


def save_conditional_checkpoint(generator, discriminator, text_encoder, g_optimizer, d_optimizer, 
                                ema, epoch, g_loss, d_loss, d_loss_real, d_loss_fake, 
                                gp_loss, fm_loss, filepath):
    """
    Save conditional GAN checkpoint including text encoder and EMA.
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
        'd_loss_real': d_loss_real,
        'd_loss_fake': d_loss_fake,
        'gp_loss': gp_loss,
        'fm_loss': fm_loss,
        'ema_state_dict': ema.state_dict() if ema else None,
    }
    
    torch.save(checkpoint, filepath)
    print(f"Conditional checkpoint saved: {filepath} (Epoch {epoch})")


def train_conditional_gan(
    num_epochs=100,
    batch_size=32,
    latent_dim=100,
    text_embed_dim=50,
    lr_g=0.0001,
    lr_d=0.0004,
    beta1=0.0,
    beta2=0.999,
    gradient_penalty_weight=0.0,
    feature_matching_weight=10.0,
    ema_decay=0.999,
    sample_interval=100,
    checkpoint_interval=10,
    seed=42,
    max_grad_norm=1.0
):
    """
    Train conditional GAN for text-to-image generation on CIFAR-10.
    
    Implements advanced stability techniques:
    - Feature matching loss
    - R1 gradient penalty (disabled by default due to inplace operations; discriminator uses spectral normalization instead)
    - Label smoothing
    - EMA (Exponential Moving Average)
    - TTUR (Two-timescale update rule)
    - Gradient clipping
    - NaN detection
    
    Note: Gradient penalty is set to 0.0 by default because the discriminator uses inplace LeakyReLU operations
    which are incompatible with gradient penalty computation. The discriminator already uses spectral normalization
    for stability, which is often sufficient.
    """
    
    try:
        set_seed(seed)
        device = get_device()
        create_output_directories()
        
        print("\n" + "="*70)
        print("CONDITIONAL TEXT-TO-IMAGE GAN TRAINING (CIFAR-10)")
        print("="*70)
        
        print("\nLoading CIFAR-10 dataset...")
        dataloader = get_cifar10_dataloader(
            batch_size=batch_size,
            train=True,
            data_dir='./data',
            num_workers=0
        )
        print(f"Dataset loaded: {len(dataloader.dataset)} images")
        
        print("\nInitializing text encoder...")
        text_encoder = TextEncoder(embedding_dim=text_embed_dim)
        print(f"Text encoder ready (embedding_dim={text_embed_dim})")
        
        print("\nCreating conditional models...")
        generator = ConditionalGenerator(
            latent_dim=latent_dim,
            embedding_dim=text_embed_dim
        ).to(device)
        discriminator = ConditionalDiscriminator(
            embedding_dim=text_embed_dim
        ).to(device)
        
        initialize_weights(generator)
        initialize_weights(discriminator)
        
        print(f"Models created and moved to {device}")
        
        print("\nInitializing EMA for generator...")
        ema = EMA(generator, decay=ema_decay)
        print(f"EMA initialized (decay={ema_decay})")
        
        print("\nSetting up optimizers with TTUR...")
        g_optimizer = optim.Adam(
            generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2)
        )
        d_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )
        print(f"Generator LR: {lr_g}, Discriminator LR: {lr_d}")
        
        criterion = nn.BCELoss()
        
        real_label_smooth = 0.9
        fake_label = 0.0
        
        print(f"\nTraining configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Text embed dim: {text_embed_dim}")
        print(f"  Label smoothing: {real_label_smooth}")
        print(f"  GP weight: {gradient_penalty_weight}")
        print(f"  FM weight: {feature_matching_weight}")
        print(f"  Max grad norm: {max_grad_norm}")
        print(f"\nStarting training...\n")
        
        avg_d_loss = 0.0
        avg_g_loss = 0.0
        avg_d_loss_real = 0.0
        avg_d_loss_fake = 0.0
        avg_gp_loss = 0.0
        avg_fm_loss = 0.0
        
        for epoch in range(1, num_epochs + 1):
            generator.train()
            discriminator.train()
            
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_d_loss_real = 0.0
            epoch_d_loss_fake = 0.0
            epoch_gp_loss = 0.0
            epoch_fm_loss = 0.0
            
            for batch_idx, (real_images, class_labels) in enumerate(dataloader, 1):
                batch_size_current = real_images.size(0)
                real_images = real_images.to(device)
                
                captions = [text_encoder.get_class_caption(label.item()) for label in class_labels]
                text_embeddings = text_encoder.embed_batch(captions, return_tensor=True).to(device)
                
                real_labels = torch.full((batch_size_current, 1), real_label_smooth, device=device)
                fake_labels = torch.full((batch_size_current, 1), fake_label, device=device)
                
                d_optimizer.zero_grad()
                real_output = discriminator(real_images, text_embeddings)
                d_loss_real = criterion(real_output, real_labels)
                d_loss_real.backward()
                
                noise = generate_noise(batch_size_current, latent_dim, device)
                fake_images = generator(noise, text_embeddings).detach()
                fake_output = discriminator(fake_images, text_embeddings)
                d_loss_fake = criterion(fake_output, fake_labels)
                d_loss_fake.backward()
                
                if gradient_penalty_weight > 0:
                    gp = compute_gradient_penalty(discriminator, real_images, text_embeddings, device)
                    gp_loss = gradient_penalty_weight * gp
                    gp_loss.backward()
                else:
                    gp_loss = torch.tensor(0.0)
                
                d_loss = d_loss_real + d_loss_fake + gp_loss
                
                if torch.isnan(d_loss):
                    print(f"\nWARNING: NaN detected in discriminator loss at epoch {epoch}, batch {batch_idx}")
                    d_optimizer.zero_grad()
                    continue
                
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
                d_optimizer.step()
                
                g_optimizer.zero_grad()
                
                noise = generate_noise(batch_size_current, latent_dim, device)
                fake_images = generator(noise, text_embeddings)
                
                real_labels = torch.full((batch_size_current, 1), 1.0, device=device)
                fake_output = discriminator(fake_images, text_embeddings)
                g_loss_adv = criterion(fake_output, real_labels)
                
                with torch.no_grad():
                    real_features = extract_discriminator_features(discriminator, real_images, text_embeddings)
                fake_features = extract_discriminator_features(discriminator, fake_images, text_embeddings)
                fm_loss = feature_matching_weight * torch.mean((real_features - fake_features) ** 2)
                
                g_loss = g_loss_adv + fm_loss
                
                if torch.isnan(g_loss):
                    print(f"\nWARNING: NaN detected in generator loss at epoch {epoch}, batch {batch_idx}")
                    continue
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
                g_optimizer.step()
                
                ema.update()
                
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                epoch_d_loss_real += d_loss_real.item()
                epoch_d_loss_fake += d_loss_fake.item()
                epoch_gp_loss += gp_loss.item()
                epoch_fm_loss += fm_loss.item()
                
                if batch_idx % sample_interval == 0:
                    print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}]")
                    print(f"  D_loss: {d_loss.item():.4f} (real: {d_loss_real.item():.4f}, fake: {d_loss_fake.item():.4f})")
                    print(f"  G_loss: {g_loss.item():.4f} (adv: {g_loss_adv.item():.4f}, fm: {fm_loss.item():.4f})")
                    print(f"  GP_loss: {gp_loss.item():.4f}")
                    
                    with torch.no_grad():
                        generator.eval()
                        sample_prompts = [
                            "a flying airplane in the sky",
                            "a red automobile on the road",
                            "a colorful bird with feathers",
                            "a cute cat with whiskers"
                        ] * 4
                        sample_text_emb = text_encoder.embed_batch(sample_prompts, return_tensor=True).to(device)
                        sample_noise = generate_noise(len(sample_prompts), latent_dim, device)
                        sample_images = generator(sample_noise, sample_text_emb)
                        save_image_grid(
                            sample_images,
                            f'outputs/training/conditional_epoch_{epoch:03d}_batch_{batch_idx:04d}.png',
                            nrow=4
                        )
                        generator.train()
            
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss_real = epoch_d_loss_real / len(dataloader)
            avg_d_loss_fake = epoch_d_loss_fake / len(dataloader)
            avg_gp_loss = epoch_gp_loss / len(dataloader)
            avg_fm_loss = epoch_fm_loss / len(dataloader)
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs} completed")
            print(f"  Avg D_loss: {avg_d_loss:.4f} (real: {avg_d_loss_real:.4f}, fake: {avg_d_loss_fake:.4f})")
            print(f"  Avg G_loss: {avg_g_loss:.4f}")
            print(f"  Avg GP_loss: {avg_gp_loss:.4f}")
            print(f"  Avg FM_loss: {avg_fm_loss:.4f}")
            print(f"{'='*70}\n")
            
            if epoch % checkpoint_interval == 0:
                checkpoint_path = f'checkpoints/conditional_checkpoint_epoch_{epoch:03d}.pth'
                save_conditional_checkpoint(
                    generator, discriminator, text_encoder,
                    g_optimizer, d_optimizer, ema,
                    epoch, avg_g_loss, avg_d_loss,
                    avg_d_loss_real, avg_d_loss_fake,
                    avg_gp_loss, avg_fm_loss,
                    checkpoint_path
                )
            
            with torch.no_grad():
                generator.eval()
                all_class_prompts = [text_encoder.get_class_caption(i) for i in range(10)]
                all_class_prompts = all_class_prompts * 6 + all_class_prompts[:4]
                class_text_emb = text_encoder.embed_batch(all_class_prompts, return_tensor=True).to(device)
                class_noise = generate_noise(len(all_class_prompts), latent_dim, device)
                class_images = generator(class_noise, class_text_emb)
                save_image_grid(
                    class_images,
                    f'outputs/training/conditional_epoch_{epoch:03d}_all_classes.png',
                    nrow=8
                )
                generator.train()
        
        final_checkpoint = 'checkpoints/conditional_final_model.pth'
        save_conditional_checkpoint(
            generator, discriminator, text_encoder,
            g_optimizer, d_optimizer, ema,
            num_epochs, avg_g_loss, avg_d_loss,
            avg_d_loss_real, avg_d_loss_fake,
            avg_gp_loss, avg_fm_loss,
            final_checkpoint
        )
        
        ema_checkpoint = 'checkpoints/conditional_ema_generator.pth'
        os.makedirs(os.path.dirname(ema_checkpoint), exist_ok=True)
        torch.save({'ema_state_dict': ema.state_dict()}, ema_checkpoint)
        print(f"EMA generator saved: {ema_checkpoint}")
        
        print("\n" + "="*70)
        print("CONDITIONAL TRAINING COMPLETED!")
        print(f"Final model saved: {final_checkpoint}")
        print(f"EMA model saved: {ema_checkpoint}")
        print(f"Training images saved: outputs/training/")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def generate_conditional_samples(
    prompts,
    num_samples_per_prompt=4,
    checkpoint_path='checkpoints/conditional_final_model.pth',
    use_ema=True,
    output_dir='outputs/generated',
    latent_dim=100,
    text_embed_dim=50
):
    """
    Generate images for specific text prompts using trained conditional GAN.
    
    Args:
        prompts: list of text prompts
        num_samples_per_prompt: number of images to generate per prompt
        checkpoint_path: path to checkpoint
        use_ema: whether to use EMA generator
        output_dir: directory to save generated images
        latent_dim: dimension of noise vector
        text_embed_dim: dimension of text embeddings
    """
    try:
        set_seed()
        device = get_device()
        
        print("\n" + "="*70)
        print("CONDITIONAL IMAGE GENERATION")
        print("="*70)
        
        print("\nInitializing text encoder...")
        text_encoder = TextEncoder(embedding_dim=text_embed_dim)
        
        print("Creating generator...")
        generator = ConditionalGenerator(
            latent_dim=latent_dim,
            embedding_dim=text_embed_dim
        ).to(device)
        
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            return
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        
        if use_ema and checkpoint.get('ema_state_dict') is not None:
            print("Applying EMA weights...")
            ema = EMA(generator)
            ema.load_state_dict(checkpoint['ema_state_dict'])
            ema.apply_shadow()
            print("EMA weights applied")
        
        generator.eval()
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating images for {len(prompts)} prompts ({num_samples_per_prompt} samples each)...")
        
        all_images = []
        all_prompt_labels = []
        
        with torch.no_grad():
            for prompt_idx, prompt in enumerate(prompts):
                print(f"  Generating for: '{prompt}'")
                
                text_emb = text_encoder.embed_text(prompt, return_tensor=True).unsqueeze(0).to(device)
                text_emb = text_emb.repeat(num_samples_per_prompt, 1)
                
                noise = generate_noise(num_samples_per_prompt, latent_dim, device)
                fake_images = generator(noise, text_emb)
                
                all_images.append(fake_images)
                all_prompt_labels.extend([prompt] * num_samples_per_prompt)
        
        all_images_tensor = torch.cat(all_images, dim=0)
        
        grid_path = f'{output_dir}/conditional_samples_grid.png'
        save_image_grid(all_images_tensor, grid_path, nrow=num_samples_per_prompt)
        
        print(f"\nGenerated images saved:")
        print(f"  Grid: {grid_path}")
        print(f"  Total images: {len(all_images_tensor)}")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


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
    parser.add_argument('--conditional', action='store_true',
                        help='Use conditional GAN (text-to-image with CIFAR-10)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Número de épocas de treinamento (default: 20 for MNIST, 100 for conditional)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Tamanho do batch (default: 128 for MNIST, 32 for conditional)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Número de amostras para gerar')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Caminho do checkpoint (default depends on mode)')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                        help='Text prompts for conditional generation (e.g., "a flying airplane" "a cute dog")')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA generator for conditional generation (default: True)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if args.conditional:
            print("=" * 70)
            print("Modo: Treinamento Condicional (Text-to-Image)")
            print("Dataset: CIFAR-10")
            print("=" * 70)
            
            epochs = args.epochs if args.epochs is not None else 100
            batch_size = args.batch_size if args.batch_size is not None else 32
            
            train_conditional_gan(
                num_epochs=epochs,
                batch_size=batch_size
            )
        else:
            print("=" * 70)
            print("Modo: Treinamento Original (MNIST)")
            print("=" * 70)
            
            epochs = args.epochs if args.epochs is not None else 20
            batch_size = args.batch_size if args.batch_size is not None else 128
            
            train_gan(
                num_epochs=epochs,
                batch_size=batch_size
            )
    else:
        if args.conditional:
            print("=" * 70)
            print("Modo: Geração Condicional (Text-to-Image)")
            print("=" * 70)
            
            checkpoint = args.checkpoint if args.checkpoint else 'checkpoints/conditional_final_model.pth'
            
            if args.prompts is None:
                prompts = [
                    "a flying airplane in the sky",
                    "a red automobile on the road",
                    "a colorful bird with feathers",
                    "a cute cat with whiskers",
                    "a graceful deer in the forest",
                    "a friendly dog with fur"
                ]
                print("Using default prompts (use --prompts to specify custom prompts)")
            else:
                prompts = args.prompts
            
            print(f"Prompts: {prompts}")
            
            generate_conditional_samples(
                prompts=prompts,
                num_samples_per_prompt=4,
                checkpoint_path=checkpoint,
                use_ema=args.use_ema
            )
        else:
            print("=" * 70)
            print("Modo: Geração Original (MNIST)")
            print("=" * 70)
            
            checkpoint = args.checkpoint if args.checkpoint else 'checkpoints/final_model.pth'
            
            generate_samples(
                num_samples=args.samples,
                checkpoint_path=checkpoint
            )
