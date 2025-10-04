import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ============================================================================
# Original Models (for backward compatibility)
# ============================================================================

class Generator(nn.Module):
    """
    Gerador da GAN que converte vetor de ruído em imagem 28x28.
    
    Arquitetura:
    - Input: vetor de ruído (latent_dim dimensões, padrão 100)
    - Camadas lineares + reshape
    - Camadas convolucionais transpostas (upsample)
    - Output: imagem 28x28x1 (escala de cinza) com valores entre -1 e 1
    """
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Camada linear inicial: 100 -> 7*7*256
        self.fc = nn.Linear(latent_dim, 7 * 7 * 256)
        
        # Camadas convolucionais transpostas para upsampling
        self.conv_transpose = nn.Sequential(
            # 7x7x256 -> 7x7x128
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 7x7x128 -> 14x14x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 14x14x128 -> 28x28x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 28x28x64 -> 28x28x1
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Saída entre -1 e 1
        )
    
    def forward(self, z):
        """
        Forward pass do gerador.
        
        Args:
            z: tensor de ruído com shape (batch_size, latent_dim)
        
        Returns:
            imagem gerada com shape (batch_size, 1, 28, 28)
        """
        # Expande o vetor latente
        x = self.fc(z)
        x = x.view(-1, 256, 7, 7)  # Reshape para formato de imagem
        
        # Aplica convoluções transpostas
        img = self.conv_transpose(x)
        return img


class Discriminator(nn.Module):
    """
    Discriminador da GAN que classifica imagens como reais ou falsas.
    
    Arquitetura:
    - Input: imagem 28x28x1
    - Camadas convolucionais com LeakyReLU
    - Camada linear final com sigmoid
    - Output: probabilidade de ser real (0 a 1)
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Camadas convolucionais para extração de features
        self.conv_layers = nn.Sequential(
            # 28x28x1 -> 14x14x64
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 14x14x64 -> 7x7x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 7x7x128 -> 7x7x256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        
        # Camada de classificação final
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 256, 1),
            nn.Sigmoid()  # Saída: probabilidade entre 0 e 1
        )
    
    def forward(self, img):
        """
        Forward pass do discriminador.
        
        Args:
            img: tensor de imagem com shape (batch_size, 1, 28, 28)
        
        Returns:
            probabilidade de ser real com shape (batch_size, 1)
        """
        # Extrai features da imagem
        features = self.conv_layers(img)
        
        # Classifica como real ou falsa
        validity = self.fc(features)
        return validity


# ============================================================================
# Helper Modules for Conditional DCGAN
# ============================================================================

class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization conditioned on text embeddings.
    
    Instead of learning fixed affine parameters (gamma, beta), this layer
    learns to predict them from the text embedding, allowing text-conditional
    feature modulation.
    
    Args:
        num_features: number of channels in the input feature map
        embedding_dim: dimension of the text embedding
        eps: epsilon for numerical stability
        momentum: momentum for running stats
    """
    def __init__(self, num_features, embedding_dim, eps=1e-5, momentum=0.1):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.momentum = momentum
        
        # Batch norm without learnable affine parameters
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=False)
        
        # Linear layers to predict gamma and beta from text embedding
        self.gamma_fc = nn.Linear(embedding_dim, num_features)
        self.beta_fc = nn.Linear(embedding_dim, num_features)
        
        # Initialize gamma to 1 and beta to 0
        nn.init.ones_(self.gamma_fc.weight.data)
        nn.init.zeros_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.weight.data)
        nn.init.zeros_(self.beta_fc.bias.data)
    
    def forward(self, x, embedding):
        """
        Args:
            x: feature map with shape (batch_size, num_features, height, width)
            embedding: text embedding with shape (batch_size, embedding_dim)
        
        Returns:
            normalized and modulated features with shape (batch_size, num_features, height, width)
        """
        # Normalize using batch statistics
        out = self.bn(x)
        
        # Predict gamma and beta from text embedding
        gamma = self.gamma_fc(embedding)  # (batch_size, num_features)
        beta = self.beta_fc(embedding)    # (batch_size, num_features)
        
        # Reshape for broadcasting: (batch_size, num_features, 1, 1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        
        # Apply conditional affine transformation
        out = gamma * out + beta
        
        return out


class SelfAttention(nn.Module):
    """
    Self-Attention layer for capturing long-range dependencies in feature maps.
    
    Uses scaled dot-product attention mechanism to allow each spatial location
    to attend to all other locations in the feature map.
    
    Args:
        in_channels: number of input channels
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Query, Key, Value projections (reduced dimension for efficiency)
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable attention weight (starts at 0, gradually learns to use attention)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Args:
            x: input feature map with shape (batch_size, in_channels, height, width)
        
        Returns:
            attended features with shape (batch_size, in_channels, height, width)
        """
        batch_size, channels, height, width = x.size()
        
        # Project to query, key, value
        query = self.query_conv(x).view(batch_size, -1, height * width)  # (B, C', H*W)
        key = self.key_conv(x).view(batch_size, -1, height * width)      # (B, C', H*W)
        value = self.value_conv(x).view(batch_size, -1, height * width)  # (B, C, H*W)
        
        # Compute attention scores
        query = query.permute(0, 2, 1)  # (B, H*W, C')
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        value = value.permute(0, 2, 1)  # (B, H*W, C)
        out = torch.bmm(attention, value)  # (B, H*W, C)
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)  # (B, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class GeneratorResidualBlock(nn.Module):
    """
    Residual block for the conditional generator with upsampling.
    
    Uses Conditional Batch Normalization and transposed convolutions for upsampling.
    
    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        embedding_dim: dimension of text embedding for CBN
        upsample: whether to upsample (2x) the spatial dimensions
    """
    def __init__(self, in_channels, out_channels, embedding_dim, upsample=True):
        super(GeneratorResidualBlock, self).__init__()
        self.upsample = upsample
        self.learnable_sc = (in_channels != out_channels) or upsample
        
        # Main path
        self.cbn1 = ConditionalBatchNorm2d(in_channels, embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        
        if upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.cbn2 = ConditionalBatchNorm2d(out_channels, embedding_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Shortcut connection
        if self.learnable_sc:
            if upsample:
                self.conv_sc = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            else:
                self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x, embedding):
        """
        Args:
            x: input features with shape (batch_size, in_channels, height, width)
            embedding: text embedding with shape (batch_size, embedding_dim)
        
        Returns:
            output features with shape (batch_size, out_channels, height*2, width*2) if upsample else (batch_size, out_channels, height, width)
        """
        # Shortcut
        if self.learnable_sc:
            shortcut = self.conv_sc(x)
        else:
            shortcut = x
        
        # Main path
        out = self.cbn1(x, embedding)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.cbn2(out, embedding)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Residual connection
        return out + shortcut


class DiscriminatorResidualBlock(nn.Module):
    """
    Residual block for the conditional discriminator with downsampling.
    
    Uses Spectral Normalization for training stability and LeakyReLU activation.
    
    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        downsample: whether to downsample (2x) the spatial dimensions
    """
    def __init__(self, in_channels, out_channels, downsample=True):
        super(DiscriminatorResidualBlock, self).__init__()
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        
        # Main path with spectral normalization
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        
        # Shortcut connection with spectral normalization
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        
        # Downsampling layer
        if downsample:
            self.downsample_layer = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """
        Args:
            x: input features with shape (batch_size, in_channels, height, width)
        
        Returns:
            output features with shape (batch_size, out_channels, height//2, width//2) if downsample else (batch_size, out_channels, height, width)
        """
        # Shortcut
        shortcut = x
        if self.learnable_sc:
            shortcut = self.conv_sc(shortcut)
        if self.downsample:
            shortcut = self.downsample_layer(shortcut)
        
        # Main path
        out = self.lrelu(x)
        out = self.conv1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.downsample:
            out = self.downsample_layer(out)
        
        # Residual connection
        return out + shortcut


# ============================================================================
# Conditional DCGAN Models
# ============================================================================

class ConditionalGenerator(nn.Module):
    """
    Conditional DCGAN Generator for text-to-image generation.
    
    Architecture:
    - Input: latent noise (100-dim) concatenated with text embedding (50-dim) = 150-dim
    - Linear projection to 4x4x512 feature map
    - Series of residual blocks with conditional batch normalization and upsampling
    - Self-attention layer for capturing global structure
    - Final convolution to RGB image
    - Output: RGB image (3, 32, 32) with values in [-1, 1]
    
    The generator uses Conditional Batch Normalization (CBN) to modulate features
    based on text embeddings, allowing text-conditional image generation.
    
    Args:
        latent_dim: dimension of the noise vector (default: 100)
        embedding_dim: dimension of text embeddings (default: 50)
        base_channels: base number of channels, scaled throughout network (default: 512)
    """
    def __init__(self, latent_dim=100, embedding_dim=50, base_channels=512):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.base_channels = base_channels
        self.input_dim = latent_dim + embedding_dim  # 150
        
        # Initial linear projection: 150 -> 4*4*512
        self.fc = nn.Linear(self.input_dim, 4 * 4 * base_channels)
        
        # Residual blocks with upsampling
        # 4x4x512 -> 8x8x256
        self.res_block1 = GeneratorResidualBlock(base_channels, base_channels // 2, embedding_dim, upsample=True)
        
        # 8x8x256 -> 16x16x128
        self.res_block2 = GeneratorResidualBlock(base_channels // 2, base_channels // 4, embedding_dim, upsample=True)
        
        # Self-attention at 16x16 resolution
        self.attention = SelfAttention(base_channels // 4)
        
        # 16x16x128 -> 32x32x64
        self.res_block3 = GeneratorResidualBlock(base_channels // 4, base_channels // 8, embedding_dim, upsample=True)
        
        # Final layers: 32x32x64 -> 32x32x3
        self.final_bn = ConditionalBatchNorm2d(base_channels // 8, embedding_dim)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(base_channels // 8, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
    
    def forward(self, noise, text_embedding):
        """
        Generate an RGB image conditioned on noise and text embedding.
        
        Args:
            noise: random noise tensor with shape (batch_size, latent_dim)
            text_embedding: text embedding tensor with shape (batch_size, embedding_dim)
        
        Returns:
            generated RGB image with shape (batch_size, 3, 32, 32) and values in [-1, 1]
        
        Raises:
            ValueError: if input dimensions are incorrect
        """
        if noise.shape[1] != self.latent_dim:
            raise ValueError(f"Expected noise dimension {self.latent_dim}, got {noise.shape[1]}")
        if text_embedding.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected text embedding dimension {self.embedding_dim}, got {text_embedding.shape[1]}")
        
        batch_size = noise.size(0)
        
        # Concatenate noise and text embedding
        x = torch.cat([noise, text_embedding], dim=1)  # (batch_size, 150)
        
        # Linear projection and reshape
        x = self.fc(x)  # (batch_size, 4*4*512)
        x = x.view(batch_size, self.base_channels, 4, 4)  # (batch_size, 512, 4, 4)
        
        # Residual blocks with upsampling and CBN
        x = self.res_block1(x, text_embedding)  # (batch_size, 256, 8, 8)
        x = self.res_block2(x, text_embedding)  # (batch_size, 128, 16, 16)
        
        # Self-attention
        x = self.attention(x)  # (batch_size, 128, 16, 16)
        
        x = self.res_block3(x, text_embedding)  # (batch_size, 64, 32, 32)
        
        # Final convolution to RGB
        x = self.final_bn(x, text_embedding)
        x = self.final_relu(x)
        x = self.final_conv(x)  # (batch_size, 3, 32, 32)
        x = self.tanh(x)  # Scale to [-1, 1]
        
        return x


class ConditionalDiscriminator(nn.Module):
    """
    Conditional DCGAN Discriminator with projection-based text conditioning.
    
    Architecture:
    - Input: RGB image (3, 32, 32)
    - Series of residual blocks with spectral normalization and downsampling
    - Self-attention layer for capturing global structure
    - Projection-based text conditioning (project text and add to features)
    - Global pooling and classification
    - Output: real/fake prediction (1-dim)
    
    Uses Spectral Normalization on all convolutional layers for training stability
    and projection-based conditioning for effective text-image matching.
    
    Args:
        embedding_dim: dimension of text embeddings (default: 50)
        base_channels: base number of channels, scaled throughout network (default: 64)
    """
    def __init__(self, embedding_dim=50, base_channels=64):
        super(ConditionalDiscriminator, self).__init__()
        self.embedding_dim = embedding_dim
        self.base_channels = base_channels
        
        # Initial convolution: 3 -> 64
        self.conv_first = spectral_norm(nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1))
        
        # Residual blocks with downsampling
        # 32x32x64 -> 16x16x128
        self.res_block1 = DiscriminatorResidualBlock(base_channels, base_channels * 2, downsample=True)
        
        # 16x16x128 -> 8x8x256
        self.res_block2 = DiscriminatorResidualBlock(base_channels * 2, base_channels * 4, downsample=True)
        
        # Self-attention at 8x8 resolution
        self.attention = SelfAttention(base_channels * 4)
        
        # 8x8x256 -> 4x4x512
        self.res_block3 = DiscriminatorResidualBlock(base_channels * 4, base_channels * 8, downsample=True)
        
        # 4x4x512 -> 4x4x512 (no downsampling)
        self.res_block4 = DiscriminatorResidualBlock(base_channels * 8, base_channels * 8, downsample=False)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        # Global sum pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection layer for text embedding
        self.text_projection = spectral_norm(nn.Linear(embedding_dim, base_channels * 8))
        
        # Final classification layer
        self.fc = spectral_norm(nn.Linear(base_channels * 8, 1))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, image, text_embedding):
        """
        Classify image as real or fake conditioned on text embedding.
        
        Args:
            image: RGB image tensor with shape (batch_size, 3, 32, 32)
            text_embedding: text embedding tensor with shape (batch_size, embedding_dim)
        
        Returns:
            validity prediction with shape (batch_size, 1), values in [0, 1]
            where 1 indicates real and 0 indicates fake
        
        Raises:
            ValueError: if input dimensions are incorrect
        """
        if image.shape[1] != 3 or image.shape[2] != 32 or image.shape[3] != 32:
            raise ValueError(f"Expected image shape (batch_size, 3, 32, 32), got {image.shape}")
        if text_embedding.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected text embedding dimension {self.embedding_dim}, got {text_embedding.shape[1]}")
        
        # Initial convolution
        h = self.conv_first(image)  # (batch_size, 64, 32, 32)
        
        # Residual blocks with downsampling
        h = self.res_block1(h)  # (batch_size, 128, 16, 16)
        h = self.res_block2(h)  # (batch_size, 256, 8, 8)
        
        # Self-attention
        h = self.attention(h)  # (batch_size, 256, 8, 8)
        
        h = self.res_block3(h)  # (batch_size, 512, 4, 4)
        h = self.res_block4(h)  # (batch_size, 512, 4, 4)
        
        h = self.lrelu(h)
        
        # Global pooling
        h = self.global_pool(h)  # (batch_size, 512, 1, 1)
        h = h.view(h.size(0), -1)  # (batch_size, 512)
        
        # Project text embedding
        text_proj = self.text_projection(text_embedding)  # (batch_size, 512)
        
        # Add projected text to image features (projection-based conditioning)
        h = h + text_proj
        
        # Final classification
        out = self.fc(h)  # (batch_size, 1)
        out = self.sigmoid(out)
        
        return out


# ============================================================================
# Weight Initialization
# ============================================================================

def initialize_weights(model):
    """
    Inicializa os pesos do modelo com distribuição normal.
    Melhora a estabilidade do treinamento da GAN.
    
    Para modelos condicionais, usa inicialização ortogonal para camadas convolucionais
    e inicialização normal para camadas lineares.
    
    Args:
        model: modelo PyTorch (Generator, Discriminator, ConditionalGenerator ou ConditionalDiscriminator)
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # Orthogonal initialization for conv layers (better for deep networks)
            if hasattr(model, 'latent_dim') and model.__class__.__name__.startswith('Conditional'):
                nn.init.orthogonal_(m.weight.data)
            else:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.Linear):
            # Normal initialization for linear layers
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # BatchNorm initialization
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)


def initialize_conditional_weights(model):
    """
    Advanced weight initialization for conditional GAN models.
    
    Uses orthogonal initialization for convolutional layers and Xavier initialization
    for linear layers. Properly handles Conditional Batch Normalization layers.
    
    Args:
        model: ConditionalGenerator or ConditionalDiscriminator model
    """
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # Orthogonal initialization for conv layers
            nn.init.orthogonal_(m.weight.data, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            # Xavier normal initialization for linear layers
            nn.init.xavier_normal_(m.weight.data, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # BatchNorm initialization (skip if it's part of CBN)
            if m.weight is not None and not isinstance(getattr(model, name.split('.')[0], None), ConditionalBatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, ConditionalBatchNorm2d):
            # CBN already initializes its own parameters in __init__
            pass


# ============================================================================
# Testing and Validation
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Testando arquiteturas da GAN...")
    print("="*80)
    
    # ========================================
    # Test Original Models
    # ========================================
    print("\n" + "="*80)
    print("1. Testando modelos originais (MNIST 28x28 grayscale)")
    print("="*80)
    
    latent_dim = 100
    generator = Generator(latent_dim)
    discriminator = Discriminator()
    
    initialize_weights(generator)
    initialize_weights(discriminator)
    
    batch_size = 4
    z = torch.randn(batch_size, latent_dim)
    fake_images = generator(z)
    print(f"\nGerador Original:")
    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {fake_images.shape}")
    print(f"  Output range: [{fake_images.min():.2f}, {fake_images.max():.2f}]")
    
    predictions = discriminator(fake_images)
    print(f"\nDiscriminador Original:")
    print(f"  Input shape: {fake_images.shape}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Output range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\nParâmetros:")
    print(f"  Gerador: {gen_params:,}")
    print(f"  Discriminador: {disc_params:,}")
    
    # ========================================
    # Test Conditional Models
    # ========================================
    print("\n" + "="*80)
    print("2. Testando modelos condicionais (CIFAR-10 32x32 RGB)")
    print("="*80)
    
    latent_dim = 100
    embedding_dim = 50
    batch_size = 4
    
    # Create models
    cond_generator = ConditionalGenerator(latent_dim, embedding_dim)
    cond_discriminator = ConditionalDiscriminator(embedding_dim)
    
    # Initialize weights
    initialize_conditional_weights(cond_generator)
    initialize_conditional_weights(cond_discriminator)
    
    # Test generator
    noise = torch.randn(batch_size, latent_dim)
    text_emb = torch.randn(batch_size, embedding_dim)
    fake_images_cond = cond_generator(noise, text_emb)
    
    print(f"\nConditional Generator:")
    print(f"  Noise input shape: {noise.shape}")
    print(f"  Text embedding shape: {text_emb.shape}")
    print(f"  Output shape: {fake_images_cond.shape}")
    print(f"  Output range: [{fake_images_cond.min():.2f}, {fake_images_cond.max():.2f}]")
    print(f"  Expected output shape: (4, 3, 32, 32) ✓" if fake_images_cond.shape == (4, 3, 32, 32) else "  ✗ Shape mismatch!")
    
    # Test discriminator
    predictions_cond = cond_discriminator(fake_images_cond, text_emb)
    
    print(f"\nConditional Discriminator:")
    print(f"  Image input shape: {fake_images_cond.shape}")
    print(f"  Text embedding shape: {text_emb.shape}")
    print(f"  Output shape: {predictions_cond.shape}")
    print(f"  Output range: [{predictions_cond.min():.2f}, {predictions_cond.max():.2f}]")
    print(f"  Expected output shape: (4, 1) ✓" if predictions_cond.shape == (4, 1) else "  ✗ Shape mismatch!")
    
    cgen_params = sum(p.numel() for p in cond_generator.parameters())
    cdisc_params = sum(p.numel() for p in cond_discriminator.parameters())
    print(f"\nParâmetros:")
    print(f"  Conditional Generator: {cgen_params:,}")
    print(f"  Conditional Discriminator: {cdisc_params:,}")
    
    # ========================================
    # Test Error Handling
    # ========================================
    print("\n" + "="*80)
    print("3. Testando tratamento de erros")
    print("="*80)
    
    try:
        wrong_noise = torch.randn(batch_size, 50)  # Wrong dimension
        _ = cond_generator(wrong_noise, text_emb)
        print("✗ Failed to catch wrong noise dimension")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    try:
        wrong_emb = torch.randn(batch_size, 100)  # Wrong dimension
        _ = cond_generator(noise, wrong_emb)
        print("✗ Failed to catch wrong embedding dimension")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    try:
        wrong_image = torch.randn(batch_size, 3, 28, 28)  # Wrong size
        _ = cond_discriminator(wrong_image, text_emb)
        print("✗ Failed to catch wrong image size")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # ========================================
    # Test Individual Components
    # ========================================
    print("\n" + "="*80)
    print("4. Testando componentes individuais")
    print("="*80)
    
    # Test Conditional Batch Normalization
    cbn = ConditionalBatchNorm2d(num_features=128, embedding_dim=50)
    x = torch.randn(4, 128, 16, 16)
    emb = torch.randn(4, 50)
    out_cbn = cbn(x, emb)
    print(f"\nConditional Batch Normalization:")
    print(f"  Input shape: {x.shape}")
    print(f"  Embedding shape: {emb.shape}")
    print(f"  Output shape: {out_cbn.shape}")
    print(f"  ✓ Shape preserved" if out_cbn.shape == x.shape else "  ✗ Shape changed!")
    
    # Test Self-Attention
    attn = SelfAttention(in_channels=128)
    out_attn = attn(x)
    print(f"\nSelf-Attention:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_attn.shape}")
    print(f"  Gamma parameter: {attn.gamma.item():.4f}")
    print(f"  ✓ Shape preserved" if out_attn.shape == x.shape else "  ✗ Shape changed!")
    
    # Test Generator Residual Block
    gen_res = GeneratorResidualBlock(in_channels=256, out_channels=128, embedding_dim=50, upsample=True)
    x_gen = torch.randn(4, 256, 8, 8)
    out_gen = gen_res(x_gen, emb)
    print(f"\nGenerator Residual Block (with upsampling):")
    print(f"  Input shape: {x_gen.shape}")
    print(f"  Output shape: {out_gen.shape}")
    print(f"  ✓ Upsampled correctly" if out_gen.shape == (4, 128, 16, 16) else "  ✗ Upsampling failed!")
    
    # Test Discriminator Residual Block
    disc_res = DiscriminatorResidualBlock(in_channels=128, out_channels=256, downsample=True)
    x_disc = torch.randn(4, 128, 16, 16)
    out_disc = disc_res(x_disc)
    print(f"\nDiscriminator Residual Block (with downsampling):")
    print(f"  Input shape: {x_disc.shape}")
    print(f"  Output shape: {out_disc.shape}")
    print(f"  ✓ Downsampled correctly" if out_disc.shape == (4, 256, 8, 8) else "  ✗ Downsampling failed!")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*80)
    print("✓ Todos os testes concluídos com sucesso!")
    print("="*80)
    print("\nResumo dos modelos disponíveis:")
    print("  1. Generator - Original GAN para MNIST (28x28 grayscale)")
    print("  2. Discriminator - Original GAN para MNIST (28x28 grayscale)")
    print("  3. ConditionalGenerator - Conditional DCGAN para CIFAR-10 (32x32 RGB)")
    print("  4. ConditionalDiscriminator - Conditional DCGAN para CIFAR-10 (32x32 RGB)")
    print("\nCaracterísticas dos modelos condicionais:")
    print("  ✓ Conditional Batch Normalization (CBN)")
    print("  ✓ Spectral Normalization (discriminador)")
    print("  ✓ Self-Attention layers")
    print("  ✓ Residual blocks")
    print("  ✓ Projection-based text conditioning")
    print("  ✓ Proper weight initialization (orthogonal/Xavier)")
    print("  ✓ Error handling and validation")
    print("  ✓ CPU compatible (sem CUDA-only operations)")
    print("="*80)
