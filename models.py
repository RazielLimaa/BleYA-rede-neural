import torch
import torch.nn as nn

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


def initialize_weights(model):
    """
    Inicializa os pesos do modelo com distribuição normal.
    Melhora a estabilidade do treinamento da GAN.
    
    Args:
        model: modelo PyTorch (Generator ou Discriminator)
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    # Teste das arquiteturas
    print("Testando arquiteturas da GAN...")
    
    # Cria os modelos
    latent_dim = 100
    generator = Generator(latent_dim)
    discriminator = Discriminator()
    
    # Inicializa os pesos
    initialize_weights(generator)
    initialize_weights(discriminator)
    
    # Teste do gerador
    batch_size = 4
    z = torch.randn(batch_size, latent_dim)
    fake_images = generator(z)
    print(f"Gerador - Input shape: {z.shape} -> Output shape: {fake_images.shape}")
    print(f"Gerador - Output range: [{fake_images.min():.2f}, {fake_images.max():.2f}]")
    
    # Teste do discriminador
    predictions = discriminator(fake_images)
    print(f"\nDiscriminador - Input shape: {fake_images.shape} -> Output shape: {predictions.shape}")
    print(f"Discriminador - Output range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    # Conta parâmetros
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\nParâmetros do Gerador: {gen_params:,}")
    print(f"Parâmetros do Discriminador: {disc_params:,}")
    print("\nModelos criados com sucesso!")
