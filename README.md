# Gerador de Imagens com GAN

## Visão Geral
Projeto completo de rede neural geradora de imagens (GAN - Generative Adversarial Network) implementado em PyTorch. O sistema inclui treinamento com MNIST, API REST e interface web para geração de imagens.

## Estrutura do Projeto

### Módulos Principais
- **models.py** - Arquiteturas do Gerador e Discriminador
  - Generator: converte vetor de ruído (100 dim) em imagem 28x28
  - Discriminator: classifica imagens como reais ou falsas
  
- **utils.py** - Funções auxiliares
  - Gerenciamento de checkpoints
  - Conversão de imagens (tensor → PIL → base64)
  - Logging e configuração de seeds
  
- **train.py** - Pipeline de treinamento
  - Dataset: MNIST (dígitos manuscritos)
  - Tracking de perdas do Gerador e Discriminador
  - Salvamento automático de checkpoints
  - Geração de amostras durante treinamento
  
- **api.py** - Servidor FastAPI
  - Endpoint `/generate` - gera imagem única
  - Endpoint `/generate-batch` - gera múltiplas imagens
  - Endpoint `/health` - status da API
  - Endpoint `/model-info` - informações do modelo
  
- **frontend/index.html** - Interface web
  - Botão para gerar imagem individual
  - Botão para gerar lote de imagens
  - Controle de seed para reprodutibilidade
  - Display de imagens geradas

## Como Usar

### 1. Treinamento (Opcional)
Para treinar o modelo com o dataset MNIST:
```bash
python train.py --mode train --epochs 20 --batch_size 128
```

Para gerar amostras de teste:
```bash
python train.py --mode generate --samples 100 --checkpoint checkpoints/final_model.pth
```

### 2. Servidor API
O servidor já está rodando automaticamente na porta 5000. Para iniciar manualmente:
```bash
python api.py
```

### 3. Interface Web
Acesse `/app` no navegador para usar a interface web.

**Nota:** O modelo funciona mesmo sem treinamento prévio (usa pesos aleatórios para demonstração). Para resultados de qualidade, execute o treinamento primeiro.

## Arquitetura da GAN

### Gerador
- Input: vetor de ruído aleatório (100 dimensões)
- Camadas: Linear → Reshape → ConvTranspose2d (3 camadas)
- Ativação: ReLU nas camadas ocultas, Tanh na saída
- Output: imagem 28x28x1 (escala de cinza)

### Discriminador
- Input: imagem 28x28x1
- Camadas: Conv2d (3 camadas) com BatchNorm e Dropout
- Ativação: LeakyReLU(0.2)
- Output: probabilidade [0,1] (real vs falso)

## Endpoints da API

- `GET /` - Informações da API
- `GET /health` - Status de saúde
- `GET /model-info` - Detalhes do modelo
- `POST /generate` - Gera uma imagem
  - Body (opcional): `{"seed": 42}` para reprodutibilidade
- `POST /generate-batch?num_images=4` - Gera múltiplas imagens

## Dependências
- torch / torchvision - Deep Learning
- fastapi / uvicorn - API REST
- pillow - Processamento de imagens
- numpy - Operações numéricas
- python-multipart - Upload de arquivos

## Diretórios
- `checkpoints/` - Modelos salvos durante treinamento
- `outputs/training/` - Amostras geradas durante treinamento
- `outputs/generated/` - Imagens geradas em batch
- `data/` - Dataset MNIST (baixado automaticamente)

## Características
- Seed fixa para reprodutibilidade
- Logging detalhado de todas as operações
- Checkpoints automáticos a cada 5 épocas
- Geração de amostras durante treinamento
- API robusta com tratamento de erros
- Frontend responsivo e intuitivo
- Funciona sem GPU (otimizado para CPU)

## Próximas Melhorias Sugeridas
1. Upgrade para DCGAN com imagens 64x64 coloridas
2. Conditional GAN para gerar dígitos específicos
3. Métricas de qualidade (FID, IS)
4. Gallery persistente de imagens geradas
5. Download de imagens pelo frontend

## Data de Criação
Outubro 2025
