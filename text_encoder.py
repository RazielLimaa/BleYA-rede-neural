import torch
import numpy as np
import logging
import os
import re
from typing import List, Optional, Union, Dict
import gensim.downloader as api

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextEncoder:
    """
    TextEncoder para conditional GAN que converte text prompts em embeddings.
    
    Funcionalidades:
    - Carrega e cacheia embeddings GloVe de 50 dimensões
    - Tokeniza e processa text prompts
    - Retorna embedding médio para cada prompt
    - Mapeia classes CIFAR-10 para captions descritivas
    - Funciona apenas com CPU (sem necessidade de CUDA)
    """
    
    CIFAR10_CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    CIFAR10_CLASS_CAPTIONS = {
        0: "a flying airplane in the sky",
        1: "a red automobile on the road",
        2: "a colorful bird with feathers",
        3: "a cute cat with whiskers",
        4: "a graceful deer in the forest",
        5: "a friendly dog with fur",
        6: "a green frog near water",
        7: "a strong horse running",
        8: "a sailing ship on the ocean",
        9: "a large truck on the highway"
    }
    
    def __init__(self, embedding_dim: int = 50, cache_dir: str = './embeddings_cache'):
        """
        Inicializa o TextEncoder.
        
        Args:
            embedding_dim: dimensão dos embeddings GloVe (padrão: 50)
            cache_dir: diretório para cachear embeddings
        """
        self.embedding_dim = embedding_dim
        self.cache_dir = cache_dir
        self.glove_model = None
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"TextEncoder inicializado com embedding_dim={embedding_dim}")
        
        self._load_glove_embeddings()
    
    def _load_glove_embeddings(self):
        """
        Carrega embeddings GloVe usando gensim.
        Downloads do modelo se não estiver presente.
        """
        try:
            logger.info(f"Carregando embeddings GloVe de {self.embedding_dim} dimensões...")
            
            model_name = f'glove-wiki-gigaword-{self.embedding_dim}'
            
            cache_file = os.path.join(self.cache_dir, f'{model_name}.model')
            
            if os.path.exists(cache_file):
                logger.info(f"Carregando modelo GloVe do cache: {cache_file}")
                import pickle
                with open(cache_file, 'rb') as f:
                    self.glove_model = pickle.load(f)
            else:
                logger.info(f"Baixando modelo GloVe '{model_name}' (pode levar alguns minutos)...")
                self.glove_model = api.load(model_name)
                
                logger.info(f"Salvando modelo GloVe no cache: {cache_file}")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.glove_model, f)
            
            logger.info(f"Modelo GloVe carregado com sucesso! Vocabulário: {len(self.glove_model)} palavras")
            
        except Exception as e:
            logger.error(f"Erro ao carregar embeddings GloVe: {str(e)}")
            logger.warning("TextEncoder funcionará com embeddings vazios")
            self.glove_model = None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza texto em palavras.
        - Converte para lowercase
        - Remove caracteres especiais
        - Divide em palavras
        
        Args:
            text: texto para tokenizar
        
        Returns:
            lista de tokens (palavras)
        """
        text = text.lower()
        
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        tokens = text.split()
        
        tokens = [token for token in tokens if token]
        
        return tokens
    
    def get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Obtém embedding de uma palavra do modelo GloVe.
        
        Args:
            word: palavra para obter embedding
        
        Returns:
            embedding numpy array de shape (embedding_dim,) ou None se não encontrada
        """
        if self.glove_model is None:
            return None
        
        if word in self.embedding_cache:
            return self.embedding_cache[word]
        
        try:
            if word in self.glove_model:
                embedding = self.glove_model[word]
                self.embedding_cache[word] = embedding
                return embedding
            else:
                return None
        except Exception as e:
            logger.debug(f"Erro ao obter embedding para '{word}': {str(e)}")
            return None
    
    def embed_text(self, prompt: str, return_tensor: bool = True) -> Union[torch.Tensor, np.ndarray, None]:
        """
        Converte text prompt em embedding médio.
        
        Args:
            prompt: texto para embedar
            return_tensor: se True, retorna torch.Tensor; se False, retorna np.ndarray
        
        Returns:
            embedding médio com shape (embedding_dim,) ou None se nenhuma palavra encontrada
        """
        if not prompt or not prompt.strip():
            logger.warning("Prompt vazio fornecido")
            if return_tensor:
                return torch.zeros(self.embedding_dim)
            else:
                return np.zeros(self.embedding_dim)
        
        tokens = self.tokenize(prompt)
        
        if not tokens:
            logger.warning(f"Nenhum token válido encontrado em: '{prompt}'")
            if return_tensor:
                return torch.zeros(self.embedding_dim)
            else:
                return np.zeros(self.embedding_dim)
        
        embeddings = []
        found_words = []
        missing_words = []
        
        for token in tokens:
            embedding = self.get_word_embedding(token)
            if embedding is not None:
                embeddings.append(embedding)
                found_words.append(token)
            else:
                missing_words.append(token)
        
        if missing_words:
            logger.debug(f"Palavras não encontradas no vocabulário GloVe: {missing_words}")
        
        if not embeddings:
            logger.warning(f"Nenhuma palavra de '{prompt}' encontrada no vocabulário GloVe")
            if return_tensor:
                return torch.zeros(self.embedding_dim)
            else:
                return np.zeros(self.embedding_dim)
        
        avg_embedding = np.mean(embeddings, axis=0)
        
        logger.debug(f"Embedding criado para '{prompt}': {len(found_words)}/{len(tokens)} palavras encontradas")
        
        if return_tensor:
            return torch.from_numpy(avg_embedding).float()
        else:
            return avg_embedding
    
    def embed_batch(self, prompts: List[str], return_tensor: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """
        Converte múltiplos prompts em embeddings.
        
        Args:
            prompts: lista de textos para embedar
            return_tensor: se True, retorna torch.Tensor; se False, retorna np.ndarray
        
        Returns:
            embeddings com shape (batch_size, embedding_dim)
        """
        embeddings = []
        
        for prompt in prompts:
            embedding = self.embed_text(prompt, return_tensor=False)
            if embedding is None:
                embedding = np.zeros(self.embedding_dim)
            embeddings.append(embedding)
        
        batch_embeddings = np.stack(embeddings, axis=0)
        
        if return_tensor:
            return torch.from_numpy(batch_embeddings).float()
        else:
            return batch_embeddings
    
    def get_class_caption(self, class_id: int) -> str:
        """
        Obtém caption descritiva para uma classe CIFAR-10.
        
        Args:
            class_id: ID da classe (0-9)
        
        Returns:
            caption descritiva da classe
        
        Raises:
            ValueError: se class_id estiver fora do intervalo [0, 9]
        """
        if not 0 <= class_id <= 9:
            raise ValueError(f"class_id deve estar entre 0 e 9, recebido: {class_id}")
        
        return self.CIFAR10_CLASS_CAPTIONS[class_id]
    
    def get_class_name(self, class_id: int) -> str:
        """
        Obtém nome da classe CIFAR-10.
        
        Args:
            class_id: ID da classe (0-9)
        
        Returns:
            nome da classe
        
        Raises:
            ValueError: se class_id estiver fora do intervalo [0, 9]
        """
        if not 0 <= class_id <= 9:
            raise ValueError(f"class_id deve estar entre 0 e 9, recebido: {class_id}")
        
        return self.CIFAR10_CLASS_NAMES[class_id]
    
    def embed_class(self, class_id: int, return_tensor: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """
        Obtém embedding para uma classe CIFAR-10.
        
        Args:
            class_id: ID da classe (0-9)
            return_tensor: se True, retorna torch.Tensor; se False, retorna np.ndarray
        
        Returns:
            embedding da caption da classe com shape (embedding_dim,)
        """
        caption = self.get_class_caption(class_id)
        return self.embed_text(caption, return_tensor=return_tensor)
    
    def clear_cache(self):
        """
        Limpa o cache de embeddings de palavras.
        """
        self.embedding_cache.clear()
        logger.info("Cache de embeddings limpo")
    
    def get_embedding_dim(self) -> int:
        """
        Retorna a dimensão dos embeddings.
        
        Returns:
            dimensão dos embeddings
        """
        return self.embedding_dim
    
    def is_loaded(self) -> bool:
        """
        Verifica se o modelo GloVe foi carregado com sucesso.
        
        Returns:
            True se o modelo está carregado, False caso contrário
        """
        return self.glove_model is not None
    
    def get_cache_size(self) -> int:
        """
        Retorna o número de palavras no cache de embeddings.
        
        Returns:
            número de palavras cacheadas
        """
        return len(self.embedding_cache)


if __name__ == "__main__":
    print("="*70)
    print("Testando TextEncoder...")
    print("="*70)
    
    encoder = TextEncoder(embedding_dim=50)
    
    print(f"\nModelo GloVe carregado: {encoder.is_loaded()}")
    print(f"Dimensão dos embeddings: {encoder.get_embedding_dim()}")
    
    print("\n" + "-"*70)
    print("Teste 1: Tokenização")
    print("-"*70)
    test_texts = [
        "Hello, World!",
        "A flying airplane in the sky.",
        "Testing 123... special chars!@#",
        ""
    ]
    
    for text in test_texts:
        tokens = encoder.tokenize(text)
        print(f"'{text}' -> {tokens}")
    
    print("\n" + "-"*70)
    print("Teste 2: Embedding de texto único")
    print("-"*70)
    test_prompts = [
        "a cute dog",
        "flying airplane",
        "beautiful sunset",
        "unknown_word_xyz_123"
    ]
    
    for prompt in test_prompts:
        embedding = encoder.embed_text(prompt)
        print(f"'{prompt}':")
        print(f"  Shape: {embedding.shape}")
        print(f"  Type: {type(embedding)}")
        print(f"  Range: [{embedding.min():.3f}, {embedding.max():.3f}]")
        print(f"  Mean: {embedding.mean():.3f}")
    
    print("\n" + "-"*70)
    print("Teste 3: Embedding em batch")
    print("-"*70)
    batch_prompts = ["a dog", "a cat", "a bird"]
    batch_embeddings = encoder.embed_batch(batch_prompts)
    print(f"Batch prompts: {batch_prompts}")
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    print(f"Batch embeddings type: {type(batch_embeddings)}")
    
    print("\n" + "-"*70)
    print("Teste 4: Classes CIFAR-10")
    print("-"*70)
    print("Classes CIFAR-10 e suas captions:")
    for class_id in range(10):
        class_name = encoder.get_class_name(class_id)
        caption = encoder.get_class_caption(class_id)
        print(f"  {class_id}: {class_name:12} -> '{caption}'")
    
    print("\n" + "-"*70)
    print("Teste 5: Embedding de classes CIFAR-10")
    print("-"*70)
    for class_id in [0, 5, 9]:
        embedding = encoder.embed_class(class_id)
        caption = encoder.get_class_caption(class_id)
        print(f"Classe {class_id} ('{caption}'):")
        print(f"  Shape: {embedding.shape}")
        print(f"  Mean: {embedding.mean():.3f}")
    
    print("\n" + "-"*70)
    print("Teste 6: Cache de embeddings")
    print("-"*70)
    print(f"Tamanho do cache antes: {encoder.get_cache_size()}")
    
    _ = encoder.embed_text("testing cache functionality")
    print(f"Tamanho do cache depois: {encoder.get_cache_size()}")
    
    encoder.clear_cache()
    print(f"Tamanho do cache após limpar: {encoder.get_cache_size()}")
    
    print("\n" + "-"*70)
    print("Teste 7: Casos especiais e erros")
    print("-"*70)
    
    try:
        _ = encoder.get_class_caption(99)
    except ValueError as e:
        print(f"✓ ValueError capturado corretamente: {e}")
    
    empty_embedding = encoder.embed_text("")
    print(f"✓ Embedding de texto vazio: shape={empty_embedding.shape}, all zeros={torch.all(empty_embedding == 0)}")
    
    nonsense_embedding = encoder.embed_text("xyzabc123def456")
    print(f"✓ Embedding de texto sem palavras válidas: shape={nonsense_embedding.shape}, all zeros={torch.all(nonsense_embedding == 0)}")
    
    print("\n" + "="*70)
    print("Todos os testes concluídos com sucesso!")
    print("="*70)
