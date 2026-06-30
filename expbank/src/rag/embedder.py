"""
Text Vector Encoding Module

Uses BGE-large-en-v1.5 model to encode text into vectors for semantic similarity computation in experience retrieval.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """
    BGE-based Text Vector Encoder

    Uses sentence-transformers library to load BGE-large-en-v1.5 model,
    encoding text into high-dimensional vectors for semantic similarity retrieval.

    Attributes:
        model_path: Model path
        model: sentence-transformers model instance
        dimension: Vector dimension (1024 for BGE-large)
        device: Actual device used (cuda or cpu)

    Example:
        >>> embedder = BGEEmbedder("models/bge-large-en-v1.5")
        >>> vectors = embedder.encode(["vulnerability description 1", "vulnerability description 2"])
        >>> print(vectors.shape)  # (2, 1024)
    """

    @staticmethod
    def _resolve_device(preferred_device: str | None) -> str:
        """
        Resolve and validate device availability, supports GPU/CPU automatic fallback

        Args:
            preferred_device: Preferred device ('cuda', 'cpu', etc.)

        Returns:
            Final available device name
        """
        if preferred_device == "cuda" or preferred_device is None:
            try:
                import torch

                if torch.cuda.is_available():
                    logger.debug("CUDA device available")
                    return "cuda"
                else:
                    logger.debug("CUDA device not available, will use CPU")
                    return "cpu"
            except ImportError:
                logger.debug("PyTorch not installed, using CPU")
                return "cpu"
        else:
            return preferred_device

    def __init__(self, model_path: str = "models/bge-large-en-v1.5", device: str | None = "cpu"):
        """
        Initialize BGE vector encoder, supports GPU/CPU automatic fallback

        Args:
            model_path: Model file path, defaults to local BGE model directory
            device: Running device ('cpu', 'cuda', etc.), None for auto-select
                    If GPU memory is insufficient or unavailable, will automatically fallback to CPU

        Raises:
            ImportError: sentence-transformers library not installed
            FileNotFoundError: Model path does not exist
        """
        self.model_path = Path(model_path)

        # Check model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        # Lazy import to avoid error if not installed
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers library first: pip install sentence-transformers")

        # Determine final device (supports automatic fallback)
        final_device = self._resolve_device(device)

        logger.info(f"Loading BGE model: {self.model_path}")
        logger.info(f"Using device: {final_device}")

        try:
            self.model = SentenceTransformer(str(self.model_path), device=final_device)
            self.device = final_device
        except RuntimeError as e:
            # If GPU loading fails (e.g., insufficient memory), automatically fallback to CPU
            if "cuda" in str(e).lower() or "out of memory" in str(e).lower():
                logger.warning(f"GPU loading failed ({e}), automatically falling back to CPU")
                self.model = SentenceTransformer(str(self.model_path), device="cpu")
                self.device = "cpu"
            else:
                raise

        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded, vector dimension: {self.dimension}, running device: {self.device}")

    def encode(
        self, texts: str | list[str], batch_size: int = 32, show_progress: bool = False, normalize: bool = True
    ) -> list[list[float]]:
        """
        Encode text into vectors, automatically reduce batch_size when GPU memory is insufficient

        Args:
            texts: Single text or text list
            batch_size: Batch size for processing large amounts of text
            show_progress: Whether to show progress bar
            normalize: Whether to normalize vectors with L2 (recommended for cosine similarity)

        Returns:
            Vector list, each vector is a float list

        Example:
            >>> embedder = BGEEmbedder()
            >>> vecs = embedder.encode(["hello world", "hello world"])
            >>> len(vecs)  # 2
            >>> len(vecs[0])  # 1024
        """
        # Normalize to list
        if isinstance(texts, str):
            texts = [texts]

        # Try encoding with initial batch_size, if failed then decrease batch_size
        current_batch_size = batch_size
        while current_batch_size > 0:
            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=current_batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=normalize,
                )
                if current_batch_size < batch_size:
                    logger.info(f"Successfully encoded, using batch_size={current_batch_size}")
                # Convert to Python list (ChromaDB needs)
                return embeddings.tolist()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    # Insufficient memory, reduce batch_size and retry
                    previous_batch_size = current_batch_size
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(
                        f"Insufficient memory, batch_size reduced from {previous_batch_size} to {current_batch_size}"
                    )
                    continue
                else:
                    raise

        # If batch_size reduced to 0 still fails, single sample exceeds GPU memory limit
        logger.error("Cannot encode text: still insufficient memory even with batch_size=1, will retry with CPU")
        # Force switch to CPU
        self.model.to("cpu")
        self.device = "cpu"
        embeddings = self.model.encode(
            texts, batch_size=1, show_progress_bar=show_progress, normalize_embeddings=normalize
        )
        return embeddings.tolist()

    def encode_single(self, text: str, normalize: bool = True) -> list[float]:
        """
        Encode single text (convenience method)

        Args:
            text: Text to encode
            normalize: Whether to normalize

        Returns:
            Vector (float list)
        """
        return self.encode([text], normalize=normalize)[0]


# Convenience function: create default encoder
def create_embedder(model_path: str = "models/bge-large-en-v1.5", device: str | None = None) -> BGEEmbedder:
    """
    Convenience function to create BGE vector encoder

    Args:
        model_path: Model path
        device: Running device

    Returns:
        BGEEmbedder instance
    """
    return BGEEmbedder(model_path=model_path, device=device)
