"""ColPali model/processor loader and embedding/scoring helpers."""
import torch
from colpali_engine.models import ColPali, ColPaliProcessor

MODEL_NAME = "vidore/colpali-v1.2"

_model = None
_processor = None


def get_model_and_processor():
    global _model, _processor
    if _model is None:
        _model = ColPali.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        _processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


def embed_images(images, batch_size=4):
    """images: list of PIL.Image -> list of tensors, each shape (n_patches, 128) on CPU."""
    model, processor = get_model_and_processor()
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor.process_images(batch).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        embeddings.extend(list(out.to(torch.float32).cpu()))
    return embeddings


def embed_queries(texts, batch_size=8):
    """texts: list of str -> list of tensors, each shape (n_tokens, 128) on CPU."""
    model, processor = get_model_and_processor()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor.process_queries(batch).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        embeddings.extend(list(out.to(torch.float32).cpu()))
    return embeddings


def score_multi_vector(query_embeddings, doc_embeddings):
    """query_embeddings: list of (n_q_tokens, 128); doc_embeddings: list of (n_patches, 128).
    Returns a (len(query_embeddings), len(doc_embeddings)) score matrix (MaxSim / late interaction)."""
    _, processor = get_model_and_processor()
    return processor.score_multi_vector(query_embeddings, doc_embeddings)
