"""Embed every knowledge-base document with the 'passage: ' prefix e5 models expect."""

import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medrag import config, data  # noqa: E402

output = Path(sys.argv[1])
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(config.EMBED_MODEL_NAME, device=device)
kb = data.load_knowledge_base(ROOT / config.DATA_PATH)
texts = ("passage: " + kb["text"]).tolist()
order = np.argsort([len(t) for t in texts])  # similar lengths batch faster
vectors = model.encode(
    [texts[i] for i in order],
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False,
).astype("float32")
result = np.empty_like(vectors)
result[order] = vectors
np.save(output, result)
print("saved", result.shape, flush=True)
