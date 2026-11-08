import hashlib
import re

import numpy as np
import pandas as pd
import pytest

from medrag.data import prepare_documents


class HashingEncoder:
    """Deterministic bag-of-words encoder, so tests need no model download."""

    dimension = 256

    def encode(self, sentences, **_):
        vectors = np.zeros((len(sentences), self.dimension), dtype="float32")
        for row, sentence in enumerate(sentences):
            for token in re.findall(r"[a-z]+", sentence.lower()):
                if token != "query":
                    bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
                    vectors[row, bucket] += 1
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def encoder():
    return HashingEncoder()


@pytest.fixture
def documents():
    return prepare_documents(
        pd.DataFrame(
            {
                "focus": ["Asthma", "Anemia", "Gout"],
                "question": ["symptoms of asthma", "treatment of anemia", "causes of gout"],
                "question_qtype": ["symptoms", "treatment", "causes"],
                "url": ["https://a.example/asthma", "https://a.example/anemia", None],
                "answer": ["Wheezing\nand  coughing.", "Iron pills.", "Uric acid crystals."],
            }
        )
    )


@pytest.fixture
def docs_only_text(documents):
    return documents.assign(text=documents["question"])
