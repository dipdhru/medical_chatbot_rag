"""Settings shared by the app, the tests and the evaluation script."""

DATA_PATH = "MedQuAD_combined.csv"
EMBEDDINGS_CACHE = "embeddings_cache.npy"
EMBED_MODEL_NAME = "intfloat/e5-base"
GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_K = 3
DEFAULT_THRESHOLD = 0.83  # calibrated by scripts/evaluate_retrieval.py
QUERY_PREFIX = "query: "  # e5 models expect this prefix on queries
