"""Load MedQuAD and build the text that gets embedded."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def prepare_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise answers and add the composite ``text`` column used for retrieval."""
    df = df.copy()
    df["answer"] = (
        df["answer"]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"  +", " ", regex=True)
        .str.strip()
    )
    df["text"] = (
        "Focus Area: " + df["focus"].fillna("")
        + " ; Question: " + df["question"].fillna("")
        + " ; Question Type: " + df["question_qtype"].fillna("")
        + " ; Source: " + df["url"].fillna("")
        + " ; Answer: " + df["answer"].fillna("")
    )
    return df


def load_knowledge_base(path: str | Path) -> pd.DataFrame:
    return prepare_documents(pd.read_csv(path))
