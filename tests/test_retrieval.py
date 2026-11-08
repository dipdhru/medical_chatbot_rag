import pytest

from medrag.retrieval import Retriever


@pytest.fixture
def retriever(encoder, docs_only_text):
    return Retriever(encoder, docs_only_text, encoder.encode(docs_only_text["text"].tolist()))


def test_answers_are_whitespace_normalised(documents):
    assert documents.loc[0, "answer"] == "Wheezing and coughing."


def test_best_match_ranks_first(retriever):
    results, best = retriever.retrieve("symptoms of asthma", k=2, threshold=0.5)
    assert results[0]["focus"] == "Asthma"
    assert results[0]["score"] == pytest.approx(best)
    assert best == pytest.approx(1.0)


def test_below_threshold_is_rejected(retriever):
    results, best = retriever.retrieve("completely unrelated words here", k=2, threshold=0.5)
    assert results is None
    assert best < 0.5


def test_results_without_a_source_url_are_dropped(retriever):
    results, _ = retriever.retrieve("causes of gout", k=3, threshold=0.5)
    assert all(result["url"].startswith("http") for result in results)
    assert "Gout" not in [result["focus"] for result in results]


def test_mismatched_embeddings_raise(encoder, docs_only_text):
    with pytest.raises(ValueError):
        Retriever(encoder, docs_only_text, encoder.encode(["only one"]))
