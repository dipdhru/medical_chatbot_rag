from types import SimpleNamespace

from medrag import generation as g

DOCS = [
    {"text": "ctx one", "url": "https://x/1", "focus": "F", "answer": "stored answer"},
    {"text": "ctx two", "url": "https://x/1", "focus": "F", "answer": "b"},
    {"text": "ctx three", "url": "https://x/2", "focus": "G", "answer": "c"},
]


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    def invoke(self, _prompt):
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.reply)


def test_prompt_contains_all_context_and_question():
    prompt = g.build_prompt("why?", DOCS)
    assert all(text in prompt for text in ("ctx one", "ctx two", "ctx three", "why?"))


def test_answer_cites_each_source_once():
    answer = g.generate_answer("q", DOCS, FakeLLM("Grounded."))
    assert answer.startswith("Grounded.")
    assert answer.count("https://x/1") == 1
    assert "https://x/2" in answer


def test_prompts_own_refusal_wording_is_detected():
    reply = "The information is not available in the provided documents."
    assert g.generate_answer("q", DOCS, FakeLLM(reply)) == g.NOT_COVERED_MESSAGE


def test_no_llm_falls_back_to_source():
    assert "stored answer" in g.generate_answer("q", DOCS, None)


def test_llm_failure_or_empty_reply_falls_back():
    for llm in (FakeLLM(error=RuntimeError("down")), FakeLLM("   ")):
        assert "AI synthesis unavailable" in g.generate_answer("q", DOCS, llm)
