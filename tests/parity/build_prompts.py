"""Regenerate ``prompts.jsonl`` deterministically: ``uv run python tests/parity/build_prompts.py``.

100 prompts: 70 plain-text completions and 30 chat conversations (rendered with each model's own
chat template at test time). Topics and styles are varied so the suite covers code, numbers,
lists, multilingual text, and long-ish contexts rather than one repetitive pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

TOPICS = [
    "photosynthesis",
    "binary search trees",
    "the French Revolution",
    "TCP congestion control",
    "compound interest",
    "black holes",
    "the water cycle",
    "garbage collection in the JVM",
    "the Pythagorean theorem",
    "volcanoes",
    "public-key cryptography",
    "the human immune system",
    "supply and demand",
    "the Roman aqueducts",
    "neural network backpropagation",
    "plate tectonics",
    "database indexing",
    "the history of jazz",
    "climate feedback loops",
    "consensus algorithms like Raft",
]
COMPLETION_TEMPLATES = [
    "Explain {t} to a curious twelve-year-old:",
    "Here are three surprising facts about {t}:\n1.",
    "Q: What is the most common misconception about {t}?\nA:",
    "Write a short Python function related to {t}, with a docstring.\n\n```python\n",
]
EXTRA_COMPLETIONS = [
    "The quick brown fox",
    "1, 1, 2, 3, 5, 8, 13,",
    "def fibonacci(n):",
    "SELECT name, COUNT(*) FROM orders",
    "Bonjour, je m'appelle Marie et je",
    "Der schnellste Weg nach Berlin ist",
    "In 1969, the Apollo 11 mission",
    "Once upon a time, in a village by the sea,",
    "The capital of Australia is",
    "To be, or not to be,",
]
CHAT_TEMPLATES = [
    "Can you summarize {t} in two sentences?",
    "What are the main trade-offs involved in {t}? Answer as a bulleted list.",
    "I am preparing a lecture on {t}. Suggest an outline.",
]


def build() -> list[dict[str, object]]:
    prompts: list[dict[str, object]] = []
    for topic in TOPICS:
        for tmpl in COMPLETION_TEMPLATES[:3]:
            prompts.append({"kind": "completion", "text": tmpl.format(t=topic)})
    prompts = prompts[:60]
    prompts += [{"kind": "completion", "text": t} for t in EXTRA_COMPLETIONS]
    for topic in TOPICS[:10]:
        for tmpl in CHAT_TEMPLATES:
            prompts.append(
                {"kind": "chat", "messages": [{"role": "user", "content": tmpl.format(t=topic)}]}
            )
    # The code-flavored template replaces a third of the generic ones for even-indexed topics.
    for j in range(0, 20, 2):
        prompts[j * 3 + 2] = {
            "kind": "completion",
            "text": COMPLETION_TEMPLATES[3].format(t=TOPICS[j]),
        }
    assert len(prompts) == 100, len(prompts)
    for idx, p in enumerate(prompts):
        p["id"] = idx
    return prompts


if __name__ == "__main__":
    out = Path(__file__).with_name("prompts.jsonl")
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in build()))
    print(f"wrote {out}")
