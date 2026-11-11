from __future__ import annotations

import json
from pathlib import Path

import pytest

from dli.server.text import (
    ChatMessage,
    ChatTemplateError,
    IncrementalDetokenizer,
    StopStringMatcher,
    TextCodec,
)
from tests.helpers.tiny import write_byte_bpe_tokenizer, write_char_tokenizer


@pytest.fixture
def char_codec(tmp_path: Path) -> TextCodec:
    write_char_tokenizer(tmp_path)
    return TextCodec(tmp_path)


def test_char_codec_round_trip_and_eos(char_codec: TextCodec) -> None:
    ids = char_codec.encode("Hi there!")
    assert len(ids) == 9
    assert char_codec.decode(ids) == "Hi there!"
    assert char_codec.eos_token_ids == {2}


def test_chat_template_renders_in_sandbox(char_codec: TextCodec) -> None:
    ids = char_codec.render_chat([ChatMessage("user", "ping")])
    assert char_codec.decode(ids) == "<user>ping<assistant>"


def test_hostile_chat_template_cannot_reach_python_internals(tmp_path: Path) -> None:
    write_char_tokenizer(tmp_path)
    cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
    # Classic Jinja escape: walk from a string to object subclasses. The sandbox must refuse it.
    cfg["chat_template"] = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ChatTemplateError, match="chat template failed"):
        TextCodec(tmp_path).render_chat([ChatMessage("user", "x")])


def test_template_raise_exception_is_a_client_error(tmp_path: Path) -> None:
    write_char_tokenizer(tmp_path)
    cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
    cfg["chat_template"] = "{{ raise_exception('roles must alternate') }}"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ChatTemplateError, match="roles must alternate"):
        TextCodec(tmp_path).render_chat([ChatMessage("user", "x")])


def test_incremental_detokenizer_never_emits_broken_utf8(tmp_path: Path) -> None:
    write_byte_bpe_tokenizer(tmp_path)
    codec = TextCodec(tmp_path)
    text = "hello 🙂 café ñandú 日本"
    ids = codec.encode(text)
    assert len(ids) > len(text.encode()) // 4, "expected multi-byte chars split across tokens"
    detok = IncrementalDetokenizer(codec, [])
    pieces = [detok.push(t) for t in ids]
    assert all("�" not in p for p in pieces)
    assert "".join(pieces) == text


def test_incremental_detokenizer_continues_after_prompt(char_codec: TextCodec) -> None:
    prompt = char_codec.encode("abc")
    detok = IncrementalDetokenizer(char_codec, prompt)
    assert "".join(detok.push(t) for t in char_codec.encode("def")) == "def"


@pytest.mark.parametrize(
    ("deltas", "stops", "expected", "stopped"),
    [
        (["Hello", " wor", "ld"], ["world"], "Hello ", True),  # spans deltas
        (["abc", "END", "zzz"], ["END"], "abc", True),
        (["ab", "c"], ["xyz"], "abc", False),  # flush releases held text
        (["The ans", "wer is 4", "2.\n\nQ:"], ["\n\nQ:", "###"], "The answer is 42.", True),
        (["#", "#", "x"], ["###"], "##x", False),  # partial match abandoned
    ],
)
def test_stop_string_matcher(
    deltas: list[str], stops: list[str], expected: str, stopped: bool
) -> None:
    m = StopStringMatcher(stops)
    out = []
    for d in deltas:
        safe, hit = m.feed(d)
        out.append(safe)
        if hit:
            break
    out.append(m.flush())
    assert "".join(out) == expected
    assert m.stopped is stopped


def test_stop_matcher_holds_back_possible_prefix() -> None:
    m = StopStringMatcher(["STOP"])
    assert m.feed("abc ST") == ("abc ", False)  # "ST" might become STOP: hold it
    assert m.feed("ART") == ("START", False)  # it did not: release it
