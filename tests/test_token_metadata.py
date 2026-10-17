"""Reply EOS comes from explicit roles, with stable ordering and validated IDs."""
import json
import subprocess
import sys

import pytest
from tokenizers import Tokenizer, models

from tensorfold.token_metadata import resolve_eos


def write(directory, name, value):
    (directory / name).write_text(json.dumps(value))


def tokenizer(directory, *, eos="turn_stop"):
    vocab = {"[UNK]": 0, "model_stop": 1, "turn_stop": 2, "other_special": 3}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.add_special_tokens(["turn_stop", "other_special"])
    tok.save(str(directory / "tokenizer.json"))
    write(directory, "tokenizer_config.json", {"eos_token": {"content": eos, "special": True},
                                               "added_tokens_decoder": {"2": {"content": "turn_stop"}}})
    return tok


def test_model_generation_and_tokenizer_named_eos_are_stable_and_do_not_stop_on_other_specials(tmp_path):
    tok = tokenizer(tmp_path)
    write(tmp_path, "generation_config.json", {"eos_token_id": [2, 1]})
    assert resolve_eos(tmp_path, 1, vocab_size=4) == (1, tok.token_to_id("turn_stop"))
    assert tok.token_to_id("other_special") not in resolve_eos(tmp_path, [1, 1], vocab_size=4)


def test_tokenizer_role_corrects_export_that_repeats_only_the_model_end(tmp_path):
    tokenizer(tmp_path)
    write(tmp_path, "generation_config.json", {"eos_token_id": 1})
    assert resolve_eos(tmp_path, 1, vocab_size=4) == (1, 2)


def test_explicit_replacement_does_not_union_model_or_generation_ids(tmp_path):
    tokenizer(tmp_path)
    write(tmp_path, "generation_config.json", {"eos_token_id": [1, 2]})
    assert resolve_eos(tmp_path, 1, vocab_size=4, override=[3, 3, 0]) == (3, 0)


@pytest.mark.parametrize("value", [[], (), True, -1, 4, 1.0, "2", [1, False], [1, -1]])
def test_invalid_or_empty_explicit_override_is_not_omission(tmp_path, value):
    with pytest.raises(ValueError, match="eos_token_ids override"):
        resolve_eos(tmp_path, 1, vocab_size=4, override=value)


@pytest.mark.parametrize("value", [True, -1, 4, 1.0, "2", [1, False]])
def test_invalid_declared_ids_refuse_without_coercion(tmp_path, value):
    write(tmp_path, "generation_config.json", {"eos_token_id": value})
    with pytest.raises(ValueError, match="generation eos_token_id"):
        resolve_eos(tmp_path, 1, vocab_size=4)


def test_added_token_declaration_must_agree_with_actual_tokenizer_ids(tmp_path):
    tokenizer(tmp_path)
    write(tmp_path, "tokenizer_config.json", {"eos_token": "turn_stop",
                                               "added_tokens_decoder": {"3": {"content": "turn_stop"}}})
    with pytest.raises(ValueError, match="conflicting token IDs"):
        resolve_eos(tmp_path, 1, vocab_size=4)


def test_missing_eos_token_never_becomes_unknown_token_id(tmp_path):
    tokenizer(tmp_path, eos="not_a_token")
    with pytest.raises(ValueError, match="absent from tokenizer.json"):
        resolve_eos(tmp_path, 1, vocab_size=4)


@pytest.mark.parametrize("value", [{}, {"content": None}, {"content": ""}, {"content": 2}])
def test_present_eos_added_token_requires_nonempty_string_content(tmp_path, value):
    write(tmp_path, "tokenizer_config.json", {"eos_token": value})
    with pytest.raises(ValueError, match="tokenizer eos_token"):
        resolve_eos(tmp_path, 1, vocab_size=4)
    assert resolve_eos(tmp_path, 1, vocab_size=4, override=[3]) == (3,)


@pytest.mark.parametrize("value", [None, {}, {"content": None}, {"content": ""}, "", 2])
def test_present_added_token_decoder_entries_require_content(tmp_path, value):
    write(tmp_path, "tokenizer_config.json", {"eos_token": "end", "added_tokens_decoder": {"2": value}})
    with pytest.raises(ValueError, match="added_tokens_decoder content"):
        resolve_eos(tmp_path, 1, vocab_size=4)


def test_null_eos_role_is_distinct_from_malformed_added_token_object(tmp_path):
    write(tmp_path, "tokenizer_config.json", {"eos_token": None})
    assert resolve_eos(tmp_path, 1, vocab_size=4) == (1,)


def test_modern_decoder_only_metadata_resolves_its_named_eos(tmp_path):
    write(tmp_path, "tokenizer_config.json", {"eos_token": "end", "added_tokens_decoder": {"2": {"content": "end"}}})
    assert resolve_eos(tmp_path, 1, vocab_size=4) == (1, 2)


def test_unigram_vocabulary_uses_exact_entry_id(tmp_path):
    write(tmp_path, "tokenizer_config.json", {"eos_token": "end"})
    write(tmp_path, "tokenizer.json", {"model": {"vocab": [["[UNK]", 0.], ["end", -1.]]}})
    assert resolve_eos(tmp_path, vocab_size=2) == (1,)


def test_family_default_applies_only_without_declared_eos(tmp_path):
    assert resolve_eos(tmp_path, vocab_size=4, fallback=2) == (2,)
    write(tmp_path, "generation_config.json", {"eos_token_id": 3})
    assert resolve_eos(tmp_path, vocab_size=4, fallback=2) == (3,)


def test_metadata_resolvers_import_without_accelerator_runtime():
    result = subprocess.run([sys.executable, "-c", ("import sys; import tensorfold.token_metadata, tensorfold.chat_metadata; "
                             "assert not any(m in sys.modules for m in ('torch','triton','mlx'))")],
                            check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
