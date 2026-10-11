"""Shared CPU-only checkpoint files for real CUDA HTTP preparation/admission tests."""
import json

from tokenizers import Tokenizer, models, pre_tokenizers


def model_directory(directory):
    words = ["[UNK]", "answer", "header", "thinking", "tools", "extra"] + [f"w{i}" for i in range(40)]
    tok = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(directory / "tokenizer.json"))
    template = ("{% for m in messages %}{{ m.content }} {% endfor %}header"
                "{% if enable_thinking %} thinking{% endif %}{% if tools %} tools{% endif %}"
                "{% if suffix %} {{ suffix }}{% endif %}")
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    return directory
