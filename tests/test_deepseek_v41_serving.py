"""DeepSeek-V4.1 on the CUDA OpenAI server: prompts by DeepSeek's V4.1 encoder (not Jinja), thinking and effort
controls, and its DSML calls read by DeepSeek's parser, whole and streamed at any chunk boundary."""
import json

import pytest

from tensorfold.families.deepseek_v4 import serving_v41
from tensorfold.families.deepseek_v4.prompts_v41 import render
from tensorfold.families.deepseek_v4.vendor import encoding_dsv41 as native
from tensorfold.server.errors import RequestError

CONTROLS = [native.SYSTEM_SP_TOKEN, native.USER_SP_TOKEN, native.ASSISTANT_SP_TOKEN, native.LATEST_REMINDER_SP_TOKEN,
            native.thinking_start_token, native.thinking_end_token, *native.DS_TASK_SP_TOKENS.values(),
            native.dsml_token]
TOOLS = [{"type": "function", "function": {"name": "weather", "description": "Forecast", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}},
         {"type": "function", "function": {"name": "files::read", "parameters": {
             "type": "object", "properties": {"path": {"type": "string"}}}}}]
CALL = {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris", "days": 3}'}}
BLOCK = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="weather">\n'
         '<｜DSML｜ parameter name="city" string="true">Paris "<b>"</｜DSML｜ parameter>\n'
         '<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n'
         '<｜DSML｜ invoke name="files::read">\n'
         '<｜DSML｜ parameter name="path" string="true">/tmp/a b.txt</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n'
         '</｜DSML｜ calls>')
EXPECTED = [("weather", {"city": 'Paris "<b>"', "days": 3}), ("files::read", {"path": "/tmp/a b.txt"})]


@pytest.fixture
def model_dir(tmp_path):
    pytest.importorskip("tokenizers")
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["weather Paris city days reasoning sure checking files read path tool call think"],
                            trainers.BpeTrainer(vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                special_tokens=["<unk>"]))
    tok.add_special_tokens([AddedToken(native.bos_token, special=True), AddedToken(native.eos_token, special=True)])
    tok.add_tokens([AddedToken(token, special=False) for token in CONTROLS])    # as the checkpoint marks them
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41", "eos_token_id": 1,
                                                      "text_config": {"max_position_embeddings": 65536}}))
    (tmp_path / "chat_template.jinja").write_text("JINJA MUST NOT RENDER {{ messages }}")
    return tmp_path


class Scripted:
    """An engine replying ``reply`` (text, then end of sentence) in rounds of ``step`` tokens."""

    def __init__(self, tok, reply="", step=1):
        self.tok, self.reply, self.step, self.prompts = tok, reply, step, []
        self.eos = (tok.token_to_id(native.eos_token),)

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts.append(list(prompt))
        ids = [*self.tok.encode(self.reply, add_special_tokens=False).ids, *self.eos][:max_tokens]
        for at in range(0, len(ids), self.step):
            if on_tokens(ids[at:at + self.step]):
                break
        return {}


def app_for(model_dir, reply="", step=1, **options):
    from tensorfold.cuda.server import App
    from tensorfold.token_metadata import load_tokenizer

    return App(Scripted(load_tokenizer(model_dir), reply, step), model_dir, "ds41", **options)


def prompt_text(app, body):
    ids = app.prepare_for_count({"max_tokens": 8, **body}, True).prompt
    return app.tok.decode(ids, skip_special_tokens=False)


def test_the_server_renders_v41_prompts_with_deepseeks_encoder(model_dir):
    from tensorfold.families.deepseek_v4.serving_v41 import V41ChatTemplate

    app = app_for(model_dir, default_thinking=True)            # the CLI's default (--thinking)
    assert isinstance(app.template, V41ChatTemplate)
    assert app.template.provenance()["source"].startswith("DeepSeek-V4.1 native Python encoder")
    chat = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    line = "Reasoning Effort: {} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    head = native.bos_token + native.SYSTEM_SP_TOKEN
    tail = "Be brief." + native.USER_SP_TOKEN + "Hi" + native.ASSISTANT_SP_TOKEN
    assert prompt_text(app, {"messages": chat}) == head + line.format(75) + tail + "<think>"
    # the friend's client: chat_template_kwargs {"thinking": bool, "reasoning_effort": 50}
    assert prompt_text(app, {"messages": chat, "chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}}) \
        == head + line.format(50) + tail + "<think>"
    assert prompt_text(app, {"messages": chat, "chat_template_kwargs": {"thinking": False, "reasoning_effort": 50}}) \
        == head + tail + "</think>"
    for effort, budget in (("low", 50), ("medium", 75), ("high", 75), ("xhigh", 100), ("max", 100), (1, 1),
                           (100, 100), ("minimal", 50)):
        assert prompt_text(app, {"messages": chat, "reasoning_effort": effort}) == \
            head + line.format(budget) + tail + "<think>"
    assert prompt_text(app, {"messages": chat, "reasoning_effort": "none"}) == head + tail + "</think>"
    assert prompt_text(app, {"messages": chat, "chat_template_kwargs": {"enable_thinking": False}}) == \
        head + tail + "</think>"
    for bad in (0, 101, True, "50", 2.5):
        with pytest.raises(RequestError, match="integer 1-100"):
            prompt_text(app, {"messages": chat, "chat_template_kwargs": {"reasoning_effort": bad}})
    assert prompt_text(app_for(model_dir), {"messages": chat}) == head + tail + "</think>"   # --no-thinking


@pytest.mark.parametrize("thinking", [False, True])
def test_tools_tool_history_and_late_system_render_as_the_encoder_does(model_dir, thinking):
    app = app_for(model_dir)
    messages = [{"role": "developer", "content": "Rules."}, {"role": "user", "content": "Weather <｜User｜>?"},
                {"role": "assistant", "content": None, "reasoning_content": "need a tool", "tool_calls": [CALL]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                {"role": "system", "content": "late note"}, {"role": "user", "content": "And tomorrow?"}]
    body = {"messages": messages, "tools": TOOLS, "chat_template_kwargs": {"thinking": thinking}}
    expected = render([{**messages[0], "role": "system"}, *messages[1:]], tools=TOOLS, thinking=thinking)
    assert prompt_text(app, body) == expected
    # tools join the existing first system message (DeepSeek's load_cases), not a separate one
    assert expected.startswith(native.bos_token + native.SYSTEM_SP_TOKEN + ("Reasoning Effort: 75" if thinking else
                                                                             "Rules.\n\n## Tools"))
    assert expected.count(native.SYSTEM_SP_TOKEN) == 2                       # the first system message and the late one
    # request text spelling a control token stays text: only the encoder's own controls are control ids
    ids = app.prepare_for_count(body, True).prompt
    assert ids.count(app.tok.token_to_id(native.USER_SP_TOKEN)) == expected.count(native.USER_SP_TOKEN) - 1


def test_images_and_template_overrides_are_refused(model_dir):
    from tensorfold.families.deepseek_v4 import cuda_chat_template

    app = app_for(model_dir)
    with pytest.raises(RequestError):
        prompt_text(app, {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})
    with pytest.raises(ValueError, match="drop --chat-template"):
        cuda_chat_template(model_dir, template_file=str(model_dir / "chat_template.jinja"))


def test_dsml_blocks_parse_as_deepseeks_parser_reads_them():
    assert serving_v41.decode_block(BLOCK) == EXPECTED
    # the encoder's own rendering of a call round trips
    text = render([{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [CALL]}],
                  tools=TOOLS[:1])
    block = text[text.rindex("<｜DSML｜ calls>"):text.rindex("</｜DSML｜ calls>") + len("</｜DSML｜ calls>")]
    assert serving_v41.decode_block(block) == [("weather", {"city": "Paris", "days": 3})]
    for bad in (BLOCK.replace('string="false">3', 'string="false">three'),          # not JSON
                BLOCK.replace('parameter>\n<｜DSML｜ parameter name="days"', 'parameter> <｜DSML｜ parameter name="days"'),  # framing
                BLOCK.replace('name="days"', 'name="city"'),                       # duplicate parameter
                BLOCK.replace('string="false">3', 'string="false">NaN'),
                '<｜DSML｜ calls>\n</｜DSML｜ calls>'):
        assert serving_v41.decode_block(bad) is None
    content, calls = serving_v41.parse("Checking.\n\n" + BLOCK, TOOLS)
    assert content == "Checking." and [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                                       for c in calls] == EXPECTED
    assert serving_v41.parse(BLOCK, TOOLS[:1]) == (BLOCK, None)                   # an unoffered tool stays text
    assert len(serving_v41.parse(BLOCK, TOOLS, max_calls=1)[1]) == 1


def reassemble(deltas):
    calls = []
    for delta in deltas:
        for part in delta.get("tool_calls", ()):
            if part["index"] == len(calls):
                calls.append({"name": part["function"]["name"], "arguments": ""})
            calls[part["index"]]["arguments"] += part["function"].get("arguments", "")
    return [(c["name"], json.loads(c["arguments"])) for c in calls]


@pytest.mark.parametrize("step", [1, 2, 3, 7, 1000])
def test_replies_split_reasoning_content_and_calls_streamed_or_not(model_dir, step):
    reply = "need the weather</think>Checking.\n\n" + BLOCK
    body = {"messages": [{"role": "user", "content": "Weather?"}], "tools": TOOLS, "max_tokens": 4096}
    finals = {}
    for stream in (False, True):
        app = app_for(model_dir, reply, step, default_thinking=True)
        deltas = []
        result = app.run({**body, "stream": stream}, True, lambda d: deltas.append(d) or True)
        assert app.engine.prompts[0][-1] == app.tok.token_to_id("<think>")
        streamed = [*deltas, result["final"]] if stream else []
        reasoning = "".join(d.get("reasoning_content", "") for d in streamed if d) if stream else result["reasoning"]
        content = "".join(d.get("content", "") for d in streamed if d) if stream else result["content"]
        assert reasoning == "need the weather" and content.strip() == "Checking."
        assert "DSML" not in content and result["finish"] == "tool_calls"
        finals[stream] = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in result["calls"]]
        if stream:
            assert reassemble(deltas) == EXPECTED and result["calls_streamed"] == 2
    assert finals[False] == finals[True] == EXPECTED


def test_a_cut_or_malformed_block_is_no_call(model_dir):
    body = {"messages": [{"role": "user", "content": "Weather?"}], "tools": TOOLS, "max_tokens": 4096, "stream": True}
    cut = BLOCK[:BLOCK.index("</｜DSML｜ invoke>")]
    app = app_for(model_dir, "Checking.\n\n" + cut, 3)
    app.engine.eos = ()                                                         # the reply runs out of tokens
    deltas = []
    result = app.run({**body, "max_tokens": len(app.tok.encode("Checking.\n\n" + cut).ids)}, True,
                     lambda d: deltas.append(d) or True)
    assert not result["calls"] and not reassemble(deltas) and result["finish"] == "length"
    bad = BLOCK.replace('string="false">3', 'string="false">three')
    app = app_for(model_dir, bad, 2)
    result = app.run(body, True, lambda d: True)
    assert not result["calls"] and result["finish"] == "stop" and "invoke" in result["content"]


def test_image_parts_expand_through_the_vision_frontend(model_dir, monkeypatch):
    """--vision: each image part renders as one trusted placeholder (request text spelling it is refused) and the
    frontend expands it to the image's span; the prompt-cache ids of a span follow the image's pixels."""
    pytest.importorskip("PIL")
    import base64
    import io

    from PIL import Image
    from tokenizers import AddedToken, Tokenizer

    from tensorfold.cuda.server import App
    from tensorfold.families.deepseek_v4.cuda import vision as V
    from tensorfold.token_metadata import load_tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.add_tokens([AddedToken(native.IMAGE_PLACEHOLDER, special=False)])
    tok.save(str(model_dir / "tokenizer.json"))
    image_id = tok.token_to_id(native.IMAGE_PLACEHOLDER)
    monkeypatch.setattr(V, "IMAGE_TOKEN_ID", image_id)
    engine = Scripted(load_tokenizer(model_dir))
    engine.vision = V.Frontend(model_dir)
    app = App(engine, model_dir, "ds41")

    def url(color):
        buf = io.BytesIO()
        Image.new("RGB", (600, 400), color).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def prepared(*colors, text="What is it?"):
        parts = [{"type": "text", "text": text}, *({"type": "image_url", "image_url": {"url": url(c)}} for c in colors)]
        return app.prepare_for_count({"max_tokens": 8, "messages": [{"role": "user", "content": parts}]}, True)

    red = prepared("red")
    vision = red.vision
    n_llm_h, n_llm_w, _, _ = V.plan_image_grid(600, 400)
    span = V.num_image_tokens(n_llm_h, n_llm_w)
    assert len(vision.blocks) == 1 and vision.blocks[0][1] - vision.blocks[0][0] == span
    assert list(red.prompt) == list(vision.token_ids) and red.prompt.count(image_id) == span
    a, b = vision.blocks[0]
    text = app.tok.decode(list(red.prompt[:a]) + [image_id] + list(red.prompt[b:]), skip_special_tokens=False)
    assert text == (native.bos_token + native.USER_SP_TOKEN + "What is it?\n\n" + native.IMAGE_PLACEHOLDER +
                    native.ASSISTANT_SP_TOKEN + native.thinking_end_token)
    with pytest.raises(RequestError):
        prepared("red", text="What is " + native.IMAGE_PLACEHOLDER + "?")
    vocab = 1000
    keys = V.prefix_ids(vision, vocab)
    assert keys[:a] == list(red.prompt[:a]) and keys[b:] == list(red.prompt[b:])
    assert all(vocab <= k < 1 << 31 for k in keys[a:b]) and len(set(keys[a:b])) == 8
    assert keys == V.prefix_ids(prepared("red").vision, vocab)
    assert keys[a:b] != V.prefix_ids(prepared("blue").vision, vocab)[a:b]
    two = prepared("red", "blue").vision
    assert len(two.blocks) == 2 and two.token_ids.count(image_id) == 2 * span
