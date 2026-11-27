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
                                                      "vocab_size": tok.get_vocab_size(),
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
        ids = self.tok.encode(self.reply, add_special_tokens=False).ids if isinstance(self.reply, str) else self.reply
        ids = [*ids, *self.eos][:max_tokens]
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


def test_native_segment_cache_preserves_tokens_and_request_origin(model_dir):
    from copy import deepcopy
    from importlib.metadata import version
    from tokenizers import AddedToken, normalizers, processors
    from tensorfold.server.prompt_literals import RenderedPrompt
    from tensorfold.server.prompt_origin import OriginText, trusted
    from tensorfold.token_metadata import load_tokenizer

    tok = load_tokenizer(model_dir)
    tok.normalizer = normalizers.Sequence([])
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    path = str(model_dir / "tokenizer.json")
    tok.save(path)
    app = app_for(model_dir, default_thinking=True)
    if version("tokenizers") != "0.23.3":
        assert app.segments is None
        return

    opener = native.USER_SP_TOKEN
    assert app.segments is not None and app.segments.opener == opener
    prompt = f" \t日本語é{opener}{opener}🙂\n"
    protection = app.template.literals
    expected = []
    for value in (trusted(prompt), OriginText(prompt), trusted(prompt)):
        text = RenderedPrompt(value, protection.literals)
        whole = app.template.encode(app.tok, text)
        assert app.template.encode(app.tok, text, segments=app.segments) == whole
        expected.append(whole)
    assert expected[0] == expected[2] != expected[1]

    body = {"messages": [{"role": "system", "content": "Read the excerpt."},
                         {"role": "user", "content": f"Literal text {opener} and a line break.\n"}],
            "tools": strict_tools(), "max_tokens": 8}
    prepared = app.prepare_for_count(body, True)
    try:
        text = app.template.render(body["messages"], tools=body["tools"], enable_thinking=True)
        assert prepared.prompt == app.template.encode(app.tok, text)
    finally:
        prepared.close()

    for crossing in (AddedToken("User", normalized=False), AddedToken(opener + "tail", normalized=True)):
        alternate = deepcopy(tok)
        alternate.add_tokens([crossing])
        alternate.save(path)
        assert app_for(model_dir).segments is None

    alternate = deepcopy(tok)
    alternate.add_special_tokens([AddedToken(opener, special=True, normalized=False)])
    alternate.save(path)
    raw = app_for(model_dir)
    assert raw.segments is not None and raw.segments.opener == opener

    tok.model.dropout = 0.25
    tok.save(path)
    assert app_for(model_dir).segments is None


def test_native_cuda_tokenize_exposes_exact_count_and_deepseek_render_metadata(model_dir):
    from http.client import HTTPConnection
    from threading import Thread

    from tensorfold.cuda.http import make_handler
    from tensorfold.server.http import Server

    app = app_for(model_dir, default_thinking=True, aliases=("alias",))
    body = {"model": "alias", "messages": [{"role": "user", "content": "Paris weather?"}],
            "chat_template_kwargs": {"enable_thinking": False}}
    prepared = app.prepare_for_count(body, True)
    try:
        result = app.tokenize({**body, "return_prompt": True})
        assert (result["count"], result["tokens"]) == (len(prepared.prompt), prepared.prompt)
        assert result["prompt"] == app.tok.decode(prepared.prompt, skip_special_tokens=False)
    finally:
        prepared.close()
    assert result["model"] == "alias"
    assert result["context_length"] == result["max_model_len"] == 65536
    assert result["harness_tokenizer_version"] == 1
    assert result["chat_template_sha256"] == app.template.provenance()["sha256"]
    assert result["capabilities"]["thinking_control"]["keyword"] == "enable_thinking"
    assert result["capabilities"]["reasoning_replay_field"] == "reasoning_content"
    assert result["capabilities"]["preserve_reasoning"] is True
    count = app.tokenize({**body, "return_tokens": False})
    assert count["count"] == result["count"] and "tokens" not in count and "prompt" not in count
    raw = app.tokenize({"prompt": "Paris"})
    assert set(raw) == {"count", "tokens", "max_model_len"}
    for flag in ("return_tokens", "return_prompt"):
        with pytest.raises(RequestError, match=f"{flag} must be a boolean"):
            app.tokenize({**body, flag: 1})
    with Server(("127.0.0.1", 0), make_handler(app)) as listener:
        server = Thread(target=listener.serve_forever)
        server.start()
        client = HTTPConnection(*listener.server_address, timeout=2)
        try:
            payload = json.dumps({**body, "return_tokens": False})
            client.request("POST", "/v1/tokenize", payload, {"Content-Type": "application/json"})
            response = client.getresponse()
            assert response.status == 200 and json.load(response) == count
        finally:
            client.close()
            listener.shutdown()
            server.join()
    assert not app.engine.prompts


def test_v41_sampling_defaults_apply_to_plain_tools_and_raw_requests(model_dir):
    from tensorfold.engine.exact_sampling import Sampling

    app = app_for(model_dir)
    body = {"messages": [{"role": "user", "content": "Paris weather?"}], "seed": 7}
    for fields in ({}, {"tools": []}, {"tools": TOOLS}, {"tools": TOOLS, "tool_choice": "none"},
                   {"tools": TOOLS, "temperature": None, "top_p": None}):
        prepared = app.prepare_for_count({**body, **fields}, True)
        try:
            assert prepared.sampling == Sampling(7, 1.0, 0, 1.0)
        finally:
            prepared.close()
    prepared = app.prepare_for_count({"prompt": "Paris weather?", "seed": 7}, False)
    try:
        assert prepared.sampling == Sampling(7, 1.0, 0, 1.0)
    finally:
        prepared.close()


def test_v41_sampling_preserves_checkpoint_launch_and_request_priority(model_dir):
    from tensorfold.cli import _sampling_defaults, build_parser
    from tensorfold.engine.exact_sampling import Sampling

    body = {"messages": [{"role": "user", "content": "Paris weather?"}], "tools": TOOLS, "seed": 7}

    def configured(config, *flags):
        (model_dir / "generation_config.json").write_text(json.dumps(config))
        args = build_parser().parse_args(["serve", str(model_dir), *flags])
        return app_for(model_dir, sampling=_sampling_defaults(args, model_dir))

    def sampling(app, **overrides):
        prepared = app.prepare_for_count({**body, **overrides}, True)
        try:
            return prepared.sampling
        finally:
            prepared.close()

    assert sampling(configured({"temperature": 0.7})) == Sampling(7, 0.7, 0, 1.0)
    assert sampling(configured({"top_p": 0.82})) == Sampling(7, 1.0, 0, 0.82)
    assert sampling(configured({"do_sample": False, "temperature": 1.0, "top_p": 0.82})) is None
    launch = configured({"do_sample": False, "top_p": 0.82, "top_k": 17},
                        "--temperature", "0.6", "--top-p", "0.89")
    assert sampling(launch) == Sampling(7, 0.6, 17, 0.89)
    assert sampling(launch, temperature=0.3, top_p=1.0, top_k=0) == Sampling(7, 0.3, 0, 1.0)
    assert sampling(launch, temperature=0) is None
    assert sampling(launch, top_p=0) is None


def test_v41_replay_metadata_retains_both_routes_without_changing_official_prompts(model_dir):
    app = app_for(model_dir, default_thinking=True)
    older, current = "reasoning from the first turn", "reasoning for the current tool call"
    messages = [
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First answer.", "reasoning_content": older},
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [CALL], "reasoning_content": current},
        {"role": "tool", "content": "sunny", "tool_call_id": "c1"},
    ]
    for advertised in (False, True):
        body = {"messages": messages, "return_prompt": True, "return_tokens": False}
        reference = messages
        if advertised:
            body["tools"] = TOOLS
            reference = [{"role": "system", "content": "", "tools": TOOLS}, *messages]
        expected = native.encode_messages(reference, thinking_mode="thinking")
        results = app.tokenize(body)
        assert results["prompt"] == expected
        assert current in expected and (older in expected) is advertised
        assert results["capabilities"]["reasoning_replay_field"] == "reasoning_content"
        assert results["capabilities"]["preserve_reasoning"] is True
    assert not app.engine.prompts


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


def test_multipart_tool_results_match_official_prompt_and_do_not_mutate_history(model_dir):
    from copy import deepcopy

    app = app_for(model_dir, default_thinking=True)
    second = {**CALL, "id": "c2"}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}]},
        {"role": "assistant", "content": "", "tool_calls": [CALL, second], "reasoning_content": "Check both."},
        {"role": "tool", "tool_call_id": "c2", "content": "Gamma"},
        {"role": "tool", "tool_call_id": "c1", "content": [
            {"type": "text", "text": "Alpha"}, {"type": "text", "text": "Beta"}]},
    ]
    original = deepcopy(messages)
    expected = native.encode_messages([{"role": "system", "content": "", "tools": TOOLS}, *messages],
                                      thinking_mode="thinking")
    assert "First\n\nSecond" in expected
    assert "<tool_result>Alpha\n\nBeta</tool_result>\n\n<tool_result>Gamma</tool_result>" in expected
    assert prompt_text(app, {"messages": messages, "tools": TOOLS}) == expected
    assert messages == original


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("prefix", ['Unclosed quote: ".', "Unclosed fence:\n```text\nexample"])
@pytest.mark.parametrize("step", [1, 19, 1000])
def test_generated_dsml_controls_survive_unmatched_prose(model_dir, thinking, prefix, step):
    reply = prefix + ("</think>" if thinking else "") + "\n\n" + BLOCK
    official = native.parse_message_from_completion_text(reply + native.eos_token, "thinking" if thinking else "chat")
    assert len(official["tool_calls"]) == len(EXPECTED)
    body = {"messages": [{"role": "user", "content": "Weather?"}], "tools": TOOLS, "max_tokens": 4096}
    for stream in (False, True):
        app = app_for(model_dir, reply, step, default_thinking=thinking)
        deltas = []
        result = app.run({**body, "stream": stream}, True, lambda d: deltas.append(d) or True)
        assert result["finish"] == "tool_calls"
        assert result["reasoning"] == official["reasoning_content"]
        assert result["content"] == official["content"]
        if stream:
            assert reassemble(deltas) == EXPECTED
            actual = [*deltas, result["final"]]
            assert "".join(d.get("reasoning_content", "") for d in actual if d) == official["reasoning_content"]
            assert "".join(d.get("content", "") for d in actual if d).strip() == official["content"]


@pytest.mark.parametrize("stream", [False, True])
def test_plain_token_spelling_cannot_become_thinking_or_dsml_control(model_dir, stream):
    from tensorfold.server.prompt_literals import RenderedPrompt
    from tensorfold.server.prompt_origin import OriginText

    app = app_for(model_dir, step=1, default_thinking=True)
    literal = 'The text says </think>, and this is literal:\n\n' + BLOCK
    ids = app.template.encode(app.tok, RenderedPrompt(OriginText(literal), app.template.literals.literals))
    assert app.tok.decode(ids, skip_special_tokens=False) == literal
    assert not set(ids).intersection(app.tok.token_to_id(text) for text in ("</think>", native.dsml_token))
    app.engine.reply = [*ids, *app.tok.encode("</think>Finished.", add_special_tokens=False).ids]
    deltas = []
    result = app.run({"messages": [{"role": "user", "content": "Show literal text."}], "tools": TOOLS,
                      "max_tokens": 4096, "stream": stream}, True, lambda d: deltas.append(d) or True)
    assert (result["reasoning"], result["content"], result["calls"], result["finish"]) == (literal, "Finished.", None, "stop")
    assert not reassemble(deltas)


@pytest.mark.parametrize("explicit", ["tool", "function"])
def test_explicit_namespaces_keep_offered_identity_and_replay(model_dir, explicit):
    from copy import deepcopy

    tool = {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
    (tool if explicit == "tool" else tool["function"])["namespace"] = "files"
    block = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="files::read">\n\n'
             '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    reference = native.parse_message_from_completion_text("\n\n" + block + native.eos_token, "chat")
    [expected] = reference["tool_calls"]
    body = {"messages": [{"role": "user", "content": "Read."}], "tools": [tool], "max_tokens": 4096}
    for stream in (False, True):
        app = app_for(model_dir, block, step=1)
        counted = app.prepare_for_count({**body, "tool_choice": {
            "type": "function", "function": {"name": "files::read"}}}, True)
        try:
            assert counted.tools == [tool]
        finally:
            counted.close()
        deltas = []
        result = app.run({**body, "stream": stream}, True, lambda d: deltas.append(d) or True)
        [call] = result["calls"]
        assert call["namespace"] == expected["namespace"] and call["function"]["name"] == expected["function"]["name"]
        assert json.loads(call["function"]["arguments"]) == json.loads(expected["function"]["arguments"])
        if stream:
            [head] = [part for delta in deltas if delta for part in delta.get("tool_calls", ()) if "id" in part]
            assert (head["namespace"], head["function"]["name"]) == ("files", "read")
        history = [*body["messages"], {"role": "assistant", "content": "", "tool_calls": [call]},
                   {"role": "tool", "tool_call_id": call["id"], "content": "Found."}]
        assert block in prompt_text(app, {**body, "messages": history})
    wrong = deepcopy(tool)
    (wrong if explicit == "tool" else wrong["function"])["namespace"] = "other"
    assert serving_v41.parse(block, [wrong]) == (block, None)


def test_strict_tool_plan_compiles_and_reads_qualified_invocations():
    from tensorfold.families.deepseek_v4.strict_tools_v41 import Plan

    tool = {"type": "function", "namespace": {"name": "files"}, "function": {"name": "read", "strict": True,
            "parameters": {"type": "object", "additionalProperties": False}}}
    plan = Plan([tool], required=True, parallel=False)
    assert any('name="files::read"' in head for head in plan.thinking_headers)
    block = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="files::read">\n\n'
             '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    _, calls, complete = plan.parse_finished(block)
    assert complete and len(calls) == 1
    assert calls[0]["namespace"] == "files" and calls[0]["function"]["name"] == "read"
    [head] = [part for delta in plan.reader().feed(block) for part in delta["tool_calls"] if "id" in part]
    assert (head["namespace"], head["function"]["name"]) == ("files", "read")


def test_plain_dsml_spelling_is_visible_and_never_calls_tools(model_dir):
    from tensorfold.server.prompt_literals import RenderedPrompt
    from tensorfold.server.prompt_origin import OriginText

    app = app_for(model_dir, step=1)
    literal = "Here is the notation.\n\n" + BLOCK
    app.engine.reply = app.template.encode(app.tok, RenderedPrompt(OriginText(literal), app.template.literals.literals))
    assert app.tok.token_to_id(native.dsml_token) not in app.engine.reply
    streamed = []
    result = app.run({"messages": [{"role": "user", "content": "Show literal text."}], "tools": TOOLS,
                      "max_tokens": 4096, "stream": True}, True, lambda d: streamed.append(d) or True)
    assert (result["content"], result["calls"], result["finish"]) == (literal, None, "stop")
    assert "".join(d.get("content", "") for d in [*streamed, result["final"]] if d) == literal
    assert not reassemble(streamed)


def test_ordinary_spelling_of_dsml_closer_cannot_complete_a_call(model_dir):
    from tensorfold.server.prompt_literals import RenderedPrompt
    from tensorfold.server.prompt_origin import OriginText

    app = app_for(model_dir, step=1)
    head = app.tok.encode(BLOCK[:-len(serving_v41.CLOSER)], add_special_tokens=False).ids
    close = app.template.encode(app.tok, RenderedPrompt(OriginText(serving_v41.CLOSER), app.template.literals.literals))
    assert app.tok.token_to_id(native.dsml_token) in head and app.tok.token_to_id(native.dsml_token) not in close
    app.engine.reply = [*head, *close]
    seen = []
    result = app.run({"messages": [{"role": "user", "content": "Check the closer."}], "tools": TOOLS,
                      "max_tokens": 4096, "stream": True}, True, lambda d: seen.append(d) or True)
    assert not result["calls"] and result["finish"] == "stop"
    assert not reassemble(seen)

def strict_tools():
    weather, files = TOOLS
    function = weather["function"]
    parameters = {**function["parameters"], "required": ["city", "days"], "additionalProperties": False}
    return [{**weather, "function": {**function, "strict": True, "parameters": parameters}}, files]


def test_strict_dsml_grammar_keeps_parameter_types_and_call_choice():
    pytest.importorskip("xgrammar")
    from xgrammar import Grammar
    from xgrammar.testing import _is_grammar_accept_string

    from tensorfold.families.deepseek_v4.strict_tools_v41 import Plan

    def accepts(plan, text):
        return _is_grammar_accept_string(Grammar.from_structural_tag(plan.spec.text), text)

    tools = strict_tools()
    block = BLOCK[:BLOCK.index('<｜DSML｜ invoke name="files::read">')] + serving_v41.CLOSER
    plan = Plan(tools, required=True, parallel=False)
    assert accepts(plan, block)
    assert not accepts(plan, BLOCK)
    assert not accepts(plan, "A normal answer")
    assert not accepts(plan, block.replace('string="false">3', 'string="true">3'))
    assert not accepts(plan, block.replace('name="days"', 'name="unknown"'))
    assert not accepts(plan, block.replace('name="weather"', 'name="unoffered"'))
    assert not accepts(plan, "\n".join(line for line in block.splitlines() if 'name="days"' not in line))
    auto = Plan(tools, required=False, parallel=True)
    assert accepts(auto, "A normal answer")
    assert accepts(auto, "Checking.\n\n" + BLOCK)
    assert not accepts(Plan(tools, required=False, parallel=False), BLOCK + "\n" + block)


def test_strict_dsml_reader_preserves_json_source_and_holds_a_partial_block():
    pytest.importorskip("xgrammar")
    from xgrammar import Grammar
    from xgrammar.testing import _is_grammar_accept_string

    from tensorfold.families.deepseek_v4.strict_tools_v41 import Plan

    tools = strict_tools()
    loose = tools[1]["function"]["name"]
    value = '{"number":1.000000000000000000001,"huge":1e309,"marker":"</｜DSML｜ parameter>"}'
    block = BLOCK.replace('string="true">/tmp/a b.txt', 'string="false">' + value)
    plan = Plan(tools, required=False, parallel=True)
    assert _is_grammar_accept_string(Grammar.from_structural_tag(plan.spec.text), block)
    reader = plan.reader()
    prefix = block.removesuffix(serving_v41.CLOSER)
    for end in range(len(prefix) + 1):
        assert not reader.feed(prefix[:end])
    assert plan.parse_finished(prefix) == (prefix.strip(), None, False)
    deltas = reader.feed(block)
    assert [d["tool_calls"][0]["index"] for d in deltas] == [0, 0, 1, 1]
    assert reader.index == 1 and reader.streamed and reader.feed(block) == []
    content, calls, complete = plan.parse_finished(block)
    assert complete and content == ""
    assert calls[1]["function"] == {"name": loose, "arguments": '{"path":' + value + "}"}
    assert deltas[-1]["tool_calls"][0]["function"]["arguments"] == calls[1]["function"]["arguments"]


@pytest.mark.parametrize("stream", [False, True])
def test_strict_dsml_token_count_and_completion_use_same_native_prompt(model_dir, stream):
    pytest.importorskip("xgrammar")
    from http.client import HTTPConnection

    from tests.test_cuda_admission import http_server, post
    from tests.test_strict_tools import _reply

    class ConstrainedScripted(Scripted):
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
            assert constraint.spec.kind == "tools"

            def send(tokens):
                constraint.advance(tokens)
                return on_tokens(tokens)

            return super().generate(prompt, max_tokens, sampling, send, draft)

    app = app_for(model_dir, default_thinking=True)
    app.engine = ConstrainedScripted(app.tok, "check</think>Checking.\n\n" + BLOCK, step=3)
    tools = strict_tools()
    body = {"model": "ds41", "messages": [{"role": "user", "content": "Weather?"}],
            "tools": tools, "stream": stream, "max_tokens": 1024}
    with http_server(app) as port:
        conn = HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            conn.request("POST", "/v1/tokenize", json.dumps(body), {"Content-Type": "application/json"})
            reply = conn.getresponse()
            counted = json.load(reply)
            assert reply.status == 200, counted
            assert not app.engine.prompts
        finally:
            conn.close()
        status, text = post(port, body, True)
    assert status == 200, text
    assert counted["count"] == len(app.engine.prompts[0])
    content, calls, finish = _reply(stream, text)
    assert content.strip() == "Checking." and finish == "tool_calls"
    assert [(name, json.loads(args)) for name, args in calls] == EXPECTED


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
