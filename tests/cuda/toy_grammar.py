"""Toy grammars for the CUDA engines' tests: token t spells t in base 22 over a JSON-like alphabet (``toy``), or in base
52 over strict tool-call markup's characters (``toy_tools``); 0 ends a reply."""

ALPHA = "abcdefgh{}:,0123456789"
RULE = 'root ::= "{" ([a-h]+ ":" [0-9]+ ",")* [a-h]+ ":" [0-9]+ "}"'


def spell(t: int) -> str:
    out = ""
    while t:
        t, d = divmod(t, len(ALPHA))
        out = ALPHA[d] + out
    return out


def toy(vocab: int):
    """(``Grammars`` over ``vocab`` tokens, the compiled rule); every token's spelling is its own."""

    import xgrammar as xgr

    from tensorfold.engine import grammar

    words = [""] + [spell(t) for t in range(1, vocab)]
    grammars = grammar.Grammars(xgr.TokenizerInfo(words, xgr.VocabType.RAW, vocab_size=vocab, stop_token_ids=[0]))
    return grammars, grammars.compile(grammar.Spec("grammar", RULE))


TOOL_ALPHA = "~abcdefghijklmnopqrstuvwxyz0123456789<>/=_\n{}[]\":, "   # "~" (digit 0) never leads a spelling
TOOL = {"type": "object", "additionalProperties": False, "required": ["n", "u", "o"],
        "properties": {"n": {"type": "integer", "minimum": 0, "maximum": 99},
                       "u": {"type": "string", "enum": ["c", "ff"]},
                       "o": {"type": "object", "additionalProperties": False, "required": ["k"],
                             "properties": {"k": {"type": "boolean"}}},
                       "z": {"type": "boolean"}}}
TOOLS = [{"type": "function", "function": {"name": "get", "parameters": TOOL, "strict": True}},
         {"type": "function", "function": {"name": "go", "strict": True}}]

# xgrammar 0.2.8's emitted required grammar bounds the first get call by leading
# blanks, markup, four parameter wrappers, finite values and closing markup.
# The object has four whitespace runs of at most32 characters. Every non-EOS
# toy token consumes at least one character. Later parallel calls and trailing
# blanks occur after this first complete call.
TOOL_FIRST_CALL_TOKENS = 32 + 27 + 4 * 28 + 2 + 2 + (11 + 4 * 32) + 5 + 24


def tool_spell(t: int) -> str:
    out = ""
    while t:
        t, d = divmod(t, len(TOOL_ALPHA))
        out = TOOL_ALPHA[d] + out
    return out


def toy_tools(vocab: int, *, required: bool = False, parallel: bool = True):
    """(``Grammars`` over ``vocab`` tokens, a strict-tools ``Plan`` for ``TOOLS``, its compiled grammar): token t spells
    t in base 52 over Qwen XML call markup's characters, 0 ends a reply."""

    import xgrammar as xgr

    from tensorfold.engine import grammar
    from tensorfold.server.strict_tools import Plan

    words = [""] + [tool_spell(t) for t in range(1, vocab)]
    grammars = grammar.Grammars(xgr.TokenizerInfo(words, xgr.VocabType.RAW, vocab_size=vocab, stop_token_ids=[0]))
    plan = Plan(TOOLS, required=required, parallel=parallel)
    return grammars, plan, grammars.compile(plan.spec)
