"""JSON container nesting at untrusted parse boundaries.

The selected CPython (3.15) checks only the C stack pointer while its json C
scanner recurses, so input may nest as deep as the thread stack allows. The
host payload projections pay for at most NESTING_LIMIT open containers on the
boundaries that parse through ``loads``; ``loads`` therefore refuses deeper
input before the C scanner runs. The limit and the refusal are the contract
CPython 3.12 enforced with its compiled C recursion counter
(Py_C_RECURSION_LIMIT, 10000): a RecursionError with 3.12's message, and a
syntax error the scanner meets first keeps precedence.

The guard bounds what the projections pay. It is not 3.12's exact threshold,
which also counted the C-entered frames already on the caller's stack (3.12.3
measured 9997 accepted levels for a direct call), so an input nested within
those few levels of 10000 parses here.
"""

import json
import re

NESTING_LIMIT = 10000

# The rest of a JSON string after its opening quote, including the closing
# quote; an unterminated string runs to the end of the input.
_STRING_TAIL = re.compile(r'[^"\\]*+(?:\\.[^"\\]*+)*+"?', re.DOTALL)


def _string_end(text, opening):
    """Index after the string that starts at ``opening`` (len(text) if open).

    memchr-speed find per quote; a string dense in escaped quotes finishes
    with one C regex match instead of a Python step per escape.
    """
    position, escaped = opening + 1, 0
    while True:
        quote = text.find('"', position)
        if quote < 0:
            return len(text)
        slash = quote
        while text[slash - 1] == "\\":
            slash -= 1
        if (quote - slash) % 2 == 0:
            return quote + 1
        escaped += 1
        if escaped > 16:
            return _STRING_TAIL.match(text, quote + 1).end()
        position = quote + 1


def _refuse_at(text, cut, kw):
    """The scanner reaches the over-limit bracket at ``cut`` unless it fails first."""
    try:
        json.loads(text[:cut], **kw)
    except json.JSONDecodeError as error:
        if error.pos < cut:
            raise json.JSONDecodeError(error.msg, text, error.pos) from None
    kind = "array" if text[cut] == "[" else "object"
    raise RecursionError(f"maximum recursion depth exceeded while decoding a JSON {kind} from a unicode string")


# Characters per NumPy block of the string-dense scan.
GUARD_BLOCK = 1 << 16


def guard_workspace_bytes(chars):
    """Transient bytes of one ``loads`` guard scan over ``chars`` decoded characters.

    Per block character, the arrays one _vector_scan iteration holds are the
    UTF-32 copy and its str slice (4 B each), index/last/before/toggles/step/
    levels and the two index results (8 B each) and three bool masks: 75 B. The
    largest single expression adds at most 5 more 8-byte temporaries. Arrays
    are released before the next block, so 128 B per block character bounds
    the peak (measured 80 B on 3.15.2/NumPy 2.5.4). 64 KiB covers array
    headers and the guard's four Python frames.
    """
    return 128 * min(chars, GUARD_BLOCK) + 65536


def _vector_scan(text, start, depth, kw, block=GUARD_BLOCK):
    """Finish a string-dense input in bounded NumPy blocks (outside a string at ``start``).

    A character is escaped after an odd backslash run; unescaped quotes toggle
    string state; brackets count only outside strings. Escapes outside
    strings are invalid JSON at which the scanner has already failed.
    """
    import numpy as np

    inside, run = 0, 0
    for low in range(start, len(text), block):
        codes = np.frombuffer(text[low:low + block].encode("utf-32-le", "surrogatepass"), dtype="<u4")
        index = np.arange(codes.size)
        slash = codes == 0x5C
        last = np.maximum.accumulate(np.where(slash, -1, index))
        before = np.empty(codes.size, dtype=np.int64)
        before[0] = run
        before[1:] = np.where(last[:-1] < 0, index[1:] + run, index[1:] - 1 - last[:-1])
        quote = (codes == 0x22) & (before % 2 == 0)
        toggles = np.cumsum(quote)
        outside = ((inside + toggles - quote) & 1) == 0
        step = (((codes == 0x5B) | (codes == 0x7B)) & outside).astype(np.int64)
        step -= ((codes == 0x5D) | (codes == 0x7D)) & outside
        levels = depth + np.cumsum(step)
        negative, deep = np.flatnonzero(levels < 0), np.flatnonzero(levels > NESTING_LIMIT)
        if deep.size and (not negative.size or deep[0] < negative[0]):
            _refuse_at(text, low + int(deep[0]), kw)
        if negative.size:
            return
        depth, inside = int(levels[-1]), int(inside + toggles[-1]) & 1
        run = int(codes.size - 1 - last[-1] if last[-1] >= 0 else codes.size + run)
        # One block's arrays at a time (guard_workspace_bytes).
        del codes, index, slash, last, before, quote, toggles, outside, step, levels, negative, deep


def _refuse_deep(text, kw):
    """Raise what 3.12's compiled-limit scanner raised if ``text`` nests too deep.

    One linear pass, entered only when the input holds more than NESTING_LIMIT
    opening brackets. The scanner's open containers equal the bracket depth on
    every input prefix it accepts; once that depth goes negative it has
    already failed. Structural spans between strings are checked
    NESTING_LIMIT characters at a time with C-speed counts (depth + opens
    bounds every prefix inside); only a block that could cross the limit is
    bisected to its exact crossing, the first opening bracket past the limit
    at ``cut``. Parsing ``text[:cut]`` (depth <= NESTING_LIMIT) reports any
    syntax error the scanner meets before that bracket, as it would on ``text``.
    Inputs dense in strings switch to bounded NumPy blocks.
    """
    depth, limit, start, size, strings = 0, NESTING_LIMIT, 0, len(text), 0
    while start < size:
        if strings > 4096 + start // 256:
            return _vector_scan(text, start, depth, kw)
        quote = text.find('"', start)
        end = size if quote < 0 else quote
        pending = [(block, min(block + limit, end)) for block in range(start, end, limit)]
        pending.reverse()
        while pending:
            low, high = pending.pop()
            opens = text.count("[", low, high) + text.count("{", low, high)
            if depth + opens <= limit:
                depth += opens - text.count("]", low, high) - text.count("}", low, high)
                if depth < 0:
                    return
                continue
            if high - low > 1:
                middle = (low + high) // 2
                pending += [(middle, high), (low, middle)]
                continue
            _refuse_at(text, low, kw)
        if quote < 0:
            return
        start = _string_end(text, quote)
        strings += 1


def loads(raw, **kw):
    """``json.loads`` refusing input nested deeper than NESTING_LIMIT.

    Decodes bytes exactly as json.loads does (detect_encoding/surrogatepass)
    and parses that same text; the bracket count is a C-speed scan of it and
    needs no copy unless refusing. Decoded text that still starts with a BOM
    goes to json.loads unchanged: its scanner fails at offset 0 without nesting.
    """
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (bytes, bytearray)):
        text = raw.decode(json.detect_encoding(raw), "surrogatepass")
        if text.startswith("\ufeff"):
            return json.loads(raw, **kw)
    else:
        return json.loads(raw, **kw)  # json's own TypeError.
    if text.count("[") + text.count("{") > NESTING_LIMIT:
        _refuse_deep(text, kw)
    return json.loads(text, **kw)


def require_stack_capacity(recursion_limit):
    """Check handler threads hold NESTING_LIMIT C encoder levels with margin.

    A C encode refused only by the thread's stack raises RecursionError and adds failure notes. Certified encoders
    run in default-attribute threads (no project stack_size): the glibc
    default stack, less its actual guard and CPython's soft limit
    (2*_PyOS_STACK_MARGIN_BYTES above the guard), must hold 512 bytes per
    encoder level and 2048 bytes per already-active Python frame.
    Encoder levels are a structural bound for the digest-bound 3.15.2
    executable: its _json recursion is encoder_listencode_obj calling itself
    (lists: 144-byte frame) or through encoder_encode_key_value (dicts:
    144 + 32 argument/alignment + 80 = 256 bytes per level), with every other
    call returning before the next level. The per-frame figure is a margin
    over a measurement (<=980 bytes per C-bridged frame), not a proof.
    """
    import ctypes
    import mmap
    import threading

    if threading.stack_size() != 0:
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    attr = ctypes.create_string_buffer(256)  # pthread_attr_t is 56 bytes on x86_64 glibc.
    size, guard = ctypes.c_size_t(), ctypes.c_size_t()
    if libc.pthread_getattr_default_np(attr) != 0:
        return False
    try:
        if (libc.pthread_attr_getstacksize(attr, ctypes.byref(size)) != 0
                or libc.pthread_attr_getguardsize(attr, ctypes.byref(guard)) != 0):
            return False
    finally:
        libc.pthread_attr_destroy(attr)
    page = mmap.PAGESIZE  # glibc rounds the guard up to whole pages.
    required = (-(-guard.value // page) * page + 2 * 16384
                + 512 * NESTING_LIMIT + 2048 * (recursion_limit + 50))
    return size.value >= required
