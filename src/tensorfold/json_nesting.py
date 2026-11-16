"""JSON container nesting at untrusted parse boundaries, per selected CPython.

CPython 3.12 raises RecursionError once its compiled C recursion counter
(Py_C_RECURSION_LIMIT, 10000) is exhausted inside the C scanner. CPython 3.14+
replaced that counter with C-stack-pointer checks, so the same input may nest
as deep as the thread stack allows. The host payload projections pay for at
most NESTING_LIMIT open containers on the boundaries that parse through
``loads``; on runtimes without the compiled counter ``loads`` therefore refuses
deeper input before the C scanner runs, with the error 3.12 raises for it.
On the compiled-counter runtime it is exactly ``json.loads``.
"""

import json
import re
import sys

NESTING_LIMIT = 10000
# Runtimes whose json C scanner/encoder enforce the compiled C recursion limit.
COMPILED_LIMIT = sys.version_info[:2] < (3, 14)

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


def _vector_scan(text, start, depth, kw, block=1 << 16):
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


def _refuse_deep(text, kw):
    """Raise what the compiled-limit scanner raises if ``text`` nests too deep.

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
    """``json.loads`` with the compiled 3.12 container-depth contract.

    Decodes bytes exactly as json.loads does (detect_encoding/surrogatepass)
    and parses that same text; the bracket count is a C-speed scan of it and
    needs no copy unless refusing. Decoded text that still starts with a BOM
    goes to json.loads unchanged: its scanner fails at offset 0 without nesting.
    """
    if COMPILED_LIMIT or isinstance(raw, str):
        text = raw
    elif isinstance(raw, (bytes, bytearray)):
        text = raw.decode(json.detect_encoding(raw), "surrogatepass")
        if text.startswith("\ufeff"):
            return json.loads(raw, **kw)
    else:
        return json.loads(raw, **kw)  # json's own TypeError.
    if not COMPILED_LIMIT and text.count("[") + text.count("{") > NESTING_LIMIT:
        _refuse_deep(text, kw)
    return json.loads(text, **kw)


def require_stack_capacity(recursion_limit):
    """Check handler threads hold NESTING_LIMIT C encoder levels with margin.

    Without the compiled counter, a C encode refused only by the thread's
    stack raises RecursionError and adds failure notes. Certified encoders
    run in default-attribute threads (no project stack_size): the glibc
    default must hold the stack margins, 512 bytes per encoder level and 2048
    bytes per already-active Python frame. Measured on the selected 3.15.2
    binary: <=106 bytes per dict/list level, <=980 bytes per C-bridged frame.
    """
    import ctypes
    import threading

    if threading.stack_size() != 0:
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    attr = ctypes.create_string_buffer(256)  # pthread_attr_t is 56 bytes on x86_64 glibc.
    size = ctypes.c_size_t()
    if libc.pthread_getattr_default_np(attr) != 0:
        return False
    try:
        if libc.pthread_attr_getstacksize(attr, ctypes.byref(size)) != 0:
            return False
    finally:
        libc.pthread_attr_destroy(attr)
    # Soft limit 2*_PyOS_STACK_MARGIN_BYTES above the base; <=64KiB guard pages.
    required = 2 * 16384 + 65536 + 512 * NESTING_LIMIT + 2048 * (recursion_limit + 50)
    return size.value >= required
