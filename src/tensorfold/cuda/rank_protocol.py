"""Bounded rank request data; limits follow the CUDA HTTP 96 MiB body contract.

Only the approved rank pair writes these messages. The native TCPStore returns
bytes before Python can inspect their size; senders therefore validate before
publication, and receivers bound decoding and expansion before use.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import io
import math

MAX_MESSAGE_BYTES = 96 << 20
MAX_GRAMMAR_ITEMS = MAX_MESSAGE_BYTES + 2


def bounded_json(body, *, separators=None, maximum=None):
    """Encode validated flat request fields without a larger transient string."""
    maximum = MAX_MESSAGE_BYTES if maximum is None else maximum
    if type(maximum) is not int or not 0 < maximum <= MAX_MESSAGE_BYTES:
        raise ValueError("rank JSON maximum must fit the existing transport envelope")
    size = 0
    with io.StringIO() as output:
        for chunk in json.JSONEncoder(separators=separators).iterencode(body):
            size += len(chunk)  # the default encoder emits ASCII
            if size > maximum:
                raise ValueError("rank request exceeds the 96 MiB transport envelope" if maximum == MAX_MESSAGE_BYTES
                                 else "rank request exceeds its declared transport envelope")
            output.write(chunk)
        return output.getvalue()


def ints(values, maximum, *, lower=0, upper=(1 << 31) - 1, empty=False):
    if (type(values) is not list or not (0 if empty else 1) <= len(values) <= maximum
            or any(type(value) is not int or not lower <= value <= upper for value in values)):
        raise ValueError("rank integer list exceeds its typed count/value contract")
    return values


def grammar(values, vocab):
    from tensorfold.engine.grammar import KINDS

    ints(values, MAX_GRAMMAR_ITEMS, empty=True)
    if values:
        if len(values) < 2 or values[0] not in range(len(KINDS)) or not 0 <= values[1] <= vocab:
            raise ValueError("rank grammar header is invalid")
        if any(values[index] > 255 for index in range(2, len(values))):
            raise ValueError("rank grammar payload must be UTF-8 bytes")
        bytes(values[index] for index in range(2, len(values))).decode("utf-8")
    return values


def packed_grammar(constraint, vocab):
    from tensorfold.engine.grammar import pack

    if constraint is not None:
        text = constraint.spec.text
        if (type(text) is not str or len(text) > MAX_MESSAGE_BYTES
                or len(text.encode("utf-8")) > MAX_MESSAGE_BYTES):
            raise ValueError("rank grammar exceeds the 96 MiB source envelope")
    return grammar(pack(constraint), vocab)


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("rank sampling values must be finite numbers")
    return value


def request_json(raw, *, counts):
    """Reject size/nesting/numeric expansion before the JSON decoder allocates.

    Rank requests contain an object of scalar values and flat lists. Strings
    are only field names; nested objects/lists have no supported schema.
    """
    if type(raw) not in (str, bytes) or len(raw) > MAX_MESSAGE_BYTES:
        raise ValueError("rank request exceeds the 96 MiB transport envelope")
    if type(raw) is bytes:
        raw = raw.decode("utf-8")
    elif len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("rank request exceeds the UTF-8 transport envelope")
    depth, quoted, escaped, key_start = 0, False, False, 0
    last_key, array_maximum, array_count, next_item = None, 0, 0, False
    allowed = set(counts) | {"max_tokens", "draft", "cached", "stop_eos", "stop"}
    seen, numeric_length = set(), 0
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
                if index - key_start > 64:
                    raise ValueError("rank JSON strings are bounded field names")
                last_key = decoder.raw_decode(raw[key_start:index + 1])[0]
                if last_key not in allowed or last_key in seen:
                    raise ValueError("rank request has unknown or duplicate fields")
                seen.add(last_key)
        elif char == '"':
            if depth != 1:
                raise ValueError("rank lists contain only scalar numbers")
            quoted = True
            key_start = index
        elif char in "[{":
            depth += 1
            if depth > 2:
                raise ValueError("rank request requires flat schema lists")
            if char == "[":
                if last_key not in counts:
                    raise ValueError("rank array field has no declared count bound")
                array_maximum, array_count, next_item = counts[last_key], 0, True
            elif depth == 2:
                raise ValueError("rank request has no nested objects")
        elif char in "]}":
            depth -= 1
        elif depth == 2:
            if char == ",":
                next_item = True
            elif not char.isspace() and next_item:
                array_count += 1
                if array_count > array_maximum:
                    raise ValueError("rank list exceeds its count before JSON expansion")
                next_item = False
        if not quoted and char in "0123456789+-.eE":
            numeric_length += 1
            if numeric_length > 24:  # maximum finite float64 repr, including sign/exponent
                raise ValueError("rank numeric token exceeds its scalar envelope")
        else:
            numeric_length = 0
    def integer(text):
        if len(text.lstrip("-")) > 20:
            raise ValueError("rank integer exceeds its 64-bit envelope")
        return int(text)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate rank request field")
            result[key] = value
        return result
    def constant(text):
        raise ValueError("rank request cannot contain non-finite constants")
    body = json.loads(raw, parse_int=integer, parse_float=lambda text: finite(float(text)),
                      parse_constant=constant, object_pairs_hook=pairs)
    if type(body) is not dict:
        raise ValueError("rank request must be an object")
    return body


@dataclass(frozen=True)
class FiniteTPContract:
    """Closed finite text role; it never changes native logical/KV capacity.

    A generic quiescent configure operation is fully funded separately. Only
    after all peers acknowledge the same role may requests use its smaller
    source/parse bounds. Oversize receives remain96MiB RAW error obligations.
    """
    context: int
    vocab: int
    inputs: int
    outputs: int
    requests: int
    depth: int

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or not 0 < value < (1 << 63):
                raise ValueError("finite TP geometry requires positive native integers")
        if self.inputs + self.outputs + 2 * (self.depth + 1) > self.context:
            raise ValueError("finite text span exceeds the native logical context")
        if self.command_bytes > MAX_MESSAGE_BYTES:
            raise ValueError("finite command schema exceeds the native frame envelope")

    def record(self):
        return {"schema": "tensorfold.finite-tp-text-v1", **asdict(self)}

    @classmethod
    def from_record(cls, record):
        if (type(record) is not dict or set(record) != {"schema", *cls.__dataclass_fields__}
                or record["schema"] != "tensorfold.finite-tp-text-v1"):
            raise ValueError("invalid finite TP role schema")
        return cls(**{name: record[name] for name in cls.__dataclass_fields__})

    @property
    def cpu_tokens(self):
        # Old/new candidate windows each contain anchor+depth drafts. Every
        # finite Stream list is bounded by input+reply plus those transient rows.
        return self.inputs + self.outputs + 2 * (self.depth + 1)

    @property
    def command_bytes(self):
        # Scalar/header bound follows the exact accepted root keys below.
        # Each metadata subtree is at most4 levels of <=16 fixed keys, with
        # strings<=128 codepoints and native63-bit integers. The variable
        # lists are prompt/points and four per-request control/order lists.
        # JSON default ensure_ascii uses <=12 bytes per Unicode codepoint.
        fixed = 16 * (2 + 12 * 128 + 1 + 20 + 1)
        metadata = 4 * fixed
        prompt_and_points = (len(str(self.vocab - 1)) + 1 + 20) * self.inputs
        per_request_control = (20 + 2 + 20 + 6 + 2 + 20 + 12 * 16 + 20 + 4 + 20) * self.requests
        drop_error = 12 * 1024 + fixed
        return metadata + prompt_and_points + per_request_control + drop_error

    @property
    def ack_bytes(self):
        prototype = {"error": "\U00010000" * 64 + ":" + "f" * 64, "state": "f" * 64}
        return len(json.dumps(prototype, separators=(",", ":")))

    def _integer(self, value, *, minimum=0, maximum=(1 << 63) - 1):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("finite TP scalar exceeds its native range")

    def _keys(self, value, keys):
        if type(value) is not dict or set(value) != set(keys):
            raise ValueError("finite TP object differs from its closed schema")

    def _ids(self, value, maximum, upper):
        if type(value) is not list or len(value) > maximum:
            raise ValueError("finite TP list exceeds its role count")
        for item in value:
            self._integer(item, maximum=upper)

    def _identity(self, value):
        if value is not None and (type(value) is not str or not 1 <= len(value) <= 128):
            raise ValueError("finite TP profile identity exceeds the existing128-character contract")

    def _execution(self, value, *, boundary):
        fields = {"row_budget", "proposal_limit", "prefill_tokens", "session_ceiling"}
        family = {"kernel_family", "graph_family"}
        if boundary:
            self._keys(value, {"epoch", "settings", "profile_id", *family})
            self._keys(value["settings"], fields)
            settings = value["settings"]
        else:
            self._keys(value, {"epoch", "profile_id", "order", "migration", *fields, *family})
            settings = {key: value[key] for key in fields}
            self._ids(value["order"], self.requests, (1 << 63) - 1)
            if value["migration"] is not None:
                self._keys(value["migration"], {"requested_degree", "request_epoch", "pending"})
                self._integer(value["migration"]["requested_degree"], minimum=1)
                self._integer(value["migration"]["request_epoch"])
                if value["migration"]["pending"] is not False:
                    raise ValueError("finite fixed-resident measurement excludes pending administrative migration")
        self._integer(value["epoch"])
        for item in settings.values():
            self._integer(item, minimum=1)
        self._identity(value["profile_id"])
        for key in family:
            if type(value[key]) is not str or not 1 <= len(value[key]) <= 128:
                raise ValueError("finite execution family must be a bounded existing family identity")

    def validate_command(self, body):
        if type(body) is not dict or type(body.get("kind")) is not str:
            raise ValueError("finite TP command requires a typed operation")
        kind = body["kind"]
        fields = {"admit": {"wire", "stream", "points"}, "round": {"execution"},
                  "cancel": {"wires"}, "finish": {"wires"}, "drop": {"error"}}
        if kind == "stop":
            self._keys(body, {"kind", "abort"})
            if type(body["abort"]) is not bool:
                raise ValueError("finite stop requires a Boolean failure flag")
            return
        if kind not in fields:
            raise ValueError("operation is outside the finite text role")
        self._keys(body, {"kind", "sequence", "control", *fields[kind]})
        self._integer(body["sequence"])
        control = body["control"]
        self._keys(control, {"forget", "cancel", "arrived", "admission_policy", "flow", "pass_rows", "execution_boundary"})
        if control["admission_policy"] != "fifo" or type(control["arrived"]) is not bool:
            raise ValueError("finite TP controls require ordinary FIFO text")
        self._integer(control["pass_rows"])
        self._ids(control["forget"], self.requests, (1 << 63) - 1)
        for name, width in (("cancel", 2), ("flow", 3)):
            rows = control[name]
            if type(rows) is not list or len(rows) > self.requests:
                raise ValueError("finite TP controls exceed live request ownership")
            for row in rows:
                if type(row) is not list or len(row) != width:
                    raise ValueError("finite TP control row differs from its schema")
                self._integer(row[0])
                if name == "cancel":
                    if type(row[1]) is not bool:
                        raise ValueError("finite cancellation must be Boolean")
                else:
                    if row[1] not in (None, "client_input", "output", "tier_unavailable", "restoring", "sequence_busy"):
                        raise ValueError("finite control has an unsupported blocker")
                    if row[2] is not None:
                        self._integer(row[2], maximum=self.outputs)
        self._execution(control["execution_boundary"], boundary=True)
        if kind == "admit":
            self._integer(body["wire"])
            self._ids(body["points"], self.inputs, self.inputs)
            value = body["stream"]
            self._keys(value, {"prompt", "count", "draft", "stop_eos", "background", "sampling", "grammar",
                "owed", "carry", "token_limit", "profile_epoch", "committed_output_tokens", "admission_ticket",
                "principal", "retention", "output_capacity"})
            self._ids(value["prompt"], self.inputs, self.vocab - 1)
            if not value["prompt"]:
                raise ValueError("finite prompt cannot be empty")
            self._integer(value["count"], minimum=1, maximum=self.outputs)
            if (type(value["draft"]) is not bool or value["stop_eos"] is not True or value["background"] is not False
                    or type(value["grammar"]) is not list or value["grammar"] or type(value["owed"]) is not list or value["owed"]
                    or value["carry"] is not None or value["token_limit"] is not False or value["profile_epoch"] is not None
                    or value["principal"] is not None or value["retention"] is not None):
                raise ValueError("finite text role excludes media/grammar/replay/retention/foreign callbacks")
            sampling = value["sampling"]
            if (type(sampling) is not list or len(sampling) != 5 or type(sampling[0]) is not int or sampling[0] != 17
                    or type(sampling[2]) is not int or sampling[2] != 0
                    or any(type(sampling[index]) not in (int, float) or sampling[index] != expected
                           for index, expected in ((1, 0), (3, 1), (4, 0)))):
                raise ValueError("finite text role requires the fixed ordinary greedy sampler")
            self._integer(value["committed_output_tokens"], maximum=0)
            self._integer(value["admission_ticket"])
            self._integer(value["output_capacity"], minimum=1, maximum=self.outputs)
        elif kind == "round":
            self._execution(body["execution"], boundary=False)
        elif kind in ("cancel", "finish"):
            self._ids(body["wires"], self.requests, (1 << 63) - 1)
        elif body["error"] is not None:
            self._keys(body["error"], {"type", "message"})
            if (type(body["error"]["type"]) is not str or len(body["error"]["type"]) > 128
                    or type(body["error"]["message"]) is not str or len(body["error"]["message"]) > 1024):
                raise ValueError("finite retirement error exceeds its existing bounded source role")

    def loads(self, payload, *, acknowledgment=False):
        maximum = self.ack_bytes if acknowledgment else self.command_bytes
        if type(payload) is not bytes or len(payload) > maximum:
            raise ValueError("finite TP received frame exceeds its declared raw-byte bound")
        # The existing closed JSON codec has no executable object reconstruction.
        value = json.loads(payload)
        if acknowledgment:
            self._keys(value, {"error", "state"})
            if (type(value["state"]) is not str or len(value["state"]) != 64
                    or value["error"] is not None and (type(value["error"]) is not str or len(value["error"]) > 129)):
                raise ValueError("finite TP acknowledgment differs from its closed error/state schema")
        else:
            self.validate_command(value)
        return value


def finite_error_identity(error):
    """Bounded wire identity, full native primary retained by existing owners.

    Hashing the full type name in bounded chunks retains distinctions beyond
    the diagnostic prefix without materializing an unbounded encoded string.
    """
    name = type.__dict__["__name__"].__get__(type(error))
    digest = hashlib.sha256()
    for start in range(0, len(name), 4096):
        digest.update(name[start:start + 4096].encode("utf-8", errors="surrogatepass"))
    return str.__getitem__(name, slice(None, 64)) + ":" + digest.hexdigest()
