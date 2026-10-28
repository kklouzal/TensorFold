"""Closed6513 bounded streaming reader reused for post-retirement capture summaries.

CHUNK/MAX_CARRY count Unicode characters; conservative UTF8 bounds are4x.
"""
import hashlib
import json

CHUNK = 1 << 20
MAX_CARRY = 16 << 20

class Stream:
    def __init__(self, path):
        self.file = path.open("r", encoding="utf-8", newline="")
        self.decoder = json.JSONDecoder()
        self.digest = hashlib.sha256()
        self.buffer, self.position, self.eof, self.bytes = "", 0, False, 0

    def more(self):
        self.buffer = self.buffer[self.position:]
        self.position = 0
        block = self.file.read(CHUNK)
        raw = block.encode("utf-8")
        self.digest.update(raw)
        self.bytes += len(raw)
        self.eof = not block
        self.buffer += block
        if len(self.buffer) > MAX_CARRY:
            raise ValueError("JSON value/carry exceeds explicit16777216-character analysis bound")

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.more()

    def expect(self, character):
        if self.peek() != character:
            raise ValueError("malformed Chrome JSON delimiter: expected" + character)
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                if end == len(self.buffer) and not self.eof:
                    self.more()
                    continue
                self.position = end
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.more()

    def events(self, header):
        self.expect("{")
        seen = False
        while self.peek() != "}":
            key = self.value()
            self.expect(":")
            if key == "traceEvents":
                if seen:
                    raise ValueError("duplicate traceEvents")
                seen = True
                self.expect("[")
                while self.peek() != "]":
                    event = self.value()
                    if not isinstance(event, dict):
                        raise ValueError("trace event must be an object")
                    yield event
                    if self.peek() != ",":
                        break
                    self.position += 1
                self.expect("]")
            else:
                header[key] = self.value()
            if self.peek() != ",":
                break
            self.position += 1
        self.expect("}")
        if not seen or self.peek():
            raise ValueError("missing events or trailing JSON data")

