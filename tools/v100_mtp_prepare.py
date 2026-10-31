"""Extract only indexed mtp.* tensors into a bounded, checksummed control import.

Use a local immutable HF checkpoint or an explicit repository/commit. Remote
preparation retrieves metadata and exact tensor ranges, never complete trunk
shards. The result keeps original tensor bytes and configuration; conversion to
FP16 remains the runtime importer's separate, qualified numerical decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.tensor_file import (
    MAX_HEADER_BYTES, _metadata_object, _regular_stream, checkpoint_path, parse_header_bytes, read_metadata_json,
)

CHUNK = 8 << 20
METADATA_LIMIT = 16 << 20
SIZES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4,
         "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def _name(name):
    if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts or "\\" in name or any(c in name for c in "?#\0")):
        raise ValueError("source file must be a safe relative checkpoint name")
    return name


def _digest(file):
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        while data := stream.read(CHUNK):
            digest.update(data)
    return digest.hexdigest()


def _json(file, value):
    temporary = file.with_suffix(file.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, file)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    primary = None
    try:
        os.fsync(fd)
    except BaseException as error:
        primary = error
    try:
        os.close(fd)
    except BaseException as cleanup:
        raise_failures(primary, [cleanup])
    if primary is not None:
        raise primary


class HTTPSOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("checkpoint redirects must preserve HTTPS")
        return super().redirect_request(request, fp, code, message, headers, newurl)


class Source:
    """A single preparation's bounded source reads and immutable file identities."""

    def __init__(self, *, directory=None, repository=None, revision=None, deadline_seconds=3600):
        if type(deadline_seconds) is not int or deadline_seconds <= 0:
            raise ValueError("preparation deadline must be a positive integer")
        self.deadline = time.monotonic() + deadline_seconds
        self.directory = Path(directory).resolve(strict=True) if directory is not None else None
        if self.directory is None:
            if (not isinstance(repository, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repository)
                    or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
                raise ValueError("remote extraction requires an explicit repository and immutable 40-hex commit")
            self.identity = {"repository": repository, "revision": revision}
            self.base = f"https://huggingface.co/{repository}/resolve/{revision}/"
            self.opener = urllib.request.build_opener(HTTPSOnly())
        else:
            self.identity = {"directory": str(self.directory)}
        self.files = {}

    def _remaining(self):
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("MTP preparation deadline expired")
        return min(30, seconds)

    def _body(self, response, count, *, exact):
        """Cooperative deadline, at most one already-issued socket read late.

        read1 returns after one buffered/raw read instead of waiting to fill
        count while a peer continually trickles bytes. DNS/header acquisition
        and an in-progress socket read retain their transport timeout bound.
        """
        if response.headers.get("Transfer-Encoding"):
            # HTTPResponse.read1's chunk parser can internally wait for several
            # raw reads. Static checkpoint sources must supply ordinary framing.
            raise ValueError("MTP preparation refuses transfer-encoded response bodies")
        buffer, received = bytearray(count), 0
        while received < count:
            self._remaining()
            part = response.read1(min(CHUNK, count - received))
            self._remaining()
            if not part:
                break
            buffer[received:received + len(part)] = part
            received += len(part)
        if exact and received != count:
            raise ValueError("source checkpoint returned a truncated byte range")
        if exact:
            self._remaining()
            extra = response.read1(1)
            self._remaining()
            if extra:
                raise ValueError("remote range response exceeds its requested extent")
        return bytes(memoryview(buffer)[:received])

    def read(self, name, start, count, *, maximum=CHUNK):
        name = _name(name)
        if (type(start) is not int or type(count) is not int or start < 0 or count <= 0
                or count > maximum):
            raise ValueError("source read exceeds its declared positive byte range")
        self._remaining()
        if self.directory is not None:
            path = checkpoint_path(self.directory, name)
            with _regular_stream(path) as stream:
                info = os.fstat(stream.fileno())
                identity = {"bytes": info.st_size, "device": info.st_dev, "inode": info.st_ino,
                            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}
                if name in self.files and self.files[name] != identity:
                    raise ValueError("local checkpoint changed during extraction")
                if start > info.st_size or count > info.st_size - start:
                    raise ValueError("source tensor range exceeds its file")
                stream.seek(start)
                data = stream.read(count)
                after = os.fstat(stream.fileno())
                if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (info.st_size, info.st_mtime_ns, info.st_ctime_ns):
                    raise ValueError("local checkpoint changed during a tensor read")
            self.files[name] = identity
        else:
            request = urllib.request.Request(self.base + urllib.parse.quote(name, safe="/") + "?download=true",
                                             headers={"Range": f"bytes={start}-{start + count - 1}",
                                                      "User-Agent": "TensorFold-bounded-MTP-preparation"})
            try:
                with self.opener.open(request, timeout=self._remaining()) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if (response.status != 206 or match is None
                            or tuple(map(int, match.groups()[:2])) != (start, start + count - 1)):
                        raise ValueError("remote checkpoint did not honor the exact bounded range")
                    identity = {"bytes": int(match[3])}
                    if name in self.files and self.files[name] != identity:
                        raise ValueError("remote checkpoint size changed during extraction")
                    data = self._body(response, count, exact=True)
                    self.files[name] = identity
            except urllib.error.URLError as error:
                status = f"HTTP {error.code}" if isinstance(error, urllib.error.HTTPError) else type(error).__name__
                # Redirect exception strings may contain signed transport URLs.
                raise RuntimeError(f"MTP source read failed: {name} range {start}+{count}: {status}") from None
        if len(data) != count:
            raise ValueError("source checkpoint returned a truncated byte range")
        return data

    def metadata(self, name):
        if name not in ("config.json", "model.safetensors.index.json"):
            raise ValueError("only the bounded config and tensor index use whole-file metadata reads")
        if self.directory is None:
            try:
                with self.opener.open(self.base + name, timeout=self._remaining()) as response:
                    if response.status != 200:
                        raise ValueError("checkpoint JSON response must be complete")
                    data = self._body(response, METADATA_LIMIT + 1, exact=False)
            except urllib.error.URLError as error:
                status = f"HTTP {error.code}" if isinstance(error, urllib.error.HTTPError) else type(error).__name__
                raise RuntimeError(f"MTP metadata read failed: {name}: {status}") from None
            if not 0 < len(data) <= METADATA_LIMIT:
                raise ValueError("checkpoint JSON exceeds the preparation metadata bound")
            self.files[name] = {"bytes": len(data)}
            return data
        self.read(name, 0, 1)
        size = self.files[name]["bytes"]
        if not 0 < size <= METADATA_LIMIT:
            raise ValueError("checkpoint JSON exceeds the preparation metadata bound")
        return self.read(name, 0, size, maximum=METADATA_LIMIT)

    def header(self, name):
        count = struct.unpack("<Q", self.read(name, 0, 8))[0]
        if not 0 < count <= min(MAX_HEADER_BYTES, METADATA_LIMIT, self.files[name]["bytes"] - 8):
            raise ValueError("source safetensors header exceeds the bounded preparation envelope")
        raw = self.read(name, 8, count, maximum=METADATA_LIMIT)
        base, header = parse_header_bytes(raw, self.files[name]["bytes"], SIZES, label=name)
        return base, header, hashlib.sha256(raw).hexdigest()


def prepare(source, output):
    output = Path(output).resolve()
    config_raw = source.metadata("config.json")
    config = _metadata_object(config_raw, "config.json")
    text = config.get("text_config") or config
    if (config.get("model_type") != "qwen4_exp" or (text.get("mtp") or {}).get("num_hidden_layers") != 1):
        raise ValueError("MTP control preparation requires the declared single-layer Flash Next source")
    index_raw = source.metadata("model.safetensors.index.json")
    index = _metadata_object(index_raw, "model.safetensors.index.json")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict):
        raise ValueError("checkpoint index must declare its weight_map")
    selected = {name: _name(file) for name, file in mapping.items() if name.startswith("mtp.")}
    if not selected or len(selected) > 4096 or len(set(selected.values())) > 64:
        raise ValueError("MTP tensor/source count exceeds the preparation envelope")
    headers, tensors, total = {}, {}, 0
    for file in sorted(set(selected.values())):
        base, header, digest = source.header(file)
        headers[file] = {"sha256": digest, "bytes": source.files[file]["bytes"]}
        for name, entry in header.items():
            if name.startswith("mtp.") and selected.get(name) != file:
                raise ValueError("MTP source header disagrees with the complete index namespace")
        for name in sorted(name for name, file_ in selected.items() if file_ == file):
            entry = header.get(name)
            if not isinstance(entry, dict) or entry["dtype"] != "BF16":
                raise ValueError("official MTP control must contain indexed original BF16 tensors")
            lo, hi = entry["data_offsets"]
            tensors[name] = {"source": file, "source_offset": base + lo, "dtype": entry["dtype"],
                             "shape": entry["shape"], "bytes": hi - lo, "data_offsets": [total, total + hi - lo]}
            total += hi - lo
    if total > 16 << 30:
        raise ValueError("MTP control payload exceeds the declared 16GiB preparation envelope")
    plan = {"schema": "tensorfold.mtp-control-import", "version": 1, "source": source.identity,
            "config_sha256": hashlib.sha256(config_raw).hexdigest(),
            "index_sha256": hashlib.sha256(index_raw).hexdigest(), "headers": headers,
            "tensor_count": len(tensors), "payload_bytes": total, "tensors": tensors,
            "chunk_bytes": CHUNK, "source_full_payload_checksums": "not read or verified"}
    work = output.with_name(output.name + ".partial")
    if output.exists():
        receipt = read_metadata_json(output / "import-manifest.json", max_bytes=METADATA_LIMIT)
        if (receipt.get("plan") != plan or _digest(output / "model.safetensors") != receipt.get("sha256")
                or _digest(output / "config.json") != plan["config_sha256"]):
            raise ValueError("existing MTP import does not match the immutable source plan")
        # Rename can have succeeded before a parent-directory barrier failed.
        # A retry must finish that publication rather than only verify content.
        _sync_directory(output)
        _sync_directory(output.parent)
        return receipt
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / "plan.json"
    if plan_path.exists():
        if read_metadata_json(plan_path, max_bytes=METADATA_LIMIT) != plan:
            raise ValueError("partial MTP import belongs to a different immutable source plan")
    else:
        _json(plan_path, plan)
    receipts = {}
    for name, entry in tensors.items():
        token = hashlib.sha256(name.encode()).hexdigest()
        final, record = work / (token + ".tensor"), work / (token + ".json")
        if final.exists() and record.exists():
            prior = read_metadata_json(record, max_bytes=256)
            if (set(prior) != {"bytes", "sha256"} or type(prior["bytes"]) is not int
                    or prior["bytes"] != entry["bytes"] or not isinstance(prior["sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", prior["sha256"]) is None):
                raise ValueError("completed tensor receipt differs from its bounded schema or source byte count")
            if final.stat().st_size == entry["bytes"] and _digest(final) == prior.get("sha256"):
                receipts[name] = {"bytes": prior["bytes"], "sha256": prior["sha256"]}
                continue
        partial = final.with_suffix(".part")
        digest = hashlib.sha256()
        with partial.open("wb") as destination:
            for at in range(0, entry["bytes"], CHUNK):
                data = source.read(entry["source"], entry["source_offset"] + at, min(CHUNK, entry["bytes"] - at))
                destination.write(data)
                digest.update(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(partial, final)
        receipts[name] = {"bytes": entry["bytes"], "sha256": digest.hexdigest()}
        _json(record, receipts[name])
    header = {name: {key: entry[key] for key in ("dtype", "shape", "data_offsets")} for name, entry in tensors.items()}
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    digest = hashlib.sha256()
    with (work / "model.safetensors.part").open("wb") as destination:
        prefix = struct.pack("<Q", len(raw)) + raw
        destination.write(prefix)
        digest.update(prefix)
        for name in tensors:
            with (work / (hashlib.sha256(name.encode()).hexdigest() + ".tensor")).open("rb") as tensor:
                while data := tensor.read(CHUNK):
                    destination.write(data)
                    digest.update(data)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(work / "model.safetensors.part", work / "model.safetensors")
    with (work / "config.json").open("wb") as destination:
        destination.write(config_raw)
        destination.flush()
        os.fsync(destination.fileno())
    receipt = {"plan": plan, "sha256": digest.hexdigest(), "bytes": 8 + len(raw) + total,
               "tensor_checksums": receipts, "execution_qualification": "unrun"}
    _json(work / "import-manifest.json", receipt)
    # Only owned staging files are removed; source artifacts are immutable.
    for name in tensors:
        token = hashlib.sha256(name.encode()).hexdigest()
        (work / (token + ".tensor")).unlink()
        (work / (token + ".json")).unlink()
    plan_path.unlink()
    _sync_directory(work)
    os.rename(work, output)
    _sync_directory(output.parent)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-dir", type=Path)
    group.add_argument("--repository")
    parser.add_argument("--revision", help="required immutable commit for a remote source")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    source = Source(directory=args.source_dir, repository=args.repository, revision=args.revision,
                    deadline_seconds=args.deadline_seconds)
    receipt = prepare(source, args.out)
    print(json.dumps({key: receipt[key] for key in ("sha256", "bytes", "execution_qualification")}))


if __name__ == "__main__":
    main()
