"""Closed two-origin derivation fixtures; no actual artifact or compiler admission."""
import base64
import copy
import csv
import io
import json
import zipfile
from pathlib import Path

import pytest
from test_sm70_gdn_chunk64_artifact import identity
from test_sm70_gdn_chunk64_artifact import installed as installed  # noqa: PLC0414 - pytest fixture re-export

from tensorfold.cuda.sm70 import gdn_chunk64_artifact as artifact


def wheel(path, files, record_mutation=None):
    members = {name: identity(raw) for name, raw in files.items()}
    root = next(Path(name).parts[0] for name in files if name.endswith(".dist-info/METADATA"))
    record = root + "/RECORD"
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    for name, item in sorted(members.items()):
        sha = base64.urlsafe_b64encode(bytes.fromhex(item["sha256"])).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + sha, item["bytes"]))
    writer.writerow((record, "", ""))
    raw = text.getvalue().encode()
    if record_mutation is not None:
        raw = record_mutation(raw)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr(record, raw)
    return members


@pytest.fixture
def derived(installed, tmp_path, monkeypatch):
    old_attrs, new_attrs = b"controlled old DictAttrs\n", b"controlled new DictAttrs\n"
    monkeypatch.setattr(artifact, "ATTRS_BEFORE", (len(old_attrs), identity(old_attrs)["sha256"]))
    monkeypatch.setattr(artifact, "ATTRS_AFTER", (len(new_attrs), identity(new_attrs)["sha256"]))
    value = copy.deepcopy(installed.value)
    value["schema"] = artifact.DERIVED_SCHEMA
    value["source_pins"] = dict(artifact.NATIVE_PINS)
    value["build"]["source_archive_sha256"] = artifact.NATIVE_ARCHIVE
    old_version, new_version = "0.1.10+fixture", "0.1.10+fixture.py"
    old_root, new_root = "tilelang-" + old_version + ".dist-info", "tilelang-" + new_version + ".dist-info"
    files = {name: (installed.site / name).read_bytes() for name in value["tilelang"]["installed_members"]}
    files[artifact.ATTRS_MEMBER] = old_attrs
    files[old_root + "/METADATA"] = ("Name: tilelang\nVersion: " + old_version + "\nSummary: unchanged\n\nbody\n").encode()
    files[old_root + "/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: native fixture\nRoot-Is-Purelib: false\nTag: cp310-abi3-linux_x86_64\n\n"
    files[old_root + "/licenses/LICENSE"] = b"unchanged attribution\n"
    base_path = tmp_path / "base.whl"
    base_members = wheel(base_path, files)
    base = {"version": old_version, "wheel": identity(base_path.read_bytes()), "installed_members": base_members}
    packaging = {"proposal_sha256": "6" * 64, "producer_sha256": "7" * 64,
                 "recipe_identity_sha256": "8" * 64, "image_id": "sha256:" + "9" * 64,
                 "tool": {"name": "wheel", "version": "0.48.0", "wheel_sha256": "a" * 64}}
    changes = [{"member": artifact.ATTRS_MEMBER, "before": identity(old_attrs), "after": identity(new_attrs)}]
    provenance = {"schema": artifact.PROVENANCE_SCHEMA,
        "native": {"source_pins": value["source_pins"], "build": value["build"],
                   "base_tilelang": {key: base[key] for key in ("version", "wheel")}},
        "python_source_pins": dict(artifact.PYTHON_PINS), "changes": changes,
        "packager": {**packaging["tool"], "producer_sha256": packaging["producer_sha256"]},
        "recipe_identity_sha256": packaging["recipe_identity_sha256"]}
    embedded = json.dumps(provenance, sort_keys=True).encode()
    new_files = {name.replace(old_root + "/", new_root + "/"): raw for name, raw in files.items()}
    new_files[artifact.ATTRS_MEMBER] = new_attrs
    new_files[new_root + "/METADATA"] = files[old_root + "/METADATA"].replace(old_version.encode(), new_version.encode())
    new_files[new_root + "/WHEEL"] = files[old_root + "/WHEEL"].replace(
        b"native fixture", artifact.PACKAGER_GENERATOR.encode())
    new_files[artifact.PROVENANCE_MEMBER] = embedded
    path = tmp_path / "derived.whl"
    members = wheel(path, new_files)
    value["tilelang"] = {"version": new_version, "wheel": identity(path.read_bytes()), "installed_members": members}
    value["derivation"] = {"schema": artifact.DERIVATION_SCHEMA, "base_tilelang": base,
                          "python_source_pins": dict(artifact.PYTHON_PINS), "changes": changes,
                          "packaging": packaging,
                          "embedded_provenance": {"member": artifact.PROVENANCE_MEMBER, **identity(embedded)}}
    for name, raw in new_files.items():
        target = installed.site / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    installed.versions["tilelang"] = new_version
    return {"value": value, "base": base_path, "wheel": path, "files": new_files,
            "installation": installed, "tmp": tmp_path, "new_root": new_root}


def selection(data, monkeypatch):
    value = data["value"]
    receipt = artifact.publish(data["tmp"] / "selection.json", value)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(data["tmp"] / receipt["cache_namespace"]))
    return artifact.OwnedArtifact(receipt["path"], receipt["bytes"], receipt["sha256"],
                                  str(data["wheel"]), receipt["required_hash_bytes"], str(data["base"]))


def test_full_derivation_inspection_preserves_both_origins_and_exact_native_bytes(derived, monkeypatch):
    selected = selection(derived, monkeypatch)
    installed = derived["installation"]
    record = selected.inspect(installed.env, installed.versions)
    assert record["schema"] == artifact.DERIVED_SCHEMA
    assert record["source_pins"] == dict(artifact.NATIVE_PINS)
    assert record["derivation"]["python_source_pins"] == dict(artifact.PYTHON_PINS)
    assert record["derivation"]["base_tilelang"]["wheel"] == derived["value"]["derivation"]["base_tilelang"]["wheel"]
    assert record["hashed_bytes"] == artifact.hash_bytes(derived["value"])


@pytest.mark.parametrize("failure", ["native-source", "python-source", "changed-path", "native-bytes",
                                   "extra-member", "packager", "self-reference"])
def test_derivation_schema_refuses_unapproved_sources_or_changes(derived, failure):
    value = copy.deepcopy(derived["value"])
    if failure == "native-source":
        value["source_pins"]["tvm"] = "f" * 40
    elif failure == "python-source":
        value["derivation"]["python_source_pins"]["tvm"] = "f" * 40
    elif failure == "changed-path":
        value["derivation"]["changes"][0]["member"] = "tilelang/compiler.py"
    elif failure == "native-bytes":
        value["tilelang"]["installed_members"]["tilelang/lib/libtilelang.so"] = identity(b"changed native")
    elif failure == "extra-member":
        value["tilelang"]["installed_members"]["tilelang/unknown.py"] = identity(b"unknown")
    elif failure == "packager":
        value["derivation"]["packaging"]["tool"]["version"] = "999"
    else:
        value["derivation"]["packaging"]["output_wheel_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        artifact.validate(value)


@pytest.mark.parametrize("failure", ["metadata", "tag", "generator", "provenance", "record"])
def test_actual_archive_chain_refuses_metadata_record_and_provenance_drift(derived, monkeypatch, failure):
    files = derived["files"]
    if failure == "metadata":
        files[derived["new_root"] + "/METADATA"] = files[derived["new_root"] + "/METADATA"].replace(
            b"Summary: unchanged", b"Summary: changed")
    elif failure == "tag":
        files[derived["new_root"] + "/WHEEL"] = files[derived["new_root"] + "/WHEEL"].replace(b"cp310", b"cp311")
    elif failure == "generator":
        files[derived["new_root"] + "/WHEEL"] = files[derived["new_root"] + "/WHEEL"].replace(b"wheel 0.48.0", b"unknown tool")
    elif failure == "provenance":
        record = json.loads(files[artifact.PROVENANCE_MEMBER])
        record["recipe_identity_sha256"] = "f" * 64
        files[artifact.PROVENANCE_MEMBER] = json.dumps(record).encode()
    mutate = (lambda raw: raw.replace(b"sha256=", b"sha255=", 1)) if failure == "record" else None
    members = wheel(derived["wheel"], files, mutate)
    value = derived["value"]
    value["tilelang"]["installed_members"] = members
    value["tilelang"]["wheel"] = identity(derived["wheel"].read_bytes())
    value["derivation"]["embedded_provenance"].update(identity(files[artifact.PROVENANCE_MEMBER]))
    selected = selection(derived, monkeypatch)
    with pytest.raises(ValueError):
        selected.inspect(derived["installation"].env, derived["installation"].versions)


def test_v1_rejects_derivation_fields_and_v2_requires_retained_base(derived, monkeypatch):
    value = copy.deepcopy(derived["value"])
    value["schema"] = artifact.SCHEMA
    with pytest.raises(ValueError):
        artifact.validate(value)
    value.pop("derivation")
    with pytest.raises(ValueError, match="explicit v2"):
        artifact.validate(value)
    selected = selection(derived, monkeypatch)
    missing = artifact.OwnedArtifact(selected.attestation, selected.attestation_bytes,
                                    selected.attestation_sha256, selected.wheel, selected.hash_byte_credit)
    with pytest.raises(ValueError, match="native base wheel"):
        missing.inspect(derived["installation"].env, derived["installation"].versions)
