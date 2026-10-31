# Guarded controller IO

`tools/gb10_controller.py` retains the finite controller protocol and command
schemas from its recorded predecessor. This source change affects only the
shared IO imports, durable JSON publication, and `Host.command` resource/error
handling. It does not authorize a controller, Docker, model, GPU or service
lifecycle action.

`Host.command` keeps the inherited descriptors, one client launch, bounded
stdout/stderr, selector loop, command deadline and five-second failure-retirement
wait. Selector retirement, client termination/wait, both pipe closes, metadata
hashing and command publication report independent failures. The original
command failure remains the raised object; the canonical cleanup transport
retains secondary failures and the original native cause/context. If diagnostic
assembly itself exhausts, raw statuses remain owned by the traceback frame.

Client or IO failure leaves `complete` false and the host uncertain. A failed
wait retains the pending client and does not establish retirement. A successful
lifecycle command discharges its uncertainty only after its result capture,
pipe retirement and publication succeed. The controller does not retry a
command or a failed close/publication.

Publication retains the original UTF-8 JSON formatting, trailing newline,
exclusive `.new` file, flush, file `fsync`, mode `0600`, atomic replace and
directory `fsync`. The shared `gb10_io` owners preserve write/fsync errors when
close also fails. Failed partial files remain diagnostic evidence. Failure
after replace can leave final bytes present while directory durability remains
unproved; file presence alone never changes the owning failure outcome.

The helper imports the exact project-owned `tensorfold.cleanup` implementation.
Hash-bound packet consumers must supply the shared helper and its authenticated
cleanup import closure before loading the controller snapshot. They must not
patch an installed controller or substitute a host subclass.

The maintained host-only check is:

```sh
python3 -B -m pytest tests/test_gb10_controller_io.py -q
```

The check uses subprocess/selector doubles and isolated temporary files; it
does not call Docker or a GPU. All **15 cases passed** in the combined 58-case
host run on Linux CPython 3.14.4 with pytest 9.0.2. The other
controller lifecycle bodies remain unchanged and gain no new qualification from
this IO fix. Actual packet binding, host checks, profile execution and retirement
verification remain separate requirements. No model-quality or performance
result is claimed.
