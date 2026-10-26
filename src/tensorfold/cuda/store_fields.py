"""Shared bounded keys for the owned retained-state Store protocol."""

MAX_KEY_BYTES = 8192  # Includes TCPStore's native '/' prefix on either backend.


def validate_store_key(key: str) -> None:
    """Reject before encoding or native IO can exceed the protocol envelope."""
    if type(key) is not str:
        raise ValueError("retained transport key field must be a string")
    # Every Unicode character needs at least one UTF8 byte. This first bound
    # also caps the temporary encoding used for the exact byte-length check.
    if len(key) >= MAX_KEY_BYTES or len(key.encode("utf-8")) + 1 > MAX_KEY_BYTES:
        raise ValueError("retained transport key field exceeds the 8192-byte protocol limit")
