"""HTTP JSON output with source-owned fatal allocation containment.

Only the certified CUDA HTTP region installs the owner hooks. Other callers
retain ordinary json.dumps behavior. No library function is replaced.
"""
import json


def allocation_failed(handler, primary):
    """An eligible handler must not turn native ENOMEM into an encode retry."""
    if getattr(handler, "certified_json_payloads", False) and isinstance(primary, MemoryError):
        handler.close_connection = True
        handler.server.fail_json_encoding(handler, primary)
        raise primary


def dumps(handler, value, *, prefix="", suffix="", as_bytes=False):
    certified = getattr(handler, "certified_json_payloads", False)
    if certified:
        # Admission linearizes under the existing connection-owner lock. Calls
        # admitted before poison may overlap; the H-handler envelope pays all.
        handler.server.begin_json_encoding(handler)
    try:
        text = json.dumps(value)
        if prefix or suffix:
            text = prefix + text + suffix
        return text.encode("utf-8") if as_bytes else text
    except MemoryError as primary:
        allocation_failed(handler, primary)
        raise


def _envelope(app):
    catalog = getattr(getattr(app, "engine", None), "retention_catalog", None)
    envelope = getattr(catalog, "host_resource_envelope", None)
    return envelope if envelope is not None and envelope.get("http_owner_scope") is not None else None


def validate_connections(handler_type, actual):
    """Check the accepted-handler owner against its already admitted budget."""
    if getattr(handler_type, "certified_json_payloads", False):
        expected = handler_type.expected_http_connections
        if type(expected) is not int or expected <= 0 or type(actual) is not int or not 0 < actual <= expected:
            raise ValueError("HTTP connection owners exceed their admitted payload envelope")


def validate_server(server, app):
    """A supplied server must enforce the same certified App's admitted cap."""
    envelope = _envelope(app)
    handler = getattr(server, "RequestHandlerClass", None)
    owner = getattr(handler, "certified_http_owner", None)
    if envelope is not None or owner is not None:
        from tensorfold.server.http import Server

        if not isinstance(server, Server) or owner is not app:
            raise ValueError("HTTP server does not own the certified App handler")
        validate_connections(handler, server.max_connections)


def bind_handler(handler_type, app):
    """Bind the actual endpoint to its admitted CUDA payload owner."""
    envelope = _envelope(app)
    certified = envelope is not None
    prior = getattr(app, "_http_payload_failure", None)
    if certified and prior is not None:
        raise prior
    handler_type.certified_json_payloads = certified
    handler_type.expected_http_connections = None if envelope is None else envelope["max_http_connections"]
    handler_type.certified_http_owner = app if certified else None

    def notify_json_failure(self, primary):
        return app._fatal_http_payload(primary)

    handler_type.notify_json_failure = notify_json_failure
    return handler_type
