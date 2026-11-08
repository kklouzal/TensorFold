"""HTTP JSON output with source-owned fatal allocation containment.

Only the certified CUDA HTTP region installs the owner hooks. Other callers
retain ordinary json.dumps behavior. No library function is replaced.
"""
import json
import sys


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
        if certified:
            retain = getattr(handler, "retain_host_error", None)
            active_error = sys.exception()
            if retain is not None and active_error is not None:
                retain(active_error)
            reserve = getattr(handler, "reserve_host_encoding", None)
            if reserve is not None:
                reserve(value, prefix=prefix, suffix=suffix)
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
        pool = getattr(app, "host_payload_pool", None)
        if pool is not None:
            pool.capacity_changed = server._capacity_changed.set


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
    pool = getattr(app, "host_payload_pool", None)
    if pool is not None:
        from .host_payload import current, handler_scope, keep_error

        if getattr(handler_type, "_host_payload_pool", None) is not None:
            raise ValueError("HTTP handler payload owner is already bound")
        original = handler_type.handle_one_request
        original_get = handler_type.do_GET
        original_post = handler_type.do_POST
        original_parse = handler_type.parse_request

        def handle_operation(self):
            from .errors import CapacityError
            from .cancellation import RequestCancelled, socket_cancellation
            from tensorfold.cleanup import raise_failures

            try:
                with handler_scope(app) as owner:
                    cancellation = socket_cancellation(self.connection, stopping=self.server.stopping)
                    owner.cancelled = lambda: cancellation.cancelled
                    try:
                        return original(self)
                    except RequestCancelled as error:
                        # Expected peer/stop cancellation is local. Its frames
                        # may still retain paid body/header roots after return.
                        keep_error(error, lease=owner)
                        self.close_connection = True
                        return
            except BaseException as primary:
                if isinstance(primary, CapacityError) and current() is None and pool.failure is None:
                    # A keepalive may outlive the last fixed header slot. Do
                    # not parse unpaid headers or turn ordinary pressure fatal.
                    self.close_connection = True
                    return
                # Unknown retirement/credit failures fence acceptance, just as
                # native encoder ENOMEM does. No later request borrows its slot.
                try:
                    self.server.fail_json_encoding(self, primary)
                except BaseException as cleanup:
                    raise_failures(primary, [cleanup])
                raise

        def parse_operation(self):
            result = original_parse(self)
            if result:
                fields = self.headers._headers
                if any(type(name) is not str or type(value) is not str for name, value in fields):
                    raise ValueError("HTTP header owner differs from selected raw-header schema")
                extent = len(self.raw_requestline) + sum(len(name) + len(value) + 4 for name, value in fields)
                owner = current()
                owner.retired_handler_bytes = pool.header_bound(extent, len(fields)) + pool.header_other_bytes
            return result

        def begin_body(self):
            owner = current()
            if owner is None or owner.pool is not pool:
                raise RuntimeError("HTTP body lost its actual handler payload owner")
            return owner.begin_body()

        def reserve_encoding(self, value, *, prefix="", suffix=""):
            owner = current()
            if owner is None or owner.pool is not pool:
                raise RuntimeError("HTTP encoder lost its actual handler payload owner")
            return owner.encode(value, prefix=prefix, suffix=suffix)

        def get_operation(self):
            from .errors import CapacityError, error_body
            path = self.path.split("?", 1)[0].rstrip("/")
            owner = current()
            monitor = path in ("/stats", "/v1/stats", "/v1/runtime")
            if monitor:
                owner.control_kind = "monitor"
            try:
                owner.control(0 if monitor else pool.metadata_wire_bytes)
            except CapacityError as error:
                BaseException.__traceback__.__set__(error, None)
                try:
                    return self._json(503, {"error": error_body(error)})
                finally:
                    # Pure credit refusal before lookup/decoder/snapshot work.
                    BaseException.__traceback__.__set__(error, None)
            try:
                return original_get(self)
            except CapacityError as error:
                keep_error(error, lease=owner)
                return self._json(503, {"error": error_body(error)})

        def post_operation(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            owner = current()
            owner.control_kind = "detokenize" if path in ("/detokenize", "/v1/detokenize") else "metadata"
            owner.control_required = path not in (
                "/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions")
            return original_post(self)

        handler_type._host_payload_pool = pool
        handler_type.handle_one_request = handle_operation
        handler_type.parse_request = parse_operation
        handler_type.begin_host_body = begin_body
        handler_type.reserve_host_encoding = reserve_encoding
        handler_type.retain_host_error = lambda self, error: keep_error(error)
        handler_type.do_GET = get_operation
        handler_type.do_POST = post_operation
    return handler_type
