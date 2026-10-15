"""Request refusals and OpenAI's error object (``ErrorResponse``: message, type, param and code, the last two null
where nothing names them), shared by both servers' routes."""

RETRY_AFTER_S = 1           # a capacity refusal's Retry-After (whole seconds, at least 1, as OpenAI's 503 gives it)


class RequestError(ValueError):
    """A request refusal: HTTP 400, or a named error after a stream starts."""


class CapacityError(RequestError):
    """A transient capacity refusal: HTTP 503 with Retry-After, OpenAI's ``server_is_overloaded``."""

    code = "server_is_overloaded"


class ModelNotFound(RequestError):
    """A request naming a model this endpoint does not serve: HTTP 404, OpenAI's ``model_not_found``."""

    code = "model_not_found"
    param = "model"


class RoundError(RuntimeError):
    """A failed round's type and message, without its engine frames or exception payloads."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class ContextLengthError(RequestError):
    """A prompt and its reply past the context window: OpenAI's context_length_exceeded, which clients compact on."""

    code = "context_length_exceeded"


CONTEXT_LIMIT = "This server's maximum context length is"      # OpenAI's wording, which clients match to compact


def refusal(problem: str) -> RequestError:
    """A refusal string as its error: a context-window one carries OpenAI's code."""

    return (ContextLengthError if problem.startswith(CONTEXT_LIMIT) else RequestError)(problem)


def status_of(exc: BaseException) -> int:
    """The HTTP status a refusal answers with: 503 for capacity, 404 for an unknown model, else 400."""

    return 503 if isinstance(exc, CapacityError) else 404 if isinstance(exc, ModelNotFound) else 400


def retry_headers(status: int) -> tuple[tuple[str, str], ...]:
    """The extra headers of a reply with ``status``: Retry-After on a 503."""

    return (("Retry-After", str(RETRY_AFTER_S)),) if status == 503 else ()


def error_body(exc: BaseException, param: str | None = None) -> dict:
    """OpenAI's error object for a refusal. ``param`` names the field a context-length refusal is about (``messages``
    for a chat completion, ``prompt`` for a completion, as OpenAI names them); other refusals name their own field or
    none."""

    named = getattr(exc, "param", None) or (param if isinstance(exc, ContextLengthError) else None)
    return {"message": str(exc) or type(exc).__name__,
            "type": "service_unavailable_error" if isinstance(exc, CapacityError) else "invalid_request_error",
            "param": named, "code": getattr(exc, "code", None)}


def invalid_request(message: str, param: str | None = None) -> dict:
    """The error object of a request refused without a ``RequestError`` (a malformed body, an unknown route or stored
    response), ``param`` the field it is about."""

    return {"message": message, "type": "invalid_request_error", "param": param, "code": None}


def server_error(message: str) -> dict:
    """The error object of a failure that is the server's (HTTP 500, or a stream's error event)."""

    return {"message": message, "type": "server_error", "param": None, "code": None}


def unknown_route(method: str, path: str) -> dict:
    """The error object of a request for a route this server does not have (HTTP 404)."""

    return invalid_request(f"unknown path: {method} {path.split('?', 1)[0]}")
