"""Bounded fake listener ownership for CLI option-forwarding tests.

The CLI constructs the listener, passes that exact owner to ``serve``, then
closes it. These fixtures acquire no sockets and preserve that lifetime.
"""

import threading


def install_cli_server(monkeypatch, request, module, *, on_serve=None):
    owners, served = [], []

    class Server:
        def __init__(self, address, handler, *, max_connections=None):
            self.address, self.handler = address, handler
            self.max_connections = max_connections
            self.stopping = threading.Event()
            self.close_calls = 0
            owners.append(self)

        def server_close(self):
            self.close_calls += 1
            self.stopping.set()

    def serve(app, host, port, *, server):
        assert any(server is owner for owner in owners)
        assert server.address == (host, port)
        assert server.close_calls == 0 and not server.stopping.is_set()
        assert server.max_connections is None or type(server.max_connections) is int
        served.append(server)
        if on_serve is not None:
            on_serve(app)

    def retired():
        assert len(owners) == len(served)
        assert all(owner is callback_owner for owner, callback_owner in zip(owners, served, strict=True))
        assert all(owner.close_calls == 1 and owner.stopping.is_set() for owner in owners)

    monkeypatch.setattr(module, "Server", Server)
    monkeypatch.setattr(module, "serve", serve)
    request.addfinalizer(retired)
