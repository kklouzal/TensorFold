"""Loopback-only NCCL rendezvous for same-host expert ranks.

TCPStore's host argument chooses a client destination, not its listener bind.
Supply an already bound socket through its supported master_listen_fd API.
Ranks and other processes under the service account remain trusted peers;
loopback binding is not an authentication protocol.
"""
from __future__ import annotations

from datetime import timedelta
import ipaddress
import math
import os
import socket

from tensorfold.cleanup import raise_failures


class UnresolvedStoreOwner(RuntimeError):
    """A native constructor may have adopted/closed this FD: retire the rank process, never retry a numeric close."""

    def __init__(self, owner):
        super().__init__("TCPStore listener ownership is unresolved after native handoff; retire this rank process")
        self.owner = owner


class _Listener:
    def __init__(self):
        self.socket = self.store = self.descriptor = None
        self.phase = "python"

    def retire_before_handoff(self):
        """Close only a descriptor still proved exclusively Python-owned, once."""
        if self.socket is None:
            return
        try:
            if self.socket.fileno() != -1:
                self.socket.close()
                self.descriptor = None
            elif self.descriptor is not None:
                descriptor, self.descriptor = self.descriptor, None
                os.close(descriptor)
        except BaseException:
            self.phase = "unresolved"
            raise
        self.phase = "retired"


def _store_type():
    from torch.distributed import TCPStore

    return TCPStore


def open_store(master: str, port: int, world: int, rank: int, *, timeout_s: float):
    """Return an ordinary TCPStore whose server listens only on the selected loopback address.

    Call during rank-process startup, before publishing that rank as ready.
    Native FD adoption is not an exception-safe Python ownership transfer: if
    construction fails with no returned Store, the original exception carries
    UnresolvedStoreOwner. That outcome requires process containment, including
    cancellation, rather than closing a possibly recycled descriptor or retrying
    startup in that process. Successful stores retain ordinary native RAII.

    No worker barrier runs inside the constructor: the caller's bounded identity
    exchange admits every rank before starting model/native collectives.
    """
    if (type(world) is not int or not 1 <= world <= 4 or type(rank) is not int or not 0 <= rank < world
            or type(port) is not int or not 0 <= port <= 65535 or (world > 1 and port == 0)):
        raise ValueError("local expert store requires 1..4 ranks and a fixed valid port for multiple ranks")
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("local expert store requires a positive finite timeout")
    if type(master) is not str:
        raise ValueError("local expert store requires a loopback address or localhost")
    host = "127.0.0.1" if master in ("", "localhost") else master
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError("same-host expert rendezvous requires a numeric loopback address or localhost") from None
    if not address.is_loopback:
        raise ValueError("same-host expert rendezvous requires a loopback address")
    factory = _store_type()
    options = dict(timeout=timedelta(seconds=timeout_s), wait_for_workers=False, use_libuv=False)
    if rank:
        return factory(host, port, world, False, **options)
    owner = _Listener()
    try:
        owner.socket = socket.socket(socket.AF_INET6 if address.version == 6 else socket.AF_INET,
                                     socket.SOCK_STREAM)
        owner.socket.bind((host, port))
        owner.socket.listen()
        actual_port = owner.socket.getsockname()[1]
        # Publish before detach, so an interrupt after detach still leaves an
        # exclusively owned descriptor until native handoff is entered.
        owner.descriptor = owner.socket.fileno()
        owner.socket.detach()
        owner.phase = "native_handoff"
        owner.store = factory(host, actual_port, world, True,
                              master_listen_fd=owner.descriptor, **options)
        owner.descriptor = None  # the returned native Store now proves adoption
        owner.phase = "native"
        return owner.store
    except BaseException as primary:
        if owner.store is not None:
            # No caller could yet borrow it; dropping the returned native owner
            # is the supported retirement, never a second close of its raw FD.
            owner.store = None
            owner.descriptor = None
            owner.phase = "retired"
            raise
        if owner.phase == "native_handoff":
            owner.phase = "unresolved"
            try:
                failures = [UnresolvedStoreOwner(owner)]
            except BaseException as diagnostic:
                # Retain this frame's owner even if diagnostic allocation fails;
                # the original startup failure remains the authoritative error.
                raise primary from diagnostic
            raise_failures(primary, failures)
        try:
            owner.retire_before_handoff()
        except BaseException as cleanup:
            try:
                failures = [cleanup, UnresolvedStoreOwner(owner)]
            except BaseException as diagnostic:
                raise primary from diagnostic
            raise_failures(primary, failures)
        raise
