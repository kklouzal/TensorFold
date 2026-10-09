"""Primary-preserving transport for completed owned cleanup operations.

Callers capture native cause/context before foreign cleanup and retire all
required work before entry. Diagnostic allocation failures preserve the primary
and retain status references in its native dictionary or traceback. Retained
failure payloads are operation-owned and read-only to consumers.
"""
from __future__ import annotations


def error_name(error):
    """Bounded actual class name without invoking exception metaclass hooks."""
    return str.__getitem__(type.__dict__['__name__'].__get__(type(error)), slice(None, 64))


def raise_grouped(primary, causes):
    """Group owned statuses, preserving the primary if allocation fails.

    The fallback list is operation-owned diagnostic state; callers must treat
    it as read-only. Grouping/retention errors become the explicit primary cause.
    No resource work happens here, and callbacks/retirement precede this call.
    """
    try:
        dictionary = BaseException.__dict__['__dict__'].__get__(primary, type(primary))
        dict.__setitem__(dictionary, '_tensorfold_retained_failures', causes)
    except BaseException as retention:
        if retention is primary:
            raise primary
        raise primary from retention
    try:
        grouped = BaseExceptionGroup('owned cleanup and retained native failures', causes)
    except BaseException as allocation:
        if allocation is primary:
            raise primary
        raise primary from allocation
    dict.__delitem__(dictionary, '_tensorfold_retained_failures')
    raise primary from grouped


def _transport(primary, errors):
    if primary is None and errors:
        primary, errors = errors[0], errors[1:]
    if primary is None:
        return
    previous = (BaseException.__cause__.__get__(primary), BaseException.__context__.__get__(primary))
    try:
        dictionary = BaseException.__dict__['__dict__'].__get__(primary, type(primary))
        retained = dict.get(dictionary, '_tensorfold_retained_failures')
        if retained is not None and type(retained) is not list:
            raise ValueError('owned retained failure payload must remain a list')
        retained = () if retained is None else tuple(retained)
    except BaseException as retention:
        if retention is primary:
            raise primary
        raise primary from retention
    others = []
    for error in errors:
        if error is not primary and all(error is not previous for previous in others):
            others.append(error)
    annotations = []
    annotation_failed = False
    messages = ['owned cleanup also failed (' +
                error_name(error) + ')'
                for error in others[:8]]
    if len(others) > 8:
        messages.append('additional owned failure count: ' + str(len(others) - 8))
    for message in messages:
        try:
            BaseException.add_note(primary, message)
        except BaseException as annotation:
            annotation_failed = True
            if annotation is not primary and all(annotation is not error for error in [*others, *annotations]):
                annotations.append(annotation)
            break
    causes = []
    if others or annotations:
        for error in (*retained, *previous):
            if error is not None and error is not primary and all(error is not item for item in causes):
                causes.append(error)
    for error in [*others, *annotations]:
        if all(error is not previous for previous in causes):
            causes.append(error)
    if len(causes) == 1 and not annotation_failed:
        raise primary from causes[0]
    if causes:
        raise_grouped(primary, causes)
    raise primary


def raise_failures(primary, errors):
    """Preserve primary identity even if cold diagnostic preparation exhausts.

    Required work has retired before entry. These original status references
    and native fields remain traceback-frame-owned if labels, annotations,
    retained-payload snapshots or group preparation itself cannot allocate.
    This fallback reports that distinct transport failure as the explicit
    cause; it does not promise allocation succeeds or a complete group forms.
    """
    if primary is None:
        if not errors:
            return
        primary = errors[0]
    native_cause = BaseException.__cause__.__get__(primary)
    native_context = BaseException.__context__.__get__(primary)
    try:
        _transport(primary, errors)
    except BaseException as failure:
        if failure is primary:
            raise
        # Keep the named native/status references in this traceback frame.
        # The guards do not format exceptions or allocate a new status payload.
        if native_cause is primary:
            native_cause = None
        if native_context is primary:
            native_context = None
        raise primary from failure


def rollback(owner, primary, close):
    """Close a partial constructor; unresolved ownership stays with its error.

    The caller must retain that error/owner until recovery or process teardown.
    A failed native retirement is never a successful rollback. Diagnostic
    retention failure preserves the owner in the primary traceback instead.
    """
    prior = (BaseException.__cause__.__get__(primary), BaseException.__context__.__get__(primary),
             BaseException.__suppress_context__.__get__(primary))
    errors = []
    try:
        close()
    except BaseException as failure:
        errors.append(failure)
        try:
            namespace = BaseException.__dict__['__dict__'].__get__(primary)
            retained = dict.get(namespace, '_tensorfold_retained_owners', [])
            if type(retained) is not list:
                raise ValueError("retained constructor owner payload must remain a list")
            if all(owner is not previous for previous in retained):
                retained = [*retained, owner]
            dict.__setitem__(namespace, '_tensorfold_retained_owners', retained)
        except BaseException as retention:
            errors.append(retention)
    if errors:
        errors.extend(root for root in prior[:2] if root is not None and root is not primary)
    else:
        BaseException.__cause__.__set__(primary, prior[0])
        BaseException.__context__.__set__(primary, prior[1])
        BaseException.__suppress_context__.__set__(primary, prior[2])
    raise_failures(primary, errors)


def finish(operations):
    """Attempt independent retirement operations, retaining the first roots.

    Capturing fields before the next callback prevents cleanup from rewriting
    a previously formed operation failure. Dependent resource release remains
    the caller's responsibility and follows only successful return.
    """
    primary, prior, errors = None, None, []
    for operation in operations:
        try:
            operation()
        except BaseException as error:
            if primary is None:
                primary = error
                prior = (BaseException.__cause__.__get__(primary), BaseException.__context__.__get__(primary),
                         BaseException.__suppress_context__.__get__(primary))
            else:
                errors.append(error)
    if primary is not None:
        BaseException.__cause__.__set__(primary, prior[0])
        BaseException.__context__.__set__(primary, prior[1])
        BaseException.__suppress_context__.__set__(primary, prior[2])
    raise_failures(primary, errors)
