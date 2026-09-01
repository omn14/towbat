"""One line per special rule that changes an outcome.

A rule that fires silently is indistinguishable from one that is not
implemented. Every rule in this engine is invisible on screen -- a Ward save
that works and a Ward save that was never coded look identical -- so the log is
the only way to tell, and it is what a bug report is written from.
"""

PREFIX = "[Rule]"

# Anything that wants to display the trace as well as print it — the on-screen
# battle log, a test harness — registers here.
_listeners = []


def add_listener(listener) -> None:
    """Register ``listener(kind, rule, subject_name, detail)``.

    *kind* is 'fired' or 'skipped'.
    """
    if listener not in _listeners:
        _listeners.append(listener)


def remove_listener(listener) -> None:
    if listener in _listeners:
        _listeners.remove(listener)


def _emit(kind: str, rule: str, subject: str, detail: str) -> None:
    for listener in list(_listeners):
        try:
            listener(kind, rule, subject, detail)
        except Exception as exc:
            # A broken display must not stop a rule from resolving.
            print(f"{PREFIX} listener {listener!r} failed: {exc}")


def subject_name(subject) -> str:
    """A readable name for a unit wrapper, a Unit, a model or a plain string."""
    if subject is None:
        return "-"
    if isinstance(subject, str):
        return subject
    for path in (('unit', 'name'), ('name',), ('unitName',)):
        value = subject
        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break
        if isinstance(value, str) and value:
            return value
    return str(subject)


def rule_log(rule: str, subject, detail: str) -> None:
    """Report that *rule* changed something for *subject*.

    *detail* should carry the numbers that decided it and what they changed,
    so the line answers "why did that happen?" without a re-run.
    """
    name = subject_name(subject)
    print(f"{PREFIX} {rule} — {name}: {detail}")
    _emit('fired', rule, name, detail)


def rule_skipped(rule: str, subject, reason: str) -> None:
    """Report that *rule* could have applied but did not.

    Worth as much as the positive case: a rule that quietly declines looks
    exactly like a rule that is broken.
    """
    name = subject_name(subject)
    print(f"{PREFIX} {rule} — {name}: not claimed ({reason})")
    _emit('skipped', rule, name, reason)
