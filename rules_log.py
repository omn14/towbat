"""One line per special rule that changes an outcome.

A rule that fires silently is indistinguishable from one that is not
implemented. Every rule in this engine is invisible on screen -- a Ward save
that works and a Ward save that was never coded look identical -- so the log is
the only way to tell, and it is what a bug report is written from.
"""

PREFIX = "[Rule]"


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
    print(f"{PREFIX} {rule} — {subject_name(subject)}: {detail}")


def rule_skipped(rule: str, subject, reason: str) -> None:
    """Report that *rule* could have applied but did not.

    Worth as much as the positive case: a rule that quietly declines looks
    exactly like a rule that is broken.
    """
    print(f"{PREFIX} {rule} — {subject_name(subject)}: not claimed ({reason})")
