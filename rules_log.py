"""One line per special rule that changes an outcome.

A rule that fires silently is indistinguishable from one that is not
implemented. Every rule in this engine is invisible on screen -- a Ward save
that works and a Ward save that was never coded look identical -- so the log is
the only way to tell, and it is what a bug report is written from.
"""

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import json

PREFIX = "[Rule]"
_context = {}
_scope = ContextVar('battle_log_scope', default={})


def set_log_context(**values):
    _context.update(values)


@contextmanager
def log_scope(**values):
    token = _scope.set({**_scope.get(), **values})
    try:
        yield
    finally:
        _scope.reset(token)


@dataclass(frozen=True)
class LogEntry:
    sequence: int
    category: str
    text: str
    subject: str
    details: str
    context: dict

    @property
    def group(self):
        return tuple(self.context.get(key) for key in ('round', 'player', 'phase', 'combat', 'initiative'))

    @property
    def heading(self):
        context = self.context
        parts = []
        if context.get('round') is not None:
            parts.append(f'Round {context["round"]} / Player {context.get("player", "-")}')
        if context.get('phase'):
            parts.append(context['phase'].removesuffix('Phase'))
        if context.get('combat'):
            parts.append(context['combat'])
        if context.get('initiative') is not None:
            parts.append(f'I{context["initiative"]}')
        return ' | '.join(parts) or 'Battle'


class BattleJournal:
    MODES = ('Summary', 'Rules', 'Debug')

    def __init__(self, limit=5000):
        self.entries = deque(maxlen=limit)
        self.sequence = 0

    def append(self, text, category='info', subject=None, details=''):
        self.sequence += 1
        entry = LogEntry(self.sequence, category, text, subject_name(subject) if subject else '',
                         details, {**_context, **_scope.get()})
        self.entries.append(entry)
        return entry

    def visible(self, mode='Summary', subject='All units'):
        excluded = {'rule', 'skip', 'dice', 'debug', 'detail'} if mode == 'Summary' else (
            {'dice', 'debug'} if mode == 'Rules' else set())
        return [entry for entry in self.entries if entry.category not in excluded
                and (subject == 'All units' or subject == entry.subject or subject in entry.text)]

    def export(self, *, structured=False):
        if structured:
            return json.dumps([asdict(entry) for entry in self.entries], indent=2)
        lines = []
        previous = None
        for entry in self.entries:
            if entry.group != previous:
                lines.append('\n' + entry.heading)
                previous = entry.group
            lines.append(f'[{entry.category}] {entry.text}')
            if entry.details:
                lines.append('  ' + entry.details.replace('\n', '\n  '))
        return '\n'.join(lines).lstrip()

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


def battle_log(text: str, category: str = 'info', *, subject=None, details='') -> None:
    """Post one line to the on-screen battle log, if there is one.

    Rules modules are imported by the tests without a ShowBase, so the Panda3D
    ``messenger`` builtin may not exist; engine code posts through here rather
    than reaching for it directly.
    """
    import builtins
    print(f'[{category.title()}] {text}')
    messenger = getattr(builtins, 'messenger', None)
    if messenger is not None:
        args = [text, category]
        if subject is not None or details:
            args.extend([subject, details])
        messenger.send('hud-log', args)


def dice_roll(values) -> None:
    """Publish the face values of a settled roll to the on-screen dice strip.

    Same messenger guard as ``battle_log``: ``dice`` is imported by the tests
    without a ShowBase.
    """
    import builtins
    messenger = getattr(builtins, 'messenger', None)
    if messenger is not None:
        messenger.send('hud-dice', [list(values)])


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
