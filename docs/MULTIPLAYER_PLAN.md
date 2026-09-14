# Online Multiplayer Plan

Speculative design for a possible future implementation. Nothing here is built.
Survey performed 2026-09-14 against the tree at that date; the line references
are anchors for orientation, not guarantees they have not moved since.

Scope: two human players, one battle, over the internet. Not an MMO, not a
lobby service, not matchmaking, not spectators.

## Why This Is Tractable

The game is turn-based and decision-driven. There is no real-time simulation to
keep in step, so none of the usual netcode machinery applies: no interpolation,
no client-side prediction, no rollback, no tick rate. A round trip of several
hundred milliseconds is invisible to a player who is deciding whether to charge.

Two findings decide the architecture.

**The engine already has a decision router.** `game.aiControls(unit)` at
`game.py:2968` answers "whose decision is this?" by unit ownership rather than
by whose turn it is, precisely because charge reactions, Break tests and pursuit
choices fall to the player who is not currently moving. Every player decision in
the game then funnels through one async suspension point,
`game.makeChoiceNew(choices, position, ..., owner=unit)` at `game.py:2983` —
60 call sites across 25 modules. That function is a ready-made network seam.

**The engine already serialises its whole state.** `persistence.save_game_state`
and `persistence.load_game_state` cover phase, round, current player, charge
stage and declarations, deployment stage, per-unit position/heading/wounds/buffs
/spells/combat locks, spells in play, challenges and terrain. A save is a
complete description of a battle in progress, which is exactly what a joining or
resynchronising peer needs.

## Why Lockstep Is Rejected

The obvious design — both machines run the same simulation from the same seed
and exchange only inputs — does not survive contact with this codebase.

- 314 direct `random.*` calls across 62 files as surveyed, roughly 27 of them
  production modules. There is no central RNG layer. `models.py:119` and
  `rulesFunctions.py:19` are partial dice helpers, but most modules roll
  directly.
- Rules logic and rendering are interleaved. `combat_resolution.py` and
  `psychology.py` import `LerpPosInterval`/`Parallel` and run charge, flee and
  rally animations inline with the state mutation they accompany.
- Lockstep requires every client to make an identical number of RNG calls in an
  identical order, forever, including inside animation code. One frame-rate
  dependent `random.uniform` in a particle effect desynchronises the battle.

Chasing determinism here means rewriting the rules layer. It is not worth it for
two players.

## Recommended Architecture: Host-Authoritative, Decision-Relayed

One machine (the host) runs the entire rules engine unchanged. It performs every
die roll and owns the only true game state. The guest runs a full client for its
own camera and rendering, but its rules engine never decides anything.

Two message families cover nearly the whole protocol:

1. **Decision request/response.** When the host reaches `makeChoiceNew` for a
   unit the remote player owns, it serialises the choices and prompt, sends
   them, and awaits. The guest raises the same `Choice` UI locally and returns
   an index. In the host this is a third branch alongside the existing AI
   auto-answer branch, plus a `remoteControls(unit)` sibling to `aiControls`.
2. **State sync.** The host broadcasts the outcome; the guest applies it.

Because only the host rolls dice, the RNG problem disappears rather than being
solved.

### Transport

Panda3D's built-in networking is a good fit, not for its features but because it
is poll-based and non-blocking: `QueuedConnectionManager`,
`QueuedConnectionListener`, `QueuedConnectionReader`, `ConnectionWriter` and
`NetDatagram` from `panda3d.core`, with `PyDatagram`/`PyDatagramIterator` from
`direct.distributed`. A single `taskMgr.add(pollNetwork, 'netPoll')` coexists
with the existing `await taskMgr.add(...)` coroutine style. Introducing an
asyncio event loop beside the ShowBase task manager would fight the engine.

Avoid the `direct.distributed` Distributed Object system. It is the Toontown MMO
stack and expects an Astron-style server cluster and DC definition files.

Payloads are JSON inside the datagram. Never pickle from the wire.

## Threat Model and Authority Placement

The authority in the design above is one of the two players' machines, so that
player can cheat. The question of whether to move authority to a neutral server
is worth answering explicitly, because the answer is less obvious than it looks.

### The netcode does not change

Host-authoritative and server-authoritative are the same architecture: one
process runs the rules and owns state, others send intents and receive results.
Whether that process belongs to a player or to a rented machine is a deployment
question, not a protocol one. The extra cost of a server sits elsewhere:

- **Running the engine headless.** The authority needs the full rules engine, and
  `combat_resolution.py` and `psychology.py` run animations inline with the state
  mutation they accompany. Panda3D's `window-type none` gives a scene graph and
  Bullet physics with no graphics context, and `bodyNP` positions are CPU-side,
  so this is a shorter hop than it sounds — but it needs proving, not assuming.
- **Operations.** A machine to keep running and deploy to, and version matching
  across three processes instead of two.
- **What is lost.** LAN and same-machine play stop working once authority is
  remote, and both players pay latency where the host previously paid none.

### The cheats are not equivalent

- **Illegal actions** — overlong moves, charges out of arc, a spell cast twice.
  Server authority is irrelevant here. This is fixed by the authority validating
  incoming commands, which Stage 2 requires regardless of where it runs. An
  authority that trusts `{unit_id, target_pos}` unchecked is exploitable wherever
  it lives.
- **Hidden information** — secret objectives, pre-battle rosters, spell
  selection. Server authority helps only if messages are also filtered per
  recipient. A neutral server broadcasting full snapshots leaks exactly as much
  as a player-host does. Orthogonal problem; see LEFTOVER.
- **Dice manipulation** — the only cheat host authority genuinely cannot stop.
  The host rolls, so the host can reroll until satisfied. Not hypothetical:
  `debug_tools.set_loaded_dice` (`debug_tools.py:519`) already replaces
  `random.randint` process-wide, and reaching it needs only `WH_DEBUG` in the
  environment.

### Commit-reveal instead of a server

The dice gap closes without a third machine. For a roll, both sides contribute
entropy: each commits a hash of a nonce, both reveal once both hashes are in,
each verifies the revealed nonce against the hash it was given, and the result
derives from the two nonces combined. Neither side can steer the outcome,
because committing happens before seeing the opponent's contribution and the
hash prevents changing your mind afterwards. Batch per phase rather than per die
and the round trips are negligible for a turn-based game.

This does not stop a modified client lying about its inputs, but validation
handles that and validation is required either way.

### Position

Do not build a server. Validate properly, and add commit-reveal dice if the roll
problem matters. Spend nothing on hosting.

The part worth doing now costs almost nothing: **treat authority as a role, not a
machine.** Never let "the authority is player 1", "the authority renders" or "the
authority owns the camera" enter the protocol, and keep the authoritative side
reachable through the same interface as the headless peer described under
Testing. Done that way, moving authority to a dedicated server later is a
deployment change. Skipped, it is a rewrite — which makes this a decision that
can be deferred only if it is not quietly foreclosed in the first week.

## Stages

**Stage 0 — Transport spike.** Two processes on localhost exchanging a
handshake, proving the poll task coexists with the render loop and the Bullet
world. Disposable.

**Stage 1 — Snapshot MVP.** The real milestone. The host resolves everything and
at each FSM phase boundary sends the full `save_game_state` payload plus the
accumulated `rules_log` output; the guest applies it through `load_game_state`.
Units teleport and nothing animates on the guest, but the game is playable, and
it reuses `persistence.py` almost unmodified. The `makeChoiceNew` relay goes in
here, so the guest genuinely makes its own charge reactions, Break tests and
pursuit calls.

**Stage 2 — Intent layer.** Guest movement has nowhere to go at Stage 1: mouse
picking and drag-to-position live in `taskMoveUnit`/`pathTowardsMouse` on the
host. Add a serialisable command — unit id, target position, target heading —
produced by the guest's local mouse picking and validated by the host before
execution. Host-side validation is not optional; a client claiming a 40" move
must be rejected, not trusted.

**Stage 3 — Animation parity.** Replace whole snapshots with event deltas, and
stream dice results alongside them so the guest re-runs the same code path with
rolls fed from the wire. This needs a roll-source indirection rather than an
attempt to seed 27 modules. `debug_tools.set_loaded_dice` (`debug_tools.py:519`)
already proves engine-wide interception of `random.randint` works and is the
natural prototype for it.

Keep the Stage 1 snapshot path as the reconciliation fallback. Compare a state
checksum at each phase boundary; on mismatch the host pushes a full snapshot and
the guest reloads. Divergence then degrades to a visual hiccup instead of a
desync.

**Stage 4 — Around the game.** Lobby, roster exchange, reconnect, deployment.

## Known Hazards

- **Hung games on disconnect.** `makeChoiceNew` ends in `await cyn.ma`, which
  waits forever. Every remote choice needs a timeout with a defined fallback —
  the host answering on the guest's behalf, as the AI branch already does with
  `next(iter(cyn.choices))`. This belongs in Stage 1. A netcode without timeouts
  hangs on its first real test.
- **Simultaneous deployment.** `deployPhase.py` is the heaviest RNG user and
  deployment is conceptually parallel. Serialise it as alternating unit-by-unit
  drops rather than running both players at once.
- **Roster ingest is untrusted input.** `models.py` uses `requests` to fetch army
  JSON from URLs. A guest-supplied roster needs schema validation, size caps, and
  no host-side URL fetching on the guest's behalf (SSRF). No file paths from the
  wire.
- **Stable unit identity.** Parity checking and movement commands need an id that
  survives serialisation. `NodePath` object identity will not do, and list index
  will not either — units die and lists reorder. Assign an id at creation and
  carry it through `persistence.py`, in the first commit. Retrofitting identity
  later is miserable.
- **Cheating.** A host who is also a player can cheat, and dice are the cheat
  that design cannot prevent. See Threat Model and Authority Placement.
- **Rule logging must cross the wire.** The project requires every rule that
  fires to announce itself, and that console output is what bug reports are
  written from. If `rules_log` lines do not reach the guest, half the debugging
  ability disappears in multiplayer — which is when it is most needed.

## Testing

Netcode needs many fast, nasty edge-case tests, but `tests/conftest.py` enforces
one module per process because Panda3D global state is not isolated, and a real
host/guest pair is therefore always two processes. Resolving that tension is a
design decision, not a test-harness detail.

### Build a headless peer first

The highest-value component is a peer that speaks the full protocol with no
Panda3D imports at all — no `ShowBase`, no `NodePath`, no Bullet. It holds the
abstract state dict, applies deltas, answers choice requests from a script, and
validates commands.

It pays off three times: most tests become single-process and fast; the protocol
layer is forced to stay free of scene-graph coupling; and it is the seed of a
dedicated server if one is ever wanted. If the headless peer cannot be written,
the protocol has leaked rendering concerns and the failure to compile is the
message.

### Layers

**L1 — Codec and validation.** Pure functions, no fixtures, milliseconds. Round-
trip every message type, then the hostile half: truncated datagrams, wrong field
types, unknown message ids, choice index `-1` or `999`, nonexistent unit ids, a
50 MB roster, an illegal move distance. Every one must be rejected without
crashing the host.

**L2 — The workhorse.** Real `MyApp` offscreen, using the existing
`loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')`
pattern from the scene tests, with `remoteControls()` forced true for player 2
and a `FakeTransport` delivering messages in memory. Covers: a charge reaction
routed to the remote player; the remote player answering during the host's turn;
a malformed answer mid-`await`; two answers for one prompt; an answer arriving
after its prompt timed out. `FakeTransport` needs knobs for delay, drop,
duplicate and reorder — deterministic fault injection beats a real flaky network
because a failure reproduces.

**L3 — Two processes, sparingly.** Only to prove the real socket path works:
connect, exchange rosters, play a turn, disconnect cleanly. Bind the host to port
0 and print the assigned port for the guest to read; fixed ports produce mystery
failures when a previous run leaks a socket.

**L4 — Replay parity oracle.** The regression net that stops multiplayer rotting
as rules are added. Record a solo game as a journal of commands, dice consumed
and `rules_log` lines. Replay it through host and guest, comparing a state
checksum — canonical JSON of `save_game_state` with volatile fields stripped,
sorted keys, SHA-256 — at every phase boundary.

On mismatch, diff the `rules_log` streams rather than failing on coordinates. A
report reading

```
- Swiftstride — Reavers: adds +3" to the maximum charge range (12" -> 15")
+ (absent on guest)
```

is diagnosable in minutes. `expected y=14.8331, got y=11.9997` is a weekend. The
project's logging discipline is what makes this possible.

Keep a corpus of journals in the repo, one per interesting scenario — a charge
with a flee reaction, a failed Break test with pursuit, a miscast, a challenge.
The corpus is the asset; the runner around it is trivial.

### Harness gotchas

- `run_tests_isolated.py` caps each module at 1536 MiB and the cap covers child
  processes. An L3 test spawning two full engines inside that cgroup will be
  OOM-killed. Either raise the cap for the netplay module or keep L3 to two
  minimal scenes. Another reason to push volume into L2.
- Do not wait for real timeouts. Use the established fast-forward pattern:
  isolated `TaskManager` plus `ClockObject.MNonRealTime`, stepping tasks,
  `ivalMgr` and events manually, patching the concrete `LerpPosHprInterval` and
  `Parallel` `__await__` rather than `Interval.Interval`.
- Async `makeChoiceNew` test doubles must accept the positional world-position
  argument. An incompatible mock inside a Panda task presents as a hundred-second
  combat timeout rather than a `TypeError`.
- Define the checksum's canonical field set explicitly. Camera position, particle
  state and anything time-derived differ between host and guest legitimately; an
  exclusion list will leak phantom desyncs.

## Smallest Useful Starting Point

Add `remoteControls(unit)` beside `aiControls`, and a third branch in
`makeChoiceNew` that sends the choice over the wire instead of awaiting the
mouse. With a hardcoded localhost connection and full-snapshot sync at phase
boundaries, that alone puts two humans in one battle. Everything after it is
making the experience bearable.

Write `FakeTransport` and the headless peer before any socket code. Starting with
`QueuedConnectionManager` means debugging netcode and rules logic simultaneously
through a socket, which is the slowest possible way to discover that charge
reactions are asking the wrong player.

## LEFTOVER

Not decided, and deliberately so:

- Hidden information. The Old World is largely open, but pre-battle roster
  secrecy, secret objectives (`battle_secondary.py`) and spell selection are not.
  A snapshot-sync host sends the guest everything, including anything meant to be
  private. Stage 1 leaks; Stage 4 must decide what is filtered per recipient.
- Whether commit-reveal dice are worth building at all. Two friends who trust
  each other need none of it, and the loaded-dice cheat already exists offline.
- Campaign play (`campaignMap.py`) and the tutorial are out of scope entirely.
- Reconnect semantics. A save is the obvious rejoin payload, but who waits, for
  how long, and what happens to an abandoned game is unspecified.
- Whether Stage 3 is worth building at all. Stage 1 plus Stage 2 may be a
  perfectly good game that simply teleports units on the guest's screen.
