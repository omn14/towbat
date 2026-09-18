# Game AI Revamp Plan

Date: 2026-09-18.
Status: Phase 2 exit gate met for the agreed acceptance matrix. The live utility
baseline is implemented and verified; complete observations, predictive modelling,
search correctness, playing-strength benchmarks, and release gates remain open.

## Goal And Scope

Build a dependable, rules-fair opponent that completes battles without manual
intervention, pursues the actual victory conditions, coordinates its army, and
makes defensible decisions under uncertainty within a desktop compute budget.

The immediate target is a strong, explainable baseline, followed by selective
chance-aware search. A perfectly detailed simulator is not required. Neither
deeper search nor machine learning compensates for incorrect state or actions.

- Support human versus AI on either side and AI versus AI using the same policies.
- Cover implemented standard battles and supported Battle March configurations,
  including the objective-free Reed Fens preset and different battlefield shapes.
- Reuse the existing rules and geometry incrementally. Do not rewrite the entire
  engine, duplicate rule effects, or require rendering during simulated play.
- Respect supported-rule limits; report missing predictive support rather than
  silently treating an important rule as having no effect.
- Preserve existing human controls, saves, rosters, and rule behavior.
- The user authorized implementation through the Phase 2 exit gate, selected
  standard battles, objective Battle March and objective-free Reed Fens, and
  chose continuous play while AI is enabled. Representative test forces were
  delegated to the implementer. Staging, commits and pushes remain unauthorized.

## Reviewed Baseline

The 2026-09-18 review traced the live controller and ran focused runtime probes.
These are historical findings before the first implementation slice below.

| Area | State At Review | Consequence |
| --- | --- | --- |
| Live controller | [game.py](../game.py) constructs `EnhancedAI`, depth 19, disabled by default; `a-up` steps a phase | No complete autonomous battle driver |
| Search routing | [aiMinimaxIntegration.py](../aiMinimaxIntegration.py) adds 3 to the strength ratio before checking 0.7 to 1.4 | Outside Combat, minimax is never selected for nonnegative model counts |
| Snapshot | [gameStateTree.py](../gameStateTree.py) uses player-list membership, local transforms, raw Movement, and the main phase index | Joined P1 characters become P2; mounted Movement 9 became 4 in a probe; Reserve Move appears as Shooting |
| Charge policy | Nearest-enemy declaration attempts run before strategic decisions | Army doctrines do not control charge commitment |
| Forward model | Fixed movement, proximity-based engagement, flat damage, two moves exhaust both armies, no round increment | Search evaluates materially different consequences |
| Search correctness | [minimaxOptimizations.py](../minimaxOptimizations.py) omits relevant cache fields, misclassifies bounds, and flips perspective during same-player quiescence | Cached values and tactical search cannot be trusted; quiescence lacks deadline/depth guards |
| Controller recovery | Missing ownership guard outside deployment, unconditional tree visualization, no `finally` unlock | Wrong-side phase advance; `None` actions and heuristic-only mode can leave the controller locked |
| Decision coverage | No general casting, normal rallying, or objective-seeking actions in the main planner | Rule-local automatic choices are not a complete turn policy |
| Alternative search | MCTS implementation is a stub | A one-iteration call fails; it must not be exposed as a working mode |
| Evaluation | Archetype weights and positioning heuristics, not actual scoring | Reported heuristic percentages are not calibrated win or break probabilities |

The real round counter is zero-based. The verified endgame defect is missing
simulated round advancement, not an early cutoff on the final playable turn.
Classifier caches also retain outdated classifications when statistics change
under the same unit name.

The two existing `EnhancedAI` integration cases in
[tests/test_charge_declarations_scene.py](../tests/test_charge_declarations_scene.py)
and [tests/test_drilled_impetuous_scene.py](../tests/test_drilled_impetuous_scene.py)
passed in isolated 512 MiB services, peak RSS 267.9 MiB. They mock decision-making
and execution. No autonomous-match, playing-strength, or search-performance
baseline was established. The full suite was not run for the review.

## First Implementation Slice

Implemented and verified on 2026-09-18, following separate user authorization:

- Snapshots reuse `characters.side_of`, capture world-space position/facing,
  preserve host membership, and omit destroyed or removed entities. Unknown
  ownership fails explicitly instead of defaulting to player two.
- Ground/mounted and flying Movement come from the model capability helpers;
  active flight mode is distinguished from ground movement and zero is preserved.
  Actual FSM state, charge stage, and both completed-turn counters survive cloning.
- Snapshot classification is refreshed per observation. Joined characters remain
  in the observation but are excluded from independent formation action selection
  in both heuristic and search generators.
- The controller checks active-player ownership and supported windows, yields to
  existing busy resolutions, handles missing decisions without phase advancement,
  works without a search tree, and releases its running lock in `finally`.
- Decisions are discarded after observed phase, player, turn-counter, charge-stage,
  or load-generation changes. A transient generation in the actual load path
  invalidates results even when reloading the same phase and turn.

Verification: all 51 cases in [tests/test_ai.py](../tests/test_ai.py) passed without
starting a game window (peak RSS 112.7 MiB). The two existing EnhancedAI charge
integration cases passed with explicit fixture ownership (peak 265.5 MiB), and
the real same-window save/load charge case passed (peak 266.0 MiB). Each module
ran sequentially through the isolated runner with a 512 MiB cap and the existing
256 MiB available-memory headroom. Editor diagnostics were clean. No full-suite,
autonomous-match, playing-strength, or search-performance claims are made.

At this first-slice checkpoint, the following remained open; subsequent command
and baseline work is recorded below.

LEFTOVER: snapshots remain mutable, name-keyed legacy dictionaries, not complete
versioned observations. They do not yet capture all effects, budgets, joined-unit
movement restrictions, terrain, objectives, or aggregate joined combat value.
Only snapshot classification caching changed; other caches remain uncorrected.
The freshness guard is window/load-based, not a revision of every same-window
mutation; it discards results but does not cancel search workers or already
scheduled commands. Command-specific completion, rejection handling, animation
cancellation, autonomous driving, charge tactics, and forward-model/search
correctness are still open. Special decision windows keep their existing handlers.

## Phase 2 Delivery And Evidence

### Implemented Baseline

[ai_policy.py](../ai_policy.py) ranks live candidates with shared legality queries.
The controller revalidates and executes them through the real engine, not the old
forward model. It now supports both players, continuous AI-versus-AI, and manual
stepping. AI remains disabled at startup. Controls in [game.py](../game.py):

- `F4`: enable/disable player two; `Shift+F4`: enable/disable player one.
- `Ctrl+F4`: toggle continuous/manual operation for the current player's AI.
- `A`: step the current enabled AI in manual/debug operation.

Disabling stops new commitments; an already committed command finishes through
its rules path. Movement, shooting/animation, casting and combat await their own
completion. Rejected actions do not invent spent allowances. Exceptions, explicit
resolver failures and repeated no-progress decisions pause the AI without skipping
a phase. Loading is refused during committed AI commands; window/load-generation
changes invalidate pending decisions. Enabled/manual settings for both players and
the standard battle's first-player setting survive saves; legacy saves default to
P1 human, continuous operation and P1 first. Driver tasks and HUD rule listeners
are released on shutdown.

The policy considers current objective control and configured rewards, legal
missile weapons and targets, support from potential/committed charges and engaged
allies, normal rallies, implemented spells, character escorts/retreats, leaving
Marching Column, and Reserve Move. It conserves a single-use casting/dispel bonus
when its threshold or battlefield pressure makes it unhelpful. Scores are labelled
utility, not probabilities. The only rule correction in this delivery is artillery
shot spending on committed misfires, documented with sources and remaining limits
in [SPECIAL_RULES_CHECKLIST.md](../SPECIAL_RULES_CHECKLIST.md).

### Decision Window Inventory

All entries use existing authoritative resolution and legal options. A legal
decline is not permission to skip compulsory work. Generic prompts select the
first offered option for their actual owner; stronger rule-local policies retain
priority. This is a basic fallback, not a strategic ranking of every prompt.

| Window | Baseline And Fallback |
| --- | --- |
| Army reports, terrain, zones, scenario setup | Existing preparation handler; acknowledge reports, select offered legal options, await placement and boundary work. Empty/invalid setup stops rather than fabricating completion. |
| Ordinary deployment, Scouts, Vanguard | Existing legal placement/movement handlers, now side-aware; bounded placement attempts, optional Vanguard move/skip. Placement exhaustion disables the owning AI. |
| Command, Rallying Cry, ordinary rally | Existing command resolution runs before ordinary rally/casting; joined characters use host ownership. No human mouse-reform wait for AI. |
| Strategy, movement and shooting spells | Shared spell construction and phase/target checks; select supported legal spell/target, otherwise decline casting. Unknown catalogue effects and unsupported target forms are not simulated. |
| Assailment, spell generation, Wizardly/Fated and Remains in Play dispels | Existing magic-local handlers, legal spell/target choices and attempt limits. Dispel policy prefers the strongest eligible wizard; otherwise Fated if available. Scarce bonus policy may retain its item. |
| Charge declarations, compulsory charges, reactions, Counter Charge, order | Shared route candidates with support scoring; declaration completion still calls the compulsory Frenzy/Impetuous resolver. Opponent-owned reactions and charge order stay inside existing resolution. No unconditional nearest-enemy charge pass. |
| Remaining movement, flight, terrain | Sample objective/support/retreat destinations, then authoritative path and budget validation. Preserve useful firing/objective positions; rejected routes yield another candidate or a legal hold. Existing flight/Drilled/terrain safeguards remain active. |
| Characters, formations, optional pivots | Legal non-marching join/leave previews and commits; redress out of Marching Column. Keep current formation/facing when no supported beneficial action is available. |
| Shooting, cannon, bombardment | Compare available missile weapons and legal targets, then real volley/artillery resolver. Engaged/blocked/out-of-range targets are rejected. A committed misfire spends one shot; invalid targeting does not. |
| Reserve Move | Separate extra-move budget through the existing window; move if useful/legal, otherwise decline. Restore the ordinary movement record afterward; no march or charge. |
| Connected combat, weapons, challenges, items | Await the real connected-combat task with existing rule-local selections and opponent ownership; surface resolver errors as failures. Equipment abilities retain existing eligibility/spending rules. |
| Panic, Break, flee, pursuit, overrun, reform | Existing nested resolution owns the task/choice until completion; controller waits on active Panic/combat/reform work. Optional choices use their local policy or offered-option fallback. |
| Objective control, secondary/scenario prompts, terminal scoring | Existing scenario handlers resolve legal choices and score; movement uses current geometry, not stale stored ownership. Terminal result comes from the authoritative battle scorer. |

### Acceptance Results

[tests/test_ai_scene.py](../tests/test_ai_scene.py) runs the real policy, FSM,
commands, Bullet geometry, dice, magic and combat with offscreen Panda3D. It replaces
figure artwork and reduces the baked mat texture only; it isolates mutable user
terrain files, presets, rosters and saves. Gameplay decisions/resolvers are not
mocked in the autonomous matrix. Both sides use five archers with Longbows, five
Chaos Warriors and a level-one Mage: mixed runtime profiles, not tournament-legal
faction lists. Each side plays six turns. Animation speed is accelerated and the
clock is fixed-step; elapsed times are not normal interactive pacing measurements.

Final report: `.pytest_cache/isolated/20260918-205648-1351831/summary.json`.
All 12 scene cases passed in 157.54 seconds, peak RSS 742.7 MiB under a 768 MiB cap.
Each match retained its 45-second deadline and 18,000-frame watchdog.

| Scenario / Seed | Actual First Player | Completed Turns P1/P2 | Seconds |
| --- | --- | --- | --- |
| Standard / 41 | 1 | 6 / 6 | 10.14 |
| Standard / 42 | 2 | 6 / 6 | 23.16 |
| Battle March / 41 | 1 | 6 / 6 | 23.41 |
| Battle March / 42 | 2 | 6 / 6 | 34.63 |
| Reed Fens / 41 | 2 | 6 / 6 | 24.30 |
| Reed Fens / 42 | 2 | 6 / 6 | 26.22 |

All six reached `BattleEnded`, kept both controllers enabled, completed preparation,
and recorded actual shooting and casting. Standard starting order is explicitly
asserted; configured scenarios retain their real roll-off/choice, so Reed Fens is
not starting-order-balanced in this small sample. Six additional live cases cover
casting, legal shots/blocked LOS/objective approach and hold, artillery/formation,
character joining, control toggles and Reserve Move budget restoration.

Additional focused evidence, all through sequential isolated services:

- 71 lightweight AI tests passed: observations, locks, ownership, stale decisions,
  rejected actions, failure/stall recovery, coordinated charge support, artillery
  misfire spending, scarce-bonus thresholds and actual item inventory, and reload
  refusal mid-command. Report `20260918-205530-1351251`, peak 116.2 MiB.
- 34 persistence and 68 Scout tests passed. Latest persistence report
  `20260918-205447-1350930`; Scout report `20260918-205042-1349252`.
- 11 affected charge/Drilled/Impetuous/Vanguard scene cases passed, including
  compulsory declaration completion and reload: `20260918-205213-1349895`,
  peak 420.5 MiB. Two joined/ordinary AI command cases passed:
  `20260918-205546-1351395`. Extended live reload coverage verifies first-player
  and both AI settings: `20260918-205627-1351664`.
- 40 selected-item cases passed excluding three unchanged Eye of Numas fixture
  failures (missing model metadata); seven focused live shooting cases passed.
  All touched Python files had clean editor diagnostics at handoff.

Match JSON records include schema/policy version, named scenario/seed, setup,
actual first player, turns, frames, elapsed time, authoritative score, action
outcomes and failure status. They are written beside the fixture's temporary save
and into JUnit properties. Failures/timeouts are also retained by the isolated
runner. Pytest emits six `record_property`/xunit2 compatibility warnings; the JSON
properties were present and parsed successfully. Search node counts and per-match
RSS are not claimed; this baseline does not search and RSS is module-level.

**Exit-gate assessment:** met for the agreed supported matrix and focused tactical
checks. No mandatory-choice bypass was observed, and explicit ownership, busy-work,
compulsory-charge and failure regressions protect those boundaries. This is finite
acceptance evidence, not proof for every army, rule combination or optional scenario.

LEFTOVER: sampled movement can miss useful routes; combat utility is a frontage/
profile proxy, not mutual damage or calibrated win odds. Broader threat maps,
activation-order planning, general formation optimisation, diverse legal armies,
held-out opponents/seeds and balanced paired strength benchmarks remain future
work. Generic first-option and conservative spell/item fallbacks can be weak.
Existing artillery damage/geometry/Misfire-table limits remain unchanged. Complete
effect-sensitive observations, worker cancellation and all legacy forward-model,
cache, quiescence and MCTS defects remain open; speculative search is disabled by
default. No full suite was run. A full shooting module exceeded its earlier
512 MiB cap; a later 1536 MiB attempt was refused by the memory admission guard,
which was not bypassed. The focused results above do not substitute for release
regression gates or human play-testing.

## Simulation Simplification Policy

Optimize for correct decision rankings at affordable cost, not perfect replay of
every die and animation. Explicitly separate reduced detail from changed rules.

| Preserve As A Contract | May Be Approximated, With Validation |
| --- | --- |
| Ownership, actual decision window, phase order, and both players' turn counters | Long-range strategic consequences beyond the search horizon |
| Action eligibility, resource use, mandatory choices, and explicit success/failure branches | Opponent response selection and candidate action count |
| No movement or attack budgets invented for unrelated units | Distant-unit influence and coarse spatial threat maps |
| Legal root actions and final execution through authoritative validators | Interior-node route cost and contact estimates, with uncertainty recorded |
| Formation, facing, terrain, and contact constraints where they decide feasibility | Candidate destinations instead of continuous-coordinate enumeration |
| Distinct living/dead, fleeing, joined, engaged, and effect-lifetime states | Grouped ordinary models and aggregate attack distributions |
| Actual terminal scoring and no access to private opponent information | Nonterminal future-value estimates and sampled future dice |

Never declare a charge successful merely because centers are nearby, exhaust
other units to shrink the tree, or invent a legal action when no route exists.
Restrict candidates or shorten the horizon instead. If a speculative route
cannot be verified cheaply, mark it approximate and refine it before committing
the plan; do not present it as guaranteed legal.

### Multiple Levels Of Detail

1. Cheap screening: unit capabilities, distance bounds, influence maps, expected
   damage, and rough objective value eliminate clearly poor candidates.
2. Tactical refinement: shared geometry and richer combat distributions evaluate
   the shortlist, especially close charge distances and high-value trades.
3. Execution: revalidate against the current live state and await the authoritative
   action outcome. Replan after dice, reactions, casualties, or changed geometry.

Do not average discrete consequences into impossible states. For example, charge
success and failure need separate outcomes, not a fractional engagement. Expected
wounds are useful for screening; shortlist evaluation must retain meaningful
variance and break/survival probabilities. A risky losing-position comeback and
a low-risk defense of a lead should not receive the same preference by default.

Validate each approximation against seeded authoritative resolutions. Measure
decision-ranking agreement and tactical regret as well as prediction error.
Record where a faster model is inaccurate and which decisions require refinement.

## Target Architecture

`Observe -> Identify decision owner -> Generate candidates -> Validate -> Score/search -> Execute -> Observe`

These are logical responsibilities, not a requirement to create a new framework
or one file per responsibility. Preserve public APIs where practical and extract
pure helpers into existing ownership boundaries first.

### Observation And State

- Immutable, versioned snapshots with stable unit identities and world-space
  positions/facings. Search must not mutate live models, NodePaths, or rule state.
- Capture actual subphase and pending decision, its owner, both turn counters,
  battlefield geometry, public objectives, score ledger, and remaining budgets.
- Represent joined characters as members with their own capabilities and wounds,
  not additional independently movable regiments or duplicate army value.
- Include effective movement modes, formations, combat links, split profiles,
  weapons, available spells, relevant effects, and spent abilities as needed by
  supported predictions. Reuse existing capability/ownership helpers.
- Separate public observations from full engine state. Hidden objectives or
  unrevealed selections, when supported, must not leak into evaluation, logs, or
  cached opponent predictions. Uncertain information needs an explicit model.
- Key caches by all decision-relevant state and model/schema versions. Handle
  load, army replacement, and changed effects; name-only caching is insufficient.

### Actions And Decisions

- Use explicit intents with parameters, decision owner, and snapshot revision.
  Shared validators return legality and a reason; refusal is a normal outcome.
- Distinguish declaring a charge, choosing a reaction, selecting charge order,
  and executing charge movement. Plan combinations without pretending the entire
  sequence is one uninterrupted action owned by the active player.
- Resolve melee at the actual connected-combat level. Do not simulate or execute
  a one-sided attack per unit when the live command resolves the entire fight.
- Retain existing unit roles as candidate generators and priors, not hard-coded
  orders that override legality, objectives, or a better evaluated action.
- Generate alternatives for activation order, holding, reforming, screening,
  flanking, charging, shooting, casting, and objective occupation. Limit branching
  through ranked candidates, not by altering action budgets.
- Route rule-local prompts through a common policy boundary while preserving
  their legal options and existing human UI. Do not use display order as strategy.

### Evaluation And Planning

- Prioritize actual scenario victory conditions. Terminal values come from the
  authoritative scoring semantics, including destroyed/fleeing units, characters,
  trophies, objectives, and turn limits where applicable.
- Nonterminal features include expected score, survival, charge and shooting
  threats, usable frontage, support, escape routes, and future objective control.
- Keep strategic assignments stable until invalidated or materially outperformed;
  avoid changing doctrine after every minor action.
- Estimate mutual damage using relevant attack profiles, Initiative, Strength,
  Toughness, armour/AP, Ward/Regeneration, and consequential special rules.
  Add combat-result and morale outcomes before claiming charge-trade prediction.
- Integrate changes in target state across a plan: casualties, blocked frontage,
  consumed reactions, buffs, and overkill alter later actions' value.
- Label utility scores as scores. Reserve probability labels for calculated or
  empirically calibrated probabilities with their assumptions recorded.

### Execution And Runtime

- Commands return structured outcomes: completed, rejected, cancelled, or failed,
  with the resulting state revision. Await actual completion, including nested
  choices, animations, Panic, pursuit, and reform when part of that command.
- Replace reliance on a global completion boolean with command-specific task or
  event identity. Keep Panda3D mutation on the owning thread.
- Use `try/finally` for locks, validate the decision owner before every command,
  and recheck snapshot freshness after background search.
- Distinguish a deadline from successful completion. Timeout must not spend an
  action, skip a compulsory resolution, or advance through unfinished work.
- Pause/toggle/cancel stops new commitments; already committed rules resolve to
  a safe boundary. Loading discards obsolete searches and rebuilds observations.
- Search has separate RNG streams from gameplay and explicit node/time/memory
  budgets. Wall-clock budgets use a monotonic clock. Every search path, including
  tactical extensions and expensive candidate generation, must be bounded or
  cooperatively cancellable.
- Log chosen action, deciding features, rejected alternatives, uncertainty,
  budget usage, and actual outcome once per decision. Reuse the battle journal;
  do not flood per-roll loops or claim an uncalibrated win percentage.

## Search Strategy

Start with utility selection and shallow rollout/beam comparisons against a
credible opponent policy. This is the reference opponent, fallback, and rollout
policy, not disposable scaffolding.

For small tactical trees, expectimax/expectiminimax can enumerate compact chance
events and adversarial responses. For larger trees, benchmark chance-aware MCTS
with heuristic priors and progressive widening of movement candidates. Sample
chance by its probability; it is not an opponent optimizing dice against us.

Always derive maximizing/minimizing ownership from the next decision, not depth
parity. Preserve consecutive actions by one player and opponent-owned reactions.
Measure horizons in meaningful decision windows or completed turns as well as
raw action depth. Prevent a long army phase from consuming the entire horizon
before any opponent response is considered.

Alpha-beta remains useful for suitable deterministic tactical subproblems. Do
not retain null-move pruning or stand-pat assumptions without proving them valid
for the decision window. Quiescence must have a bounded horizon, correct actor,
legal progress, and deadline checks. Never cache incomplete results as exact.

MCTS is a candidate to evaluate, not a promised improvement. Promote it only if
it improves held-out tactical and match results at comparable compute budgets.
Learned policy/value models and self-play are deferred until simulation throughput,
correctness, datasets, and reproducible evaluation justify their cost. MuZero,
LLM-controlled combat, and learning game rules from pixels are not initial scope.

## Phased Delivery

Checked items record verified work; unchecked items remain open. Each phase records its tests,
measurements, limitations, and explicit `LEFTOVER:` items before handoff.

### Phase 0: Establish Reproducible Evidence

- [x] Turn snapshot ownership/transforms, movement modes, phase windows, clone
  independence, classification freshness, and controller recovery into regressions.
- [ ] Turn state hashes, quiescence actor, round advancement, and global move
  exhaustion probes into regressions.
- [x] Define fixture-owned roster/configuration scenarios, without depending on
  mutable user presets or saves. Reuse existing test helpers and scene fixtures.
- [ ] Record the old policy's outcomes where it can finish; count crashes/stalls
  as failures, not omitted benchmark games.
- [ ] Define artifact schema: seed, policy/config version, scenario, actions,
  terminal score, failure reason, timing, node count, and peak memory.

Exit gate: known defects are reproducible, with no claimed playing-strength
baseline based solely on the existing mocked phase tests.

### Phase 1: Correct Observations And Controller Lifecycle

- [x] Correct snapshot ownership/world transforms, movement modes, joined-member
  action selection, actual FSM/charge state, both counters, and snapshot classification.
- [ ] Complete versioned observations, effect-sensitive caches, and pending-decision
  ownership beyond the captured FSM/charge state.
- [x] Add ownership checks, unconditional lock cleanup, safe handling of no-action
  results, and operation with search/visualization disabled.
- [x] Reject obsolete decisions at controller boundaries after observed window
  changes or reloads into the same window.
- [x] Establish command-specific completion for movement, shooting and combat;
  preserve mandatory phase-boundary continuations and pause at safe boundaries.
- [ ] Add general cancellation of scheduled commands/search workers and complete
  same-window effect revisions beyond the current load/window guards.
- [x] Add a bounded autonomous driver for either side and both sides, preserving
  a manual single-step/debug mode and safe pause/load behavior.
- [x] Keep faulty speculative search out of the default decision path until its
  correctness gates pass; retain old code only as an explicitly experimental mode.

Exit gate: scripted legal commands complete full configured battles with both
starting orders; wrong-side calls, stale results, rejection, and exceptions do
not mutate unrelated state or leave locks stuck. Scripted completion does not
yet establish autonomous tactical competence.

### Phase 2: Complete A Legal Baseline Opponent

- [x] Connect shared candidate validation and eliminate the unconditional
  nearest-enemy charge pass that bypasses policy.
- [x] Provide legal actions or explicit conservative fallbacks for setup/deployment,
  Scouts/Vanguard, command/rally, implemented
  casting windows, dispels, charges/reactions/order, remaining movement,
  shooting/artillery, Reserve Move, combat choices, and post-combat decisions.
- [x] Provide a legal baseline for supported equipment abilities, character
  joining/leaving, formation changes, and scenario-specific decisions.
- [x] Add objective/scoring-aware utility, terrain-validated movement, appropriate holding,
  target selection, and coordinated charge candidates.
- [ ] Expand terrain/threat evaluation beyond existing route/LOS validation and
  the simple nearest-enemy pressure/retreat heuristic.
- [x] Adapt existing rule-local policies instead of losing their safeguards.
  Enumerate every supported decision window and its fallback explicitly.

Exit gate: the baseline completes unmocked autonomous matches in the agreed
scenario matrix, never bypasses mandatory choices, and passes tactical checks
for target legality, objectives, support, and conserving scarce abilities.

Gate met on 2026-09-18 with the evidence and limitations above; this does not close
the remaining Phase 0/1 infrastructure or the stronger threat-evaluation item.

### Phase 3: Build A Calibrated Fast Forward Model

- [ ] Extract compact numeric state transitions, without scene creation or
  graphical dice. Share authoritative pure rule helpers where practical.
- [ ] Implement explicit success/failure and chance outcomes, effect lifetimes,
  coherent casualty/contact updates, correct turn progression, and terminal score.
- [ ] Build layered combat/morale estimates and selective geometry refinement.
- [ ] Document predictive coverage per consequential rule; use conservative
  fallback or richer resolution for unsupported high-impact interactions.
- [ ] Differential-test deterministic outcomes with injected dice and compare
  stochastic distributions and action rankings across seeded scenarios.

Exit gate: exact contracts agree with the live resolver; approximation errors
are measured with predeclared tolerances. Record rollouts/second and memory per
state before deciding which search depth or algorithm is affordable.

### Phase 4: Add Selective Chance-Aware Search

- [ ] First establish bounded shallow search against a brute-force reference on
  tiny games, including consecutive same-player actions and reactions.
- [ ] Repair or replace cache bounds, state keys, tactical extensions, and time
  handling; cached/uncached results agree for exact small-tree tests.
- [ ] Compare utility-only, shallow rollouts/beam, and chance-aware MCTS using the
  same observations, legal candidates, evaluation, and compute budgets.
- [ ] Add coordinated action-order exploration and refine uncertain top plans.
- [ ] Return a validated baseline action when interrupted before a completed
  search result; never execute an obsolete snapshot's choice blindly.

Exit gate: search beats the legal baseline on held-out tactical cases and paired
matches within the agreed latency/memory budget, without correctness regressions.

### Phase 5: Tune, Integrate, And Release

- [ ] Calibrate evaluation and risk preferences on training scenarios, retaining
  separate held-out maps, matchups, and seeds.
- [ ] Set difficulty through candidate quality/budget and policy choices, not
  hidden information, altered dice, or illegal actions.
- [ ] Persist user-facing AI settings and necessary strategic intent; do not
  serialize worker threads or speculative trees. Resume safely at supported
  save boundaries and document legacy-save defaults.
- [ ] Update user documentation and retire misleading depth/performance claims
  and redundant legacy paths only after replacement coverage exists.
- [ ] Run affected integration gates and the required full isolated suite at the
  release checkpoint. Add offscreen UI checks only for changed controls/views.

Exit gate: publish supported scope, benchmark results, reproducibility details,
known limits, and human play-test observations. Do not claim universal rules
coverage or strong play based solely on self-play against the same policy.

## Verification And Performance Gates

| Level | Required Evidence |
| --- | --- |
| Pure contracts | Snapshot ownership/world geometry, independent clones, actor transitions, budgets, scoring, chance probabilities, cache invalidation, no live-state or gameplay-RNG mutation |
| Tactical scenarios | Safe vs losing charge, coordinated flank, charge failure/Counter Charge, blocked shot with legal alternative, rally, spell timing, preserve General, objective trade, final-turn scoring |
| Formation/rule matrix | Formed and loose units, joined characters, mounts/crew, multiwound targets, flight/ground mode, relevant terrain and active effects |
| Live execution | Actual planner and executor through all decision windows, both sides, rejected actions, nested choices, pause/toggle, save/reload, complete battles |
| Match evaluation | Side-swapped and starting-order-balanced games, diverse opponent policies, held-out maps/armies, completion failures included |
| Runtime | Median/p95/max decision and whole-turn latency, worst main-thread slice, nodes/rollouts, cache hit rate, memory, cancellation delay |

Proposed initial desktop targets, to be confirmed after Phase 3 profiling:
routine decisions under 250 ms p95, critical decisions under 3 seconds p95,
main-thread AI work slices under 10 ms, and cancellation observed within 100 ms
outside an indivisible engine operation. Report decision time separately from
animations and rule resolution. Define whole-turn and worker-memory budgets from
measured army sizes; a per-action deadline alone is not a turn budget.

Run an initial batch of at least 50 paired seeds per major matchup, expanding
when uncertainty is too large to decide promotion. Pair seeds and swap sides;
different policies may consume random events differently, so seed pairing does
not guarantee identical dice histories. Report win/draw/loss, score differential,
confidence intervals clustered by paired seed, stalls, and illegal/rejected
attempts. Scripted and diverse heuristic opponents complement self-play. Human
play-tests assess exploitability, pacing, clarity, and repetitive behavior.

Tests remain sequential and memory-bounded through
[run_tests_isolated.py](../run_tests_isolated.py). Every terminal command starts
with `source .venv/bin/activate`. Never combine multiple test modules in one
pytest process or bypass the memory admission guard. Preserve the full-suite
1536 MiB cap plus 256 MiB headroom. New long-running match benchmarks also need
isolated bounded workers and watchdogs, not uncontrolled parallel scene creation.
Do not rerun the full scene suite after every narrow fix.

## Migration And First Implementation Slice

Keep [aiMinimaxIntegration.py](../aiMinimaxIntegration.py) as the live adapter
while extracting responsibilities incrementally. Reuse suitable parts of
[strategyAdvisor.py](../strategyAdvisor.py) and
[unitTypeClassifier.py](../unitTypeClassifier.py) for candidate generation, not
as authoritative outcome predictors. Reconcile the separate evaluation in
[gameStateAnalyzer.py](../gameStateAnalyzer.py) with the planner so explanations
describe the actual decision. Replace legacy transitions in
[gameStateTree.py](../gameStateTree.py) behind verified contracts before tuning
[minimaxOptimizations.py](../minimaxOptimizations.py).

The first implementation slice should be Phase 0 regressions plus the bounded
Phase 1 fixes for snapshot correctness and controller recovery. Its deliverable
is trustworthy observations and a controller that fails safely, not stronger
search. Subsequent command integration should proceed one decision window at a
time with focused live tests.

AI prediction must not silently change game rules. When implementation requires
a rule correction, check the current wording and overriding FAQ at
<https://tow.whfb.app/>, cite the source page in the relevant code/docstring, log
outcome-changing rules, and update
[SPECIAL_RULES_CHECKLIST.md](../SPECIAL_RULES_CHECKLIST.md) with corrections and
explicit leftovers.

## Deferred Work And References

LEFTOVER: unchecked items remain outstanding. Phase 2 gate completion does not
repair the old simulator or establish a playing-strength benchmark. Neural training, hidden-information
search beyond supported scenarios, broad new rule implementation, multiplayer
transport, and army-list optimization are separate future scopes.

- [OpenSpiel core API](https://openspiel.readthedocs.io/en/latest/api_reference.html):
  reference contracts for legal actions, decision owners, transitions, observations,
  chance outcomes, and terminal returns; not a drop-in Warhammer engine.
- [OpenSpiel algorithms](https://openspiel.readthedocs.io/en/latest/algorithms.html):
  reference implementations and testing status for search and learning approaches.
- [MuZero](https://arxiv.org/abs/1911.08265): learned policy/value/model search as a
  long-term research direction, not a prerequisite for a competent opponent.
- [MINIMAX_README.md](../MINIMAX_README.md): historical architecture documentation;
  its performance estimates are not verified acceptance evidence for this revamp.