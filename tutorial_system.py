"""Tutorial Scenario Manager.

Manages a sequence of scripted battles that progressively introduce
game mechanics.  Each tutorial is defined by a JSON file containing an
ordered list of battles with army compositions, terrain layouts, win
conditions, and contextual hint text.

Usage from ``game.py``::

    from tutorial_system import TutorialManager

    # In MyApp.__init__ (after FSM, terrain_manager, etc. are ready):
    self.tutorial = TutorialManager(self)
    self.accept('t', self.tutorial.start, ['tutorials/tutorial_basics.json'])

The manager handles:
* Loading / parsing tutorial JSON files
* Clearing and rebuilding the battlefield between battles
* Army loading for both players
* Terrain placement per battle
* Displaying phase-aware tutorial hints
* Win-condition checking (per-turn)
* Transitioning to the next battle on victory
"""

import json
import os

from panda3d.core import Point3, Vec4, TextNode, TransparencyAttrib, Texture as P3DTexture
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectButton, DirectFrame, DirectLabel, DGG
from direct.gui.OnscreenImage import OnscreenImage
import gui_theme as T

# ── Re-export theme constants under private names for backward compat ────────
_PARCHMENT       = T.PARCHMENT
_PARCHMENT_DARK  = T.PARCHMENT_DARK
_GOLD            = T.GOLD
_INK             = T.INK
_INK_FADED       = T.INK_FADED
_RED_WAX         = T.RED_WAX
_GREEN_BANNER    = T.GREEN_BANNER
_HINT_BG         = T.HINT_BG
_HINT_FG         = T.HINT_FG
_SHADOW          = T.SHADOW
_FONT_PATH       = T.FONT_PATH

_TEX_DIR         = T.TEX_DIR
_TEX_PARCHMENT   = T.TEX_PARCHMENT
_TEX_VELLUM      = T.TEX_VELLUM
_TEX_BORDER      = T.TEX_BORDER
_TEX_BUTTON      = T.TEX_BUTTON
_TEX_BUTTON_HOVER = T.TEX_BUTTON_HOVER
_TEX_BUTTON_RED  = T.TEX_BUTTON_RED
_TEX_VICTORY     = T.TEX_VICTORY


def _tex_frame(image_path, parent=None, pos=(0, 0, 0), scale=(1, 1, 1)):
    """Create an OnscreenImage used as a textured background panel."""
    img = OnscreenImage(
        image=image_path,
        pos=pos,
        scale=scale,
        parent=parent,
    )
    img.setTransparency(TransparencyAttrib.MAlpha)
    return img


def _tex_button(text, font, pos, command, parent=None,
                normal=_TEX_BUTTON, hover=_TEX_BUTTON_HOVER,
                scale=0.06, pad=(0.5, 0.25)):
    """Create a DirectButton with texture backgrounds."""
    btn = DirectButton(
        text=text,
        text_font=font,
        text_fg=T.BTN_TEXT,
        scale=scale,
        pos=pos,
        command=command,
        parent=parent,
        frameTexture=normal,
        frameColor=(1, 1, 1, 1),
        pad=pad,
        relief=DGG.FLAT,
    )
    btn.setTransparency(TransparencyAttrib.MAlpha)
    return btn


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_json(filepath: str) -> dict:
    """Read and return a JSON file as a dict."""
    with open(filepath, 'r') as fh:
        return json.load(fh)


# ── TutorialBattle ────────────────────────────────────────────────────────────

class TutorialBattle:
    """In-memory representation of a single battle within a tutorial."""

    def __init__(self, data: dict):
        self.id: str = data.get('id', 'unnamed')
        self.name: str = data.get('name', 'Unnamed Battle')
        self.description: str = data.get('description', '')
        self.hints: list[dict] = data.get('hints', [])
        self.player1_army: dict = data.get('player1_army', {})
        self.player2_army: dict = data.get('player2_army', {})

        # ── Objectives — each is a dict {"text": ..., "condition": {...}}
        # For backwards compatibility, also accept a plain string list in
        # "objectives" plus a single "win_condition" dict.
        raw_objectives = data.get('objectives', [])
        raw_wc = data.get('win_condition', None)
        raw_wcs = data.get('win_conditions', None)

        if raw_wcs is not None:
            # New format: list of {"text": ..., "condition": {...}}
            self.objectives: list[dict] = raw_wcs
        elif raw_wc is not None:
            # Legacy single win_condition — pair plain objectives with it
            self.objectives = []
            for obj_text in raw_objectives:
                self.objectives.append({
                    'text': obj_text,
                    'condition': dict(raw_wc),
                })
            # If there were no objective strings, create one from the condition
            if not self.objectives:
                self.objectives.append({
                    'text': raw_wc.get('description', 'Complete the objective'),
                    'condition': dict(raw_wc),
                })
        else:
            self.objectives = [{'text': t, 'condition': {}} for t in raw_objectives]
        self.terrain: list[dict] = data.get('terrain', [])
        self.deployment: dict = data.get('deployment', {})


# ── TutorialManager ──────────────────────────────────────────────────────────

class TutorialManager:
    """Orchestrates a tutorial campaign — a sequence of battles with hints."""

    # How often (seconds) the win-condition checker runs
    _CHECK_INTERVAL = 2.0

    def __init__(self, game):
        self.game = game
        self.battles: list[TutorialBattle] = []
        self.current_index: int = -1
        self.active: bool = False
        self.tutorial_name: str = ''

        # ── UI elements ──────────────────────────────────────────────
        self._hint_text: OnscreenText | None = None
        self._objective_text: OnscreenText | None = None
        self._title_text: OnscreenText | None = None
        self._transition_frame: DirectFrame | None = None
        self._next_btn: DirectButton | None = None

        # Track which hints the player has already seen (by phase)
        self._shown_hints: set[str] = set()

        # Persist the tutorial filepath so we can reference it later
        self._filepath: str = ''

    # ──────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────

    def start(self, filepath: str, battle_index: int = 0):
        """Load a tutorial JSON and begin at *battle_index*."""
        self._filepath = filepath
        data = _load_json(filepath)
        self.tutorial_name = data.get('name', 'Tutorial')
        self.battles = [TutorialBattle(b) for b in data.get('battles', [])]
        if not self.battles:
            print('[Tutorial] No battles found in', filepath)
            return
        self.active = True
        self.current_index = battle_index
        self._create_ui()
        self._load_battle(self.current_index)

    def stop(self):
        """End the tutorial and tear down UI."""
        self.active = False
        self._stop_win_check()
        self._destroy_ui()
        self.battles.clear()
        self.current_index = -1
        print('[Tutorial] Stopped.')

    def advance(self):
        """Move to the next battle in the sequence.
        If we are on the last battle, show a completion screen."""
        if not self.active:
            return
        next_idx = self.current_index + 1
        if next_idx >= len(self.battles):
            self._show_completion()
            return
        self.current_index = next_idx
        self._load_battle(next_idx)

    def skip_battle(self):
        """Allow the player to skip the current battle."""
        self.advance()

    @property
    def current_battle(self) -> TutorialBattle | None:
        if 0 <= self.current_index < len(self.battles):
            return self.battles[self.current_index]
        return None

    # ──────────────────────────────────────────────────────────────────
    #  Battle lifecycle
    # ──────────────────────────────────────────────────────────────────

    def _load_battle(self, index: int):
        """Clear the field and set up a new battle."""
        battle = self.battles[index]
        print(f'\n{"=" * 60}')
        print(f'  TUTORIAL — Battle {index + 1}/{len(self.battles)}: {battle.name}')
        print(f'  {battle.description}')
        print(f'{"=" * 60}\n')

        self._shown_hints.clear()
        self._stop_win_check()

        # 1. Clear battlefield
        self._clear_battlefield()

        # 2. Write temporary army JSONs for the existing loader
        p1_path = self._write_temp_army(battle.player1_army, '_tutorial_p1.json')
        p2_path = self._write_temp_army(battle.player2_army, '_tutorial_p2.json')

        # 3. Load armies via existing helpers
        self.game.load_player1_army(p1_path)
        self.game.load_player2_army(p2_path)

        # Remember initial counts so win checks work after units are removed
        self._initial_p1_count = len(self.game.player1Units)
        self._initial_p2_count = len(self.game.player2Units)

        # 4. Terrain
        for t in battle.terrain:
            self.game.terrain_manager.add_terrain(
                t['type'],
                Point3(t['x'], t['y'], 0.1),
                t['width'],
                t['depth'],
            )

        # 5. Update overlay UI
        self._update_title(battle)
        self._update_objectives(battle)
        self._update_hint_for_phase(self._current_phase_name())

        # 6. Restart at deploy phase (must happen BEFORE round counter
        #    reset because exiting CombatPhase calls next_turn())
        self.game.fsm.request('DeployPhase')

        # 7. Reset round counter to round 1, player 1 — after the FSM
        #    transition so exitCombatPhase's next_turn() doesn't overwrite it
        if hasattr(self.game, 'roundCounter'):
            self.game.roundCounter.currentRoundPlayer = [0] * self.game.roundCounter.nPlayers
            self.game.roundCounter.request('PlayerOne')
            self.game.roundCounter.update_round_display()

        # 8. Start periodic win-condition checking
        self._start_win_check()

        # 9. Hook into FSM phase changes to update hints
        self.game.accept('tutorial-phase-change', self._on_phase_change)

    def _clear_battlefield(self):
        """Remove all units, terrain, and reset game state for a fresh battle."""
        # Remove units from Bullet world and scene graph
        for u in list(self.game.units):
            try:
                body = u.bodyNP.node()
                self.game.world.removeRigidBody(body)
            except Exception:
                pass
            try:
                u.bodyNP.removeNode()
            except Exception:
                pass
            try:
                u.model.removeNode()
            except Exception:
                pass
        self.game.units.clear()
        self.game.player1Units.clear()
        self.game.player2Units.clear()

        # Remove copied nodes
        for c in self.game.unitCopies:
            try:
                c.removeNode()
            except Exception:
                pass
        self.game.unitCopies.clear()

        # Clear terrain
        self.game.terrain_manager.clear()

    # ──────────────────────────────────────────────────────────────────
    #  Win-condition checking
    # ──────────────────────────────────────────────────────────────────

    def _start_win_check(self):
        taskMgr.doMethodLater(
            self._CHECK_INTERVAL, self._check_win_task,
            'tutorial_win_check',
        )

    def _stop_win_check(self):
        taskMgr.remove('tutorial_win_check')

    def _check_win_task(self, task):
        """Periodic task — evaluate every objective's condition.
        Updates the objective display and wins when ALL are complete."""
        if not self.active or self.current_battle is None:
            return task.done

        all_done = True
        for obj in self.current_battle.objectives:
            # Once an objective is completed it stays completed (sticky)
            if obj.get('_completed', False):
                continue
            condition = obj.get('condition', {})
            done = self._evaluate_condition(condition)
            if done:
                obj['_completed'] = True
            else:
                all_done = False

        # Refresh the objectives UI with current completion status
        self._update_objectives(self.current_battle)

        if all_done:
            self._on_battle_won()
            return task.done

        return task.again

    def _evaluate_condition(self, condition: dict) -> bool:
        """Return True when a single condition is satisfied."""
        ctype = condition.get('type', '')

        if ctype == 'destroy_all_enemies':
            # Units get removed from player2Units on death, so check
            # that enemies existed at battle start and none remain now.
            alive = [
                u for u in self.game.player2Units
                if u.bodyNP and not u.bodyNP.isEmpty()
                and u.state not in ('IsFleeing',)
            ]
            had_enemies = getattr(self, '_initial_p2_count', 0) > 0
            return len(alive) == 0 and had_enemies

        if ctype == 'all_units_moved':
            if not self.game.player1Units:
                return False
            return all(u.hasMovedThisTurn for u in self.game.player1Units)

        if ctype == 'survive_turns':
            target = condition.get('turns', 1)
            if hasattr(self.game, 'roundCounter'):
                return self.game.roundCounter.current_round >= target
            return False

        if ctype == 'unit_charged':
            return any(u.isInCombat for u in self.game.player1Units)

        if ctype == 'all_units_deployed':
            if not self.game.player1Units:
                return False
            return all(u.isDeployed for u in self.game.player1Units)

        # Unknown or empty — treat as satisfied (informational objective)
        return False

    def _on_battle_won(self):
        """Handle victory for the current battle."""
        battle = self.current_battle
        print(f'[Tutorial] Battle won: {battle.name}')
        self._show_transition_screen(battle)

    # ──────────────────────────────────────────────────────────────────
    #  Phase-change hint system
    # ──────────────────────────────────────────────────────────────────

    def _current_phase_name(self) -> str:
        """Return the FSM state name of the current phase."""
        try:
            return self.game.fsm.state
        except Exception:
            return ''

    def _on_phase_change(self, phase_name: str = ''):
        """Called when the game FSM transitions to a new phase."""
        if not phase_name:
            phase_name = self._current_phase_name()
        self._update_hint_for_phase(phase_name)

    def _update_hint_for_phase(self, phase_name: str):
        """Collect all hints for *phase_name* and show the first one."""
        if not self.active or self.current_battle is None:
            return

        # Gather all hints that match this phase
        matching = []
        for hint in self.current_battle.hints:
            hint_phase = hint.get('phase', '')
            if hint_phase in phase_name or phase_name in hint_phase:
                matching.append(hint['text'])

        if not matching:
            self._show_hint('')
            self._hide_next_hint_btn()
            return

        # Show the first hint and enable cycling button if there are more
        self._hint_queue = matching
        self._hint_queue_index = 0
        self._show_hint(self._hint_queue[0])
        self._update_next_hint_btn()

    def _on_next_hint_click(self):
        """Advance to the next hint in the queue."""
        if not self._hint_queue:
            return
        self._hint_queue_index = (self._hint_queue_index + 1) % len(self._hint_queue)
        self._show_hint(self._hint_queue[self._hint_queue_index])
        self._update_next_hint_btn()

    def _update_next_hint_btn(self):
        """Show or hide the Next Hint button based on queue size."""
        if len(self._hint_queue) > 1 and self._next_hint_btn:
            count = f"({self._hint_queue_index + 1}/{len(self._hint_queue)})"
            self._next_hint_btn['text'] = f"Next Hint {count}"
            self._next_hint_btn.show()
        else:
            self._hide_next_hint_btn()

    def _hide_next_hint_btn(self):
        if self._next_hint_btn:
            self._next_hint_btn.hide()

    # ──────────────────────────────────────────────────────────────────
    #  UI creation / updates
    # ──────────────────────────────────────────────────────────────────

    def _create_ui(self):
        """Build the persistent tutorial overlay elements with a medieval theme."""
        self._med_font = loader.loadFont(_FONT_PATH)

        # ── Parchment panel (top-centre) for title + objectives ──────
        self._obj_frame = DirectFrame(
            frameColor=(1, 1, 1, 1),
            frameSize=(-0.48, 0.48, -0.38, 0.02),
            pos=(0, 0, 0.95),
            frameTexture=_TEX_PARCHMENT,
            relief=DGG.RAISED,
            borderWidth=(0.012, 0.012),
        )
        self._obj_frame.setTransparency(TransparencyAttrib.MAlpha)

        # Top ornamental border strip
        self._obj_border = OnscreenImage(
            image=_TEX_BORDER,
            pos=(0, 0, 0.02),
            scale=(0.49, 1, 0.02),
            parent=self._obj_frame,
        )
        self._obj_border.setTransparency(TransparencyAttrib.MAlpha)

        self._title_text = OnscreenText(
            text='', pos=(0, -0.04), scale=0.055,
            fg=_INK, align=TextNode.ACenter,
            shadow=_SHADOW, mayChange=True,
            font=self._med_font,
            parent=self._obj_frame,
        )
        self._objective_text = OnscreenText(
            text='', pos=(-0.45, -0.10), scale=0.04,
            fg=_INK, align=TextNode.ALeft,
            shadow=(0, 0, 0, 0.3), mayChange=True,
            font=self._med_font,
            parent=self._obj_frame,
        )

        # ── Hint scroll (bottom-centre) ──────────────────────────────
        self._hint_frame = DirectFrame(
            frameColor=(1, 1, 1, 1),
            frameSize=(-0.65, 0.65, -0.18, 0.05),
            pos=(0, 0, -0.65),
            frameTexture=_TEX_VELLUM,
            relief=DGG.RAISED,
            borderWidth=(0.01, 0.01),
        )
        self._hint_frame.setTransparency(TransparencyAttrib.MAlpha)

        self._hint_text = OnscreenText(
            text='', pos=(0, -0.02), scale=0.045,
            fg=_HINT_FG, align=TextNode.ACenter,
            shadow=_SHADOW, mayChange=True,
            font=self._med_font,
            wordwrap=28,
            parent=self._hint_frame,
        )

        self._next_hint_btn = _tex_button(
            text='Next Hint (1/1)',
            font=self._med_font,
            pos=(0, 0, -0.88),
            command=self._on_next_hint_click,
            parent=base.aspect2d,
            scale=0.04,
            pad=(0.5, 0.25),
        )
        self._next_hint_btn.hide()
        self._hint_queue: list[str] = []
        self._hint_queue_index: int = 0

    def _destroy_ui(self):
        for elem in (self._obj_frame, self._hint_frame,
                      self._title_text, self._objective_text,
                      self._hint_text, self._next_hint_btn):
            if elem:
                try:
                    elem.destroy()
                except Exception:
                    pass
        self._obj_frame = None
        self._hint_frame = None
        self._title_text = None
        self._objective_text = None
        self._hint_text = None
        self._next_hint_btn = None
        self._destroy_transition_screen()

    def _update_title(self, battle: TutorialBattle):
        if self._title_text:
            label = (f"Tutorial: {battle.name}  "
                     f"({self.current_index + 1}/{len(self.battles)})")
            self._title_text.setText(label)

    def _update_objectives(self, battle: TutorialBattle):
        if self._objective_text:
            lines = []
            for obj in battle.objectives:
                text = obj.get('text', str(obj)) if isinstance(obj, dict) else str(obj)
                done = obj.get('_completed', False) if isinstance(obj, dict) else False
                marker = '[x]' if done else '[ ]'
                lines.append(f"  {marker} {text}")
            self._objective_text.setText(
                'Objectives:\n' + '\n'.join(lines) if lines else ''
            )

    def _show_hint(self, text: str):
        if self._hint_text:
            self._hint_text.setText(text)

    # ── Transition / completion screens ───────────────────────────────

    def _show_transition_screen(self, battle: TutorialBattle):
        """Show a 'battle won' overlay with a button to continue."""
        self._destroy_transition_screen()
        med_font = loader.loadFont(_FONT_PATH)

        self._transition_frame = DirectFrame(
            frameColor=(1, 1, 1, 1),
            frameSize=(-0.85, 0.85, -0.4, 0.4),
            pos=(0, 0, 0),
            frameTexture=_TEX_VICTORY,
            relief=DGG.RAISED,
            borderWidth=(0.015, 0.015),
        )
        self._transition_frame.setTransparency(TransparencyAttrib.MAlpha)

        OnscreenText(
            text='-- Victory! --',
            pos=(0, 0.22), scale=0.09,
            fg=_GOLD,
            shadow=_SHADOW,
            font=med_font,
            parent=self._transition_frame,
        )
        OnscreenText(
            text=f'{battle.name} complete.',
            pos=(0, 0.08), scale=0.055,
            fg=_INK,
            shadow=(0, 0, 0, 0.3),
            font=med_font,
            parent=self._transition_frame,
        )

        is_last = (self.current_index + 1 >= len(self.battles))
        btn_text = 'Finish Tutorial' if is_last else 'Next Battle >>'

        self._next_btn = _tex_button(
            text=btn_text,
            font=med_font,
            pos=(0, 0, -0.18),
            command=self._on_transition_click,
            parent=self._transition_frame,
            scale=0.06,
        )

    def _show_completion(self):
        """Tutorial finished — show final congratulations."""
        self._destroy_transition_screen()
        self._stop_win_check()
        med_font = loader.loadFont(_FONT_PATH)

        self._transition_frame = DirectFrame(
            frameColor=(1, 1, 1, 1),
            frameSize=(-0.95, 0.95, -0.45, 0.45),
            pos=(0, 0, 0),
            frameTexture=_TEX_VICTORY,
            relief=DGG.RAISED,
            borderWidth=(0.015, 0.015),
        )
        self._transition_frame.setTransparency(TransparencyAttrib.MAlpha)

        OnscreenText(
            text='-- Tutorial Complete! --',
            pos=(0, 0.22), scale=0.08,
            fg=_GOLD,
            shadow=_SHADOW,
            font=med_font,
            parent=self._transition_frame,
        )
        OnscreenText(
            text=(f'You have finished "{self.tutorial_name}".\n'
                  f'You are ready for a real battle, General!'),
            pos=(0, 0.04), scale=0.05,
            fg=_INK,
            shadow=(0, 0, 0, 0.3),
            font=med_font,
            parent=self._transition_frame,
            wordwrap=28,
        )
        self._next_btn = _tex_button(
            text='Close',
            font=med_font,
            pos=(0, 0, -0.22),
            command=self._on_finish_click,
            parent=self._transition_frame,
            normal=_TEX_BUTTON_RED,
            hover=_TEX_BUTTON_RED,
            scale=0.06,
        )

    def _destroy_transition_screen(self):
        if self._transition_frame:
            self._transition_frame.destroy()
            self._transition_frame = None
        if self._next_btn:
            self._next_btn = None

    def _on_transition_click(self):
        self._destroy_transition_screen()
        self.advance()

    def _on_finish_click(self):
        self._destroy_transition_screen()
        self.stop()

    # ──────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _write_temp_army(army_data: dict, filename: str) -> str:
        """Write army data to a temporary JSON file and return its path."""
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), filename
        )
        with open(path, 'w') as fh:
            json.dump(army_data, fh, indent=2)
        return path

    def get_progress(self) -> dict:
        """Return a summary dict of tutorial progress (for save/load)."""
        return {
            'filepath': self._filepath,
            'current_index': self.current_index,
            'active': self.active,
        }

    def restore_progress(self, progress: dict):
        """Resume a tutorial from saved progress."""
        fp = progress.get('filepath', '')
        idx = progress.get('current_index', 0)
        if fp and os.path.isfile(fp):
            self.start(fp, battle_index=idx)
