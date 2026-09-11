import math
import random
from models import *
from direct.fsm.FSM import FSM
from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode
from panda3d.core import Point3, TextNode, BitMask32, TextPropertiesManager, TextProperties, LineSegs
from rules_log import rule_log
from characters import JOIN_TAG
from skirmish import layout_positions

class unit:
    def __init__(self, name: str, model: model, nmodels: int, files: int, ranks: int):
        self.name = name
        self.nmodels = nmodels
        self.model = model
        self.files = files
        self.ranks = ranks

class unitGraphics(FSM):
    def __init__(self,Game, name, modelpath, unit=None, scale=1.0,BulletWorld=None,color=(1,0,0,1),bitmask=1):
        FSM.__init__(self, 'unitFSM')
        self.game=Game
        self.unitName=name
        self.unit =unit
        self.world=BulletWorld
        self.color=color
        self.bitmask=bitmask
        self.model = self.loadFigureModel(modelpath)
        self.model.setScale(scale)
        self.model.setColor(self.color)
        self.model.reparentTo(render)
        self.unitWidth=abs(self.model.getTightBounds()[1][0]-self.model.getTightBounds()[0][0])
        self.unitHeight=abs(self.model.getTightBounds()[1][1]-self.model.getTightBounds()[0][1])
        #self.model.ls()
        children = self.model.getChildren()
        #children.sort(key=lambda np: np.getName())
        #children[-1].removeNode()
        ranks=self.unit.ranks
        files=self.unit.files
        if self.unit.nmodels!=len(children):
            diffnmodel=self.unit.nmodels-len(children)
            for i in range(diffnmodel):
                clone=children[0].copyTo(self.model)
                children.append(clone)
        
        while len(children)>self.unit.nmodels:
            children[-1].removeNode()
            children = self.model.getChildren()

        self.modelWidth=abs(children[0].getTightBounds()[1][0]-children[0].getTightBounds()[0][0])
        self.modelHeight=abs(children[0].getTightBounds()[1][1]-children[0].getTightBounds()[0][1])
        # Prefer the database base size (mounted models use their mount's base).
        self.baseSize = self.unit.model.get_base_size() if self.unit and self.unit.model else None
        if self.baseSize:
            self.modelWidth = self.baseSize[0] / MM_PER_UNIT
            self.modelHeight = self.baseSize[1] / MM_PER_UNIT

        self.request('Idle')

        # Front-rank slot held by a joined character, if any. Read by
        # layOutRanks and slotCount, so it must exist before either runs.
        self.characterSlot = None

        # Skirmishers deploy as a loose blob (~1" apart), not rigid ranks/files.
        self.isSkirmisher = bool(self.unit and self.unit.model
                                 and self.unit.model.is_skirmisher())
        self.skirmishCombat = False
        self.skirmishLayout = []
        if self.isSkirmisher:
            side = max(1, math.ceil(math.sqrt(self.unit.nmodels)))
            gap = 0.6                                   # loose spacing (< 1")
            self._skirmSX = self.modelWidth + gap
            self._skirmSY = self.modelHeight + gap
            self._skirmCols = side
            self._skirmRows = math.ceil(self.unit.nmodels / side)
            self._arrange_skirmish_blob()
        else:
            self.layOutRanks(files, children)

        #self.unitWidth=abs(self.model.getTightBounds()[1][0]-self.model.getTightBounds()[0][0])
        #self.unitHeight=abs(self.model.getTightBounds()[1][1]-self.model.getTightBounds()[0][1])

        #self.model.setPos(self.unitWidth*2, -self.modelHeight/2,0)


        #self.model.setPos(35,0,0)
        self.setUpCollisions()
        self._varyModelTones()

        # Floating ring above the model to flag character units.
        self.characterMarker = None
        self._add_character_marker()

        #children[-1].removeNode()

        self.isInCombat=False
        self.isInCombatWith=[]
        self.isInCombatFlank=[]
        self.hasMovedThisTurn=False
        self.marchedThisTurn=False   # marched: no shooting or Magic Missiles (p. 123)
        self.hasAttackedThisTurn=False
        self.standAndShootWounds=0   # counts towards the combat that follows (p. 151)
        self.attemptedRallyThisTurn=False
        self.usedRallyingCry=False
        self.chargedThisTurn=False   # set when charging into combat; grants the charge bonus this turn
        self.wasChargedThisTurn=False
        self.countsAsChargeTargetNextTurn=False
        self.countsAsChargedNextTurn=False  # caught a unit that fell back; the locked combat is fought next turn
        self.cannotPursueThisTurn=False  # joined a new combat mid-phase; restrains and reforms instead (p. 157)
        self.chargeDistance=0.0      # inches actually covered by that charge (Impact Hits need 3"+)
        self.cannotChargeThisTurn=False  # set on rally (Fall Back in Good Order); no charge this turn
        self.moveSpentThisTurn=0.0   # inches of the allowance a manoeuvre has already used
        self.manoeuvreThisTurn=None  # a unit may perform ONE manoeuvre per move (p. 124)
        self.redressDelta=0          # models moved to/from the front rank so far, max 5 (p. 125)
        self.isChargingMove=False    # true while making a charge move (exempt from Panic)
        self.panicTestedThisPhase=False  # one Panic test per phase (No Need for Hysterics)
        self.fledThisPhase=False         # one flee move per phase (The Limits of Endurance)
        self.usedStubborn=False      # Stubborn may refuse only the FIRST Break test of the battle
        self.usedShieldwall=False
        self.isDisrupted=False       # a quarter or more of the models in difficult terrain: no Rank Bonus
        self.spellsCastThisTurn=[]   # a Wizard may attempt each spell once, and only Level of them
        self.boundSpellPhases=[]    # one Bound attempt per phase, independent of Wizard slots (p. 109)
        self.cannotCastThisTurn=False  # spent by a Miscast result of 8+
        self.isGeneral=False         # army commander; radiates Inspiring Presence
        self.isBSB=False             # carries the Battle Standard (Hold Your Ground)
        self.woundsOnModel=0         # unsaved wounds on the current multi-wound model
        self.startOfBattleModels=self.unit.nmodels  # drives the 50% flee/fall-back split
        self.startOfPhaseModels=self.unit.nmodels  # drives the 25% heavy-casualties check
        self.startOfPhaseEngaged=False  # was already fighting when the phase began (p. 157)
        self.endedInUnit=False
        self.tacticalRole=None      # set by AI: e.g. {'role': 'CHARGE', 'target': '...', 'reason': '...'}
        self.madePursuitChoice=False
        self.isDeployed=False
        self.scoutDeploymentChoice=None  # None, 'normal', or 'scouts'
        self.deployedAsScouts=False      # deployment history; survives turn resets (p. 177)
        self.vanguardDone=False
        self.madeVanguardMove=False
        text=f"{self.isInCombat}\n{self.hasMovedThisTurn}\n{self.hasAttackedThisTurn}"
        
        """ self.text_node = OnscreenText(
            text=text,
            scale=scale,
            fg=color,
            align=0,  # Center alignment
            mayChange=True
        ) """
        self.text = TextNode('node name')
        # Register inline colour properties (done once per process via global manager)
        _tpm = TextPropertiesManager.getGlobalPtr()
        if not _tpm.hasProperties('stat_low'):
            _tp_low = TextProperties()
            _tp_low.setTextColor(1.0, 0.25, 0.25, 1.0)   # red for below-average
            _tpm.setProperties('stat_low', _tp_low)
        if not _tpm.hasProperties('stat_high'):
            _tp_high = TextProperties()
            _tp_high.setTextColor(0.4, 1.0, 0.4, 1.0)    # green for above-average
            _tpm.setProperties('stat_high', _tp_high)
        self.text.setText(text)
        # Text only: the HUD draws it in screen space on hover. As a billboard
        # on the unit it ran off the bottom edge when the unit was low down.

        
    # Average human-level baseline for above-average detection
    _STAT_AVERAGE = {'M': 4, 'WS': 3, 'BS': 3, 'S': 3, 'T': 3, 'W': 1, 'I': 3, 'A': 1, 'Ld': 7}
    _STAT_KEYS    = ['M', 'WS', 'BS', 'S', 'T', 'W', 'I', 'A', 'Ld']
    _COL_W        = 4   # nominal column width, for separator and wrap widths only

    def _stat_table(self, label: str, ch: dict) -> str:
        """Return a two-line stat table (header + values) for the given characteristics dict.
        Stats above average are marked '+' in green; below average are marked '-' in red.

        Columns are separated by tabs, not padded with spaces: the HUD renders
        this text in the proportional theme font, where a space is narrower
        than a digit and no amount of padding lines the rows up. The tab stop
        is set by HUD.TIP_TAB_WIDTH.
        """
        header = f"{label}\n"
        hdr_row = "\t".join(self._STAT_KEYS)
        val_parts = []
        for k in self._STAT_KEYS:
            raw = ch.get(k, '-')
            try:
                val = int(raw)
                avg = self._STAT_AVERAGE.get(k, 99)
                if val > avg:
                    cell = f"\x01stat_high\x01{raw}+\x02"
                elif val < avg:
                    cell = f"\x01stat_low\x01{raw}-\x02"
                else:
                    cell = str(raw)
            except (ValueError, TypeError):
                cell = str(raw)
            val_parts.append(cell)
        val_row = "\t".join(val_parts)
        return header + hdr_row + "\n" + val_row + "\n"

    def _weapon_desc(self, w: dict) -> str:
        """One-line weapon summary: name + stats (+ weapon special rules)."""
        name = w.get('name', 'weapon')
        if w.get('tag') == 'ranged':
            shots = w.get('ranged_shots_dice') or w.get('ranged_shots', 1)
            stats = (f"R{w.get('ranged_range', '-')} S{w.get('ranged_strength', '-')} "
                     f"AP{w.get('ranged_AP', 0)} x{shots}")
        else:
            parts = []
            if w.get('strength'):
                parts.append(f"S:{w['strength']}")
            if w.get('ap'):
                parts.append(f"AP:{w['ap']}")
            stats = " ".join(parts) if parts else "Combat"
        rules = w.get('special_rules') or []
        if rules:
            stats += " [" + ", ".join(rules) + "]"
        return f"  {name}: {stats}"

    def _wrap(self, items, width: int) -> str:
        """Comma-join items and word-wrap to the given display width."""
        words = ", ".join(items).split(", ")
        wrapped, cur = "", ""
        for word in words:
            candidate = (cur + ", " + word) if cur else word
            if len(candidate) <= width:
                cur = candidate
            else:
                wrapped += cur + "\n"
                cur = word
        if cur:
            wrapped += cur + "\n"
        return wrapped

    def updateTextNode(self):
        sep = "-" * (self._COL_W * len(self._STAT_KEYS))
        row = f"[ {self.unitName} ]\n"
        row += sep + "\n"

        if self.unit and self.unit.model and self.unit.model.characteristics:
            m = self.unit.model
            row += self._stat_table(m.name, m.characteristics)

            # Mount section — found via special_rules tag 'mount'
            mount_rule = next(
                (r for r in m.special_rules if r.get('tag') == 'mount' and r.get('mountUnit')),
                None
            )
            if mount_rule:
                mount_obj = mount_rule['mountUnit']
                # mountUnit may be a model (has .characteristics) or a unit (has .model)
                if hasattr(mount_obj, 'characteristics'):
                    mount_name = mount_obj.name
                    mount_ch   = mount_obj.characteristics
                elif hasattr(mount_obj, 'model') and mount_obj.model:
                    mount_name = mount_obj.name
                    mount_ch   = mount_obj.model.characteristics
                else:
                    mount_name, mount_ch = None, None
                if mount_ch:
                    row += sep + "\n"
                    row += self._stat_table(mount_name, mount_ch)

            # Weapons — each weapon with its stats and weapon special rules.
            weapon_ids = {id(w) for w in m.weapons.values()}
            weapon_lines, seen_w = [], set()
            for w in m.weapons.values():
                key = (w.get('name') or '').lower()
                if key in seen_w:
                    continue
                seen_w.add(key)
                weapon_lines.append(self._weapon_desc(w))
            if weapon_lines:
                row += sep + "\n"
                row += "Weapons:\n"
                row += "\n".join(weapon_lines) + "\n"

            # Armour — equipped pieces and the resulting save (7 = no save).
            save = getattr(m, 'armor_save', 7)
            armour = getattr(m, 'armour', None) or []
            row += sep + "\n"
            save_str = f"{save}+" if save <= 6 else "none"
            if armour:
                row += "Armour : " + ", ".join(armour) + "\n"
            row += f"Save   : {save_str}\n"

            # Special rules — unit rules from the catalogue plus any coded rules,
            # excluding the weapons (listed above) and the mount.
            rule_names = list(m.characteristics.get('Special Rules', []) or [])
            for r in m.special_rules:
                if not isinstance(r, dict) or id(r) in weapon_ids:
                    continue
                # The mount and the join marker have their own lines already.
                if r.get('tag') in ('mount', JOIN_TAG):
                    continue
                name = r.get('name')
                if name and name not in rule_names:
                    rule_names.append(name)
            if rule_names:
                row += sep + "\n"
                row += "Special Rules:\n"
                row += self._wrap(rule_names, self._COL_W * len(self._STAT_KEYS))

        row += sep + "\n"

        # Combat state
        combat_str   = "Yes" if self.isInCombat          else "No"
        moved_str    = "Yes" if self.hasMovedThisTurn     else "No"
        attacked_str = "Yes" if self.hasAttackedThisTurn  else "No"
        row += f"State  : {self.state}\n"
        row += f"Combat : {combat_str:<3}  Moved : {moved_str:<3}  Atk : {attacked_str}\n"

        if self.isGeneral:
            row += "Command: General\n"
        elif self.isBSB:
            row += "Command: Battle Standard\n"
        else:
            psy = getattr(self.game, 'psychology', None) if self.game else None
            if psy is not None:
                ld, general = psy.leadership_of(self)
                if general is not None:
                    row += f"Command: Ld {ld} from {general.unitName}\n"
                bsb = psy.battle_standard_of(self)
                if bsb is not None:
                    row += f"Standard: re-rolls from {bsb.unitName}\n"

        if self.isInCombatWith:
            names = ", ".join(u.unitName for u in self.isInCombatWith)
            row += f"Vs     : {names}\n"
        if self.isInCombatFlank:
            row += f"Flanks : {self.isInCombatFlank}\n"

        char = getattr(self, 'joinedCharacter', None)
        if char is not None:
            note = " (retired from combat)" if getattr(
                char, 'retiredFromCombat', False) else ""
            row += f"Joined : {char.unitName}{note} — hover its base\n"
        host = getattr(self, 'hostUnit', None)
        if host is not None:
            row += f"Joined : riding with {host.unitName}\n"
        from command_groups import living_command
        command = living_command(self)
        if command:
            row += 'Command: ' + ', '.join(entry.get('name', entry['role']) +
                                         (' (retired)' if entry.get('retired') else '')
                                         for entry in command) + '\n'

        if self.tacticalRole:
            role_str = self.tacticalRole['role']
            if self.tacticalRole.get('target'):
                role_str += f" -> {self.tacticalRole['target']}"
            row += f"Role   : {role_str}\n"

        self.text.setText(row)

    def layOutRanks(self, files=None, children=None):
        """Place the unit's models in ranks and files, front rank first.

        A joined character holds a slot of its own, which the unit's models step
        around: it stands *in* the front rank, and the model it displaces falls
        through to the back rather than the character forming a rank in front.
        """
        if getattr(self, 'isSkirmisher', False) and not getattr(self, 'skirmishCombat', False):
            self.applySkirmishLayout()
            return
        files = max(1, int(files if files is not None else self.unit.files))
        children = self.model.getChildren() if children is None else children
        from command_groups import command_positions, living_command
        command = living_command(self)
        positions = command_positions(self, files)
        occupied = set(positions.values())
        if self.characterSlot is not None:
            candidates = [slot for slot in range(max(len(children), files) + 1) if slot not in occupied]
            if getattr(self, 'characterCombatReturnSlot', None) is None or self.characterSlot not in candidates:
                self.characterSlot = min(candidates, key=lambda slot: (slot // files, abs(slot % files - files // 2)))
        reserved = self.characterSlot
        slot = 0
        for index, child in enumerate(children):
            child.clearPythonTag('command_role')
            if index < len(command):
                entry = command[index]
                place = positions[id(entry)]
                child.setPythonTag('command_role', entry['role'])
            else:
                while slot == reserved or slot in occupied:
                    slot += 1
                place = slot
                slot += 1
            row, col = divmod(place, files)
            child.setPos(Point3(col * self.modelWidth, -row * self.modelHeight, 0))

    def slotCount(self):
        """Grid slots the unit fills: its own models, plus a joined character."""
        return self.unit.nmodels + (1 if self.characterSlot is not None else 0)

    def placeCharacter(self):
        """Sit a joined character in the slot the ranks were laid out around."""
        char = getattr(self, 'joinedCharacter', None)
        if char is None or self.characterSlot is None or char.bodyNP.isEmpty():
            return
        if self.isSkirmisher and not self.skirmishCombat:
            position = getattr(self, 'skirmishCharacterPosition', None)
            if position is None:
                rightmost = max(self.skirmishLayout, key=lambda record: record['x'])
                position = [rightmost['x'] + (self.modelWidth + char.modelWidth) / 2 + 0.6,
                            rightmost['y']]
                self.skirmishCharacterPosition = position
            char.bodyNP.setPos(position[0], position[1], 0)
            return
        files = max(1, self.unit.files)
        if getattr(char, 'retiredFromCombat', False):
            # Refused a challenge and hid in the rear ranks (p. 210).
            row = max(1, -(-max(1, self.unit.nmodels) // files))
            col = files // 2
        else:
            row, col = divmod(self.characterSlot, files)
        char.bodyNP.setPos(self.model.getPos()
                           + Point3(col * self.modelWidth,
                                    -row * self.modelHeight, 0))

    def rebuildFootprint(self):
        """Resize the collision box to the current formation.

        A Bullet shape cannot be resized in place, so the body leaves the world,
        swaps its box, and goes back in.
        """
        if not self.world or self.bodyNP.isEmpty() or self.model.isEmpty():
            return
        self.world.removeRigidBody(self.bodyNP.node())
        for shape in self.bodyNP.node().shapes:
            self.bodyNP.node().removeShape(shape)
        box_size = self.footprintSize()
        self.bodyNP.node().addShape(BulletBoxShape(box_size * 0.5))
        self.bodyNP.node().setMass(0)
        self.world.attachRigidBody(self.bodyNP.node())
        self.applyFootprint(box_size)

    def footprintSize(self):
        """Collision-box size for the unit's current formation.

        Taken from the models' bases when the catalogue provides base data, so
        the box matches the frontage the unit actually occupies rather than the
        extents of its mesh.
        """
        bounds = self.model.getTightBounds()
        box_size = bounds[1] - bounds[0]
        # Skirmishers occupy a loose blob; size the footprint to cover it.
        if getattr(self, 'isSkirmisher', False) and not self.skirmishCombat:
            box_size.setX(max((abs(record['x']) + self.modelWidth / 2
                               for record in self.skirmishLayout), default=self.modelWidth / 2) * 2)
            box_size.setY(max((abs(record['y']) + self.modelHeight / 2
                               for record in self.skirmishLayout), default=self.modelHeight / 2) * 2)
            character = getattr(self, 'joinedCharacter', None)
            position = getattr(self, 'skirmishCharacterPosition', None)
            if character is not None and position is not None:
                box_size.setX(max(box_size.x, 2 * abs(position[0]) + character.modelWidth))
                box_size.setY(max(box_size.y, 2 * abs(position[1]) + character.modelHeight))
        elif getattr(self, 'baseSize', None):
            files = max(1, self.unit.files)
            slots = self.slotCount()
            cols = min(files, slots)
            rows = (slots + files - 1) // files
            box_size.setX(self.modelWidth * cols)
            box_size.setY(self.modelHeight * rows)
        return box_size

    def applyFootprint(self, box_size):
        """Update everything derived from the footprint: the measured width and
        depth, the front/back markers and where the models sit in the box."""
        self.bodyNPfront.setPos(0, box_size.y * 0.45, 0)
        self.bodyNPback.setPos(0, -box_size.y * 0.45, 0)
        self.unitWidth = box_size.x * self.bodyNP.getScale().x
        self.unitHeight = box_size.y * self.bodyNP.getScale().y
        if self.isSkirmisher and not self.skirmishCombat:
            self.model.setPos(0, 0, 0)
            self.applySkirmishLayout()
            return
        self.model.setPos(-box_size.x / 2 + self.modelWidth / 2,
                          box_size.y / 2 - self.modelHeight / 2, 0)

    def setUpCollisions(self):
        if self.world:
            box_size = self.footprintSize()
            shape = BulletBoxShape(box_size * 0.5)  # BulletBoxShape takes half-extents
            body = BulletRigidBodyNode('UnitCollision-' + self.unitName)
            body.addShape(shape)
            body.setMass(0)  # Static object
            #body = BulletCharacterControllerNode(shape, 0.4, 'UnitCollision-' + self.unitName)
            
            
            self.bodyNP = render.attachNewNode(body)
            self.bodyNPfront = self.bodyNP.attachNewNode("front")
            self.bodyNPback = self.bodyNP.attachNewNode("back")
            self.bodyNP.setCollideMask(BitMask32.bit(1))
            self.world.attachRigidBody(body)
            #self.world.attachCharacter(self.bodyNP.node())

            self.model.node().setName('Model-' + self.unitName)
            self.model.reparentTo(self.bodyNP)
            self.bodyNP.setScale(1.0)
            self.applyFootprint(box_size)
            #self.model.flattenLight()
    
    def loadFigureModel(self, modelpath):
        return loader.loadModel(modelpath)

    def _varyModelTones(self):
        """Give each miniature its own tone, so a regiment reads as many models
        rather than one solid block of colour.

        Colour *scale*, not colour: the game sets the models' colour outright to
        grey out an illegal placement and to flag a shooting target, and a scale
        multiplies that instead of fighting it.
        """
        # Seeded with the name itself, not hash(): str hashing is salted per
        # process, so the pattern would be reshuffled on every launch.
        rng = random.Random(self.unitName)
        for child in self.model.getChildren():
            v = rng.uniform(0.80, 1.14)
            # A little warm/cool drift as well, or it reads as a brightness ramp.
            child.setColorScale(v * rng.uniform(0.98, 1.05),
                                v,
                                v * rng.uniform(0.95, 1.02), 1.0)

    def _add_character_marker(self):
        """Attach a small billboarded ring above the model for character units."""
        ch = self.unit.model.characteristics if self.unit and self.unit.model else {}
        if str(ch.get('Category', '')).strip().lower() != 'characters':
            return

        radius = max(self.modelWidth, self.modelHeight) * 0.4
        segs = LineSegs()
        segs.setColor(1.0, 0.85, 0.1, 1.0)   # gold
        segs.setThickness(3.0)
        steps = 32
        for i in range(steps + 1):
            a = (i / steps) * 2 * math.pi
            x = radius * math.cos(a)
            z = radius * math.sin(a)
            (segs.moveTo if i == 0 else segs.drawTo)(x, 0, z)

        self.characterMarker = self.bodyNP.attachNewNode(segs.create())
        try:
            top_z = self.model.getTightBounds(self.bodyNP)[1].z
        except Exception:
            top_z = self.unitHeight
        self.characterMarker.setPos(0, 0, top_z + radius + 1)
        self.characterMarker.setBillboardPointEye()
        self.characterMarker.setLightOff()
        self.characterMarker.setBin("fixed", 0)
        self.characterMarker.setDepthWrite(False)

    def enterIdle(self):
        #messenger.send('unit-move-complete')
        self.hasMovedThisTurn=False
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[],appendTask=False)

    def enterMoved(self):
        self.hasMovedThisTurn=True
        if not base.resolvingCombat:
            messenger.send('unit-move-complete')
        else:
            print(f"WARNING: {self.unitName} entered Moved during resolvingCombat, signal deferred to combat resolver")
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[],appendTask=False)

    def enterInCombat(self):
        self.isInCombat=True
        self.formUpForCombat()
        if not base.resolvingCombat:
            messenger.send('unit-move-complete')
        else:
            print(f"WARNING: {self.unitName} entered InCombat during resolvingCombat, signal deferred to combat resolver")
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[], appendTask=False)
    
    def exitInCombat(self):
        self.isInCombat=False
        self.isInCombatWith=[]
        self.isInCombatFlank=[]
        self.roundsFought=0
        from characters import return_through_ranks
        return_through_ranks(self)
        # A challenge lasts only as long as the combat that held it (p. 211).
        for challenge in list(getattr(base, 'challenges', None) or []):
            if self in challenge.hosts():
                base.challenges.remove(challenge)
        # A model that refused a challenge may rejoin the rank once its unit is
        # no longer engaged (p. 210).
        for entry in getattr(self.unit, 'command', []):
            if entry.get('retired'):
                entry['retired'] = False
                rule_log('Refusing a Challenge', self,
                         f"{entry.get('name', 'Champion')} returns to the fighting rank; combat ended (p. 210)")
        char = getattr(self, 'joinedCharacter', None)
        if char is not None and getattr(char, 'retiredFromCombat', False):
            char.retiredFromCombat = False
            from spell_effects import refresh_self_spells
            refresh_self_spells(char)
            self.placeCharacter()
            rule_log('Refusing a Challenge', char,
                     "its unit is no longer engaged, so it returns to the "
                     "fighting rank (p. 210)")
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[], appendTask=False)

    def _arrange_skirmish_blob(self):
        """Create a coherent starting layout, without random gaps (p. 184)."""
        positions = layout_positions(len(self.model.getChildren()),
                                     self.modelWidth, self.modelHeight)
        self.skirmishLayout = [dict(id=index, x=position[0], y=position[1])
                               for index, position in enumerate(positions)]
        self.applySkirmishLayout()

    def applySkirmishLayout(self):
        """Project saved base positions onto miniatures; terrain Z is derived."""
        parent = getattr(self, 'bodyNP', self.model)
        for child, record in zip(self.model.getChildren(), self.skirmishLayout):
            height = child.getZ(parent)
            child.setPos(parent, record['x'], record['y'], height)
            child.setPythonTag('skirmish_id', record['id'])

    def restoreSkirmishLayout(self, saved=None):
        """Restore exact model identities/positions, or migrate an older save."""
        self.isSkirmisher = self.unit.model.is_skirmisher()
        children = list(self.model.getChildren())
        while len(children) < self.unit.nmodels:
            children.append(children[0].copyTo(self.model))
        for child in children[self.unit.nmodels:]:
            child.removeNode()
        self.skirmishCombat = bool(saved.get('combat', False)) if saved else self.isInCombat
        self.skirmishCharacterPosition = saved.get('character') if saved else None
        if self.isSkirmisher:
            if saved is not None:
                self.skirmishLayout = [dict(record) for record in saved['models']]
            else:
                self._arrange_skirmish_blob()
        if self.isSkirmisher and not self.skirmishCombat:
            self.applySkirmishLayout()
        else:
            self.layOutRanks()
        self.placeCharacter()
        self.rebuildFootprint()

    def savedSkirmishLayout(self):
        if not self.isSkirmisher:
            return None
        return dict(models=[dict(record) for record in self.skirmishLayout],
                    combat=self.skirmishCombat,
                    character=getattr(self, 'skirmishCharacterPosition', None))

    def formUpForCombat(self):
        """Snap a skirmisher's models into a tight fighting rank for combat."""
        if not getattr(self, 'isSkirmisher', False) or self.skirmishCombat:
            return
        self.skirmishCombat = True
        children = self.model.getChildren()
        self.layOutRanks(min(self.unit.files or 5, len(children)), children)
        self.rebuildFootprint()

    def spreadToSkirmish(self):
        """Separate current bases (p. 185); house rule: fleeing units wait for Rally."""
        from math import hypot
        from rules_log import rule_skipped
        from skirmish import separated_positions
        if (not getattr(self, 'isSkirmisher', False) or not self.skirmishCombat
                or self.bodyNP.isEmpty() or getattr(self, 'hostUnit', None) is not None):
            return
        if self.isInCombat or self.state == 'IsFleeing':
            reason = ('still engaged (p. 185)' if self.isInCombat else
                      'still fleeing; waits for a successful Rally (house rule)')
            rule_skipped('Skirmishers', self, f'{reason}; keeps compact fighting ranks')
            return
        children = list(self.model.getChildren())
        positions = [child.getPos(self.bodyNP) for child in children]
        boxes = [(position.x, position.y, self.modelWidth / 2, self.modelHeight / 2, 0)
                 for position in positions]
        character = getattr(self, 'joinedCharacter', None)
        if character is not None:
            position = character.bodyNP.getPos(self.bodyNP)
            boxes.append((position.x, position.y, character.modelWidth / 2, character.modelHeight / 2, 0))
        try:
            separated = separated_positions(boxes)
        except ValueError as error:
            rule_skipped('Skirmishers', self, f'cannot separate bases: {error}; keeps compact formation')
            return
        for record, position in zip(self.skirmishLayout, separated):
            record['x'], record['y'] = position
        if character is not None:
            self.skirmishCharacterPosition = list(separated[-1])
        self.skirmishCombat = False
        self.applySkirmishLayout()
        self.placeCharacter()
        self.rebuildFootprint()
        scale = self.bodyNP.getScale(render)
        longest = max((hypot((after[0] - before[0]) * scale.x,
                             (after[1] - before[1]) * scale.y)
                       for before, after in zip(boxes, separated)), default=0.0)
        rule_log('Skirmishers', self,
                 f'unengaged and not fleeing: separates {len(boxes)} bases; '
                 f'largest adjustment {longest:.5f}"; preserves layout and facing (p. 185)')
        self.updateTextNode()

    def enterIsFleeing(self):
        self.attemptedRallyThisTurn=False
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[], appendTask=False)
        pass

    def enterIsPursuing(self):
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[], appendTask=False)
        pass
