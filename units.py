import math
import random
from models import *
from direct.fsm.FSM import FSM
from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode
from panda3d.core import Point3, TextNode, BitMask32, TextPropertiesManager, TextProperties, LineSegs

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
        self.model = loader.loadModel(modelpath)
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

        # Skirmishers deploy as a loose blob (~1" apart), not rigid ranks/files.
        self.isSkirmisher = bool(self.unit and self.unit.model
                                 and self.unit.model.is_skirmisher())
        if self.isSkirmisher:
            side = max(1, math.ceil(math.sqrt(self.unit.nmodels)))
            gap = 0.6                                   # loose spacing (< 1")
            self._skirmSX = self.modelWidth + gap
            self._skirmSY = self.modelHeight + gap
            self._skirmCols = side
            self._skirmRows = math.ceil(self.unit.nmodels / side)
            self._arrange_skirmish_blob()
        else:
            for i, child in enumerate(children):
                row = i // files
                col = i % files
                p=Point3(col * (self.modelWidth ),-row * (self.modelHeight ), 0)
                pp=p-Point3(self.unitWidth*2, -self.modelHeight/2,0)
                child.setPos(p)
            #child.setPos((col - (files - 1) / 2) * (self.modelWidth / files), (row - (ranks - 1) / 2) * (self.modelHeight / ranks), 0)

        #self.unitWidth=abs(self.model.getTightBounds()[1][0]-self.model.getTightBounds()[0][0])
        #self.unitHeight=abs(self.model.getTightBounds()[1][1]-self.model.getTightBounds()[0][1])

        #self.model.setPos(self.unitWidth*2, -self.modelHeight/2,0)


        #self.model.setPos(35,0,0)
        self.setUpCollisions()

        # Floating ring above the model to flag character units.
        self.characterMarker = None
        self._add_character_marker()

        #children[-1].removeNode()

        self.isInCombat=False
        self.isInCombatWith=[]
        self.isInCombatFlank=[]
        self.hasMovedThisTurn=False
        self.hasAttackedThisTurn=False
        self.attemptedRallyThisTurn=False
        self.chargedThisTurn=False   # set when charging into combat; grants the charge bonus this turn
        self.chargeDistance=0.0      # inches actually covered by that charge (Impact Hits need 3"+)
        self.cannotChargeThisTurn=False  # set on rally (Fall Back in Good Order); no charge this turn
        self.isChargingMove=False    # true while making a charge move (exempt from Panic)
        self.panicTestedThisPhase=False  # one Panic test per phase (No Need for Hysterics)
        self.usedStubborn=False      # Stubborn may refuse only the FIRST Break test of the battle
        self.isGeneral=False         # army commander; radiates Inspiring Presence
        self.isBSB=False             # carries the Battle Standard (Hold Your Ground)
        self.woundsOnModel=0         # unsaved wounds on the current multi-wound model
        self.startOfBattleModels=self.unit.nmodels  # drives the 50% flee/fall-back split
        self.startOfPhaseModels=self.unit.nmodels  # drives the 25% heavy-casualties check
        self.endedInUnit=False
        self.tacticalRole=None      # set by AI: e.g. {'role': 'CHARGE', 'target': '...', 'reason': '...'}
        self.madePursuitChoice=False
        self.isDeployed=False
        text=f"{self.isInCombat}\n{self.hasMovedThisTurn}\n{self.hasAttackedThisTurn}"
        
        """ self.text_node = OnscreenText(
            text=text,
            scale=scale,
            fg=color,
            align=0,  # Center alignment
            mayChange=True
        ) """
        self.text = TextNode('node name')
        mono_font = loader.loadFont('cmtt12')
        if mono_font:
            self.text.setFont(mono_font)
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
        #self.text_node = self.model.attachNewNode(self.text)
        self.text_node = self.bodyNP.attachNewNode(self.text)
        self.text_node.setPos(self.unitWidth/3, self.unitHeight*2, 5)
        self.text_node.setScale(0.06)
        self.text_node.setBillboardPointEye(-5, fixed_depth=True)
        self.text_node.setBin("fixed", 0)
        self.text_node.setDepthWrite(False)
        self.text_node.setDepthTest(False)
        self.text_node.hide()

        
    # Average human-level baseline for above-average detection
    _STAT_AVERAGE = {'M': 4, 'WS': 3, 'BS': 3, 'S': 3, 'T': 3, 'W': 1, 'I': 3, 'A': 1, 'Ld': 7}
    _STAT_KEYS    = ['M', 'WS', 'BS', 'S', 'T', 'W', 'I', 'A', 'Ld']
    _COL_W        = 4   # fixed column width for monospace alignment

    def _stat_table(self, label: str, ch: dict) -> str:
        """Return a two-line stat table (header + values) for the given characteristics dict.
        Stats above average are marked '+' in green; below average are marked '-' in red.
        """
        cw = self._COL_W
        header = f"{label}\n"
        hdr_row = "".join(f"{k:^{cw}}" for k in self._STAT_KEYS)
        val_parts = []
        for k in self._STAT_KEYS:
            raw = ch.get(k, '-')
            try:
                val = int(raw)
                avg = self._STAT_AVERAGE.get(k, 99)
                if val > avg:
                    visible = str(raw) + '+'
                    cell = f"\x01stat_high\x01{visible}\x02"
                elif val < avg:
                    visible = str(raw) + '-'
                    cell = f"\x01stat_low\x01{visible}\x02"
                else:
                    visible = str(raw)
                    cell = visible
            except (ValueError, TypeError):
                visible = str(raw)
                cell = visible
            # Centre-pad using the visible text length (markup bytes have no display width)
            padding = max(0, cw - len(visible))
            lpad = padding // 2
            rpad = padding - lpad
            val_parts.append(' ' * lpad + cell + ' ' * rpad)
        val_row = "".join(val_parts)
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
                if not isinstance(r, dict) or id(r) in weapon_ids or r.get('tag') == 'mount':
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

        if self.tacticalRole:
            role_str = self.tacticalRole['role']
            if self.tacticalRole.get('target'):
                role_str += f" -> {self.tacticalRole['target']}"
            row += f"Role   : {role_str}\n"

        self.text.setText(row)

    def footprintSize(self):
        """Collision-box size for the unit's current formation.

        Taken from the models' bases when the catalogue provides base data, so
        the box matches the frontage the unit actually occupies rather than the
        extents of its mesh.
        """
        bounds = self.model.getTightBounds()
        box_size = bounds[1] - bounds[0]
        # Skirmishers occupy a loose blob; size the footprint to cover it.
        if getattr(self, 'isSkirmisher', False):
            box_size.setX(self._skirmSX * self._skirmCols)
            box_size.setY(self._skirmSY * self._skirmRows)
        elif getattr(self, 'baseSize', None):
            files = max(1, self.unit.files)
            cols = min(files, self.unit.nmodels)
            rows = (self.unit.nmodels + files - 1) // files
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
        self.spreadToSkirmish()
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[], appendTask=False)

    def _arrange_skirmish_blob(self):
        """Scatter the models into the loose skirmish blob (deterministic)."""
        children = self.model.getChildren()
        side = getattr(self, '_skirmCols', 1)
        rng = random.Random(hash(self.unitName) & 0xffffffff)
        for i, child in enumerate(children):
            row = i // side
            col = i % side
            jx = rng.uniform(-0.25, 0.25)
            jy = rng.uniform(-0.25, 0.25)
            child.setPos(Point3(col * self._skirmSX + jx,
                                -row * self._skirmSY + jy, 0))

    def formUpForCombat(self):
        """Snap a skirmisher's models into a tight fighting rank for combat."""
        if not getattr(self, 'isSkirmisher', False):
            return
        children = self.model.getChildren()
        files = max(1, min(self.unit.files or 5, len(children)))
        for i, child in enumerate(children):
            row = i // files
            col = i % files
            child.setPos(Point3(col * self.modelWidth, -row * self.modelHeight, 0))

    def spreadToSkirmish(self):
        """Return a skirmisher's models to the loose blob after combat."""
        if getattr(self, 'isSkirmisher', False):
            self._arrange_skirmish_blob()

    def enterIsFleeing(self):
        self.attemptedRallyThisTurn=False
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[], appendTask=False)
        pass

    def enterIsPursuing(self):
        taskMgr.doMethodLater(0.1, self.updateTextNode, "updateTextNode",extraArgs=[], appendTask=False)
        pass
