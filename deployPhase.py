from panda3d.core import Point3, BitMask32
import random
from strategyAdvisor import StrategyAdvisor
from unitTypeClassifier import UnitTypeClassifier, UnitType, SupportRole
from characters import is_character, has_joined_character, same_player, join_unit


# ── Deployment zone ───────────────────────────────────────────────────

# One world unit is one inch. The zone runs the full width of the 72x48 board
# and 12 deep off the owning player's edge; game_fsm builds the frame that
# encloses it from these, and moves it between the two players.
DEPLOY_ZONE_WIDTH = 72
DEPLOY_ZONE_DEPTH = 12

# Undeployed units wait beside the board, never on it: a unit sitting inside
# the zone at the start of the phase reads as one that has already been placed.
# Player one queues west of the zone, player two east.
STAGING_MARGIN = 2.0    # clear space between the board edge and a waiting unit
STAGING_GAP = 1.0       # between units down the column


def stage_undeployed(game):
    """Queue each player's undeployed units up beside their deployment zone.

    Units are lined up by their inner face rather than their centre, so a wide
    regiment and a narrow one leave the same clearance from the board edge.
    """
    x_face = DEPLOY_ZONE_WIDTH / 2 + STAGING_MARGIN
    for units, side in ((game.player1Units, -1), (game.player2Units, 1)):
        # Start level with the outer edge of that player's zone and march in.
        y = side * 2 * DEPLOY_ZONE_DEPTH
        for unit in units:
            if unit.isDeployed:
                continue
            width = getattr(unit, 'unitWidth', 4.0) or 4.0
            depth = getattr(unit, 'unitHeight', 4.0) or 4.0
            unit.bodyNP.setPos(side * (x_face + width / 2),
                               y - side * depth / 2, 0)
            unit.bodyNP.node().setTransformDirty()
            y -= side * (depth + STAGING_GAP)


# ── Strategy-aware deployment for AI ──────────────────────────────────

# Cache to persist the chosen strategy across deploy calls within a game
_deploy_strategy_cache = {}


def _get_ai_deploy_position(game, unit):
    """
    Use the StrategyAdvisor to pick a smart deploy position for *unit*
    based on the chosen army strategy and the opponent's already-deployed
    units.  Prints a short reasoning line to the terminal.
    """
    classifier = UnitTypeClassifier()
    advisor = StrategyAdvisor(classifier)

    my_units = game.player2Units
    enemy_units = game.player1Units

    # Extract model objects from unitGraphics for the classifier
    my_models = [u.unit.model for u in my_units]
    enemy_models = [u.unit.model for u in enemy_units]

    # ── 1. Pick / recall the army strategy ────────────────────────────
    if 'strategy' not in _deploy_strategy_cache:
        strats = advisor.recommend_strategies(my_models, from_dict=False)
        best_strat, fit = strats[0] if strats else (None, 0)
        _deploy_strategy_cache['strategy'] = best_strat
        _deploy_strategy_cache['fit'] = fit

        # Analyse matchup against the enemy
        matchup = advisor.analyse_matchup(my_models, enemy_models, from_dict=False)
        _deploy_strategy_cache['matchup'] = matchup

        print("\n" + "=" * 60)
        print(f"  AI DEPLOY — Strategy: {best_strat.name} (fit {fit:.0%})")
        print(f"  {best_strat.description}")
        print(f"  Matchup verdict: {matchup['verdict']}")
        print("=" * 60)
    
    strategy = _deploy_strategy_cache['strategy']
    matchup  = _deploy_strategy_cache['matchup']

    # ── 2. Classify this unit ─────────────────────────────────────────
    main_type, support_role = classifier.classify_from_model(unit.unit.model)
    type_label = classifier.get_type_label(main_type, support_role)

    # ── 3. Gather enemy positions already on the board ────────────────
    deployed_enemies = [e for e in enemy_units if e.isDeployed]
    enemy_positions = []
    for e in deployed_enemies:
        pos = e.bodyNP.getPos()
        e_type, e_role = classifier.classify_from_model(e.unit.model)
        enemy_positions.append((pos.x, pos.y, e_type, e_role, e.unit.name))

    # Average enemy X (centre of mass) — fallback to 0 if none deployed
    avg_enemy_x = 0.0
    if enemy_positions:
        avg_enemy_x = sum(p[0] for p in enemy_positions) / len(enemy_positions)

    # Deploy bounds for player 2
    x_min, x_max = -34.0, 34.0
    y_min, y_max = 13.0, 23.0

    # ── 4. Decide position based on strategy + unit type ──────────────
    reason = ""

    if strategy and strategy.name == "Hammer and Anvil":
        if main_type == UnitType.ANVIL or main_type == UnitType.BASIC:
            # Anvil/basic: deploy centrally, facing the enemy centre
            x = _clamp(avg_enemy_x + random.uniform(-4, 4), x_min, x_max)
            y = random.uniform(y_min, y_min + 4)  # front of deploy zone
            reason = f"Centre-front as anvil to pin enemy (facing enemy centre at x≈{avg_enemy_x:.0f})"
        elif main_type == UnitType.HAMMER:
            # Hammer: deploy off to one flank
            flank = _pick_weak_flank(enemy_positions, avg_enemy_x)
            x = _clamp(flank + random.uniform(-3, 3), x_min, x_max)
            y = random.uniform(y_min + 2, y_max - 2)
            reason = f"Flank position (x≈{flank:.0f}) to deliver hammer charge"
        elif support_role == SupportRole.FAST:
            flank = _pick_weak_flank(enemy_positions, avg_enemy_x)
            x = _clamp(flank + random.uniform(-2, 2), x_min, x_max)
            y = random.uniform(y_min + 4, y_max)
            reason = f"Wide flank (x≈{flank:.0f}) — fast unit to sweep around"
        elif support_role == SupportRole.SHOOTING:
            x = _clamp(avg_enemy_x + random.uniform(-6, 6), x_min, x_max)
            y = random.uniform(y_max - 4, y_max)
            reason = "Rear-centre to maximise shooting arcs before contact"
        else:
            x = _clamp(avg_enemy_x + random.uniform(-8, 8), x_min, x_max)
            y = random.uniform(y_min, y_min + 5)
            reason = "Supporting centre line"

    elif strategy and strategy.name == "Refused Flank":
        # Concentrate everything on the weak flank
        flank = _pick_weak_flank(enemy_positions, avg_enemy_x)
        if support_role == SupportRole.SHOOTING or main_type == UnitType.CANNON_FODDER:
            # Delay screen on the opposite side
            x = _clamp(-flank + random.uniform(-4, 4), x_min, x_max)
            y = random.uniform(y_min, y_min + 4)
            reason = f"Delay screen on the strong flank (x≈{-flank:.0f}) to buy time"
        else:
            x = _clamp(flank + random.uniform(-5, 5), x_min, x_max)
            y = random.uniform(y_min, y_min + 6)
            reason = f"Concentrated strike force on weak flank (x≈{flank:.0f})"

    elif strategy and strategy.name == "Gunline":
        if support_role == SupportRole.SHOOTING:
            x = _clamp(avg_enemy_x + random.uniform(-8, 8), x_min, x_max)
            y = random.uniform(y_max - 5, y_max)
            reason = "Shooting line — maximum range before enemy contact"
        elif main_type in (UnitType.ANVIL, UnitType.BASIC):
            x = _clamp(avg_enemy_x + random.uniform(-6, 6), x_min, x_max)
            y = random.uniform(y_min, y_min + 3)
            reason = "Front screen to protect the shooting line"
        else:
            x = _clamp(avg_enemy_x + random.uniform(-10, 10), x_min, x_max)
            y = random.uniform(y_min + 2, y_max - 2)
            reason = "Reserve — counter-charge anything that breaks through"

    elif strategy and strategy.name == "Fast Strike":
        if support_role == SupportRole.FAST or main_type == UnitType.HAMMER:
            flank = _pick_weak_flank(enemy_positions, avg_enemy_x)
            x = _clamp(flank + random.uniform(-4, 4), x_min, x_max)
            y = random.uniform(y_min + 2, y_max - 2)
            reason = f"Fast strike wing (x≈{flank:.0f}) — exploit mobility"
        else:
            x = _clamp(avg_enemy_x + random.uniform(-4, 4), x_min, x_max)
            y = random.uniform(y_min, y_min + 3)
            reason = "Centre anchor while fast units manoeuvre"

    elif strategy and strategy.name == "Horde Rush":
        # Spread wide across the entire front
        spread = random.uniform(x_min + 4, x_max - 4)
        x = spread
        y = random.uniform(y_min, y_min + 4)
        reason = "Wide front — flood the battlefield with bodies"

    elif strategy and strategy.name == "Attrition Grind":
        if main_type in (UnitType.ANVIL, UnitType.SUPERIOR):
            x = _clamp(avg_enemy_x + random.uniform(-6, 6), x_min, x_max)
            y = random.uniform(y_min, y_min + 3)
            reason = "Grind line — engage across the front and hold"
        else:
            x = _clamp(avg_enemy_x + random.uniform(-8, 8), x_min, x_max)
            y = random.uniform(y_min + 3, y_max - 2)
            reason = "Second line support for the grinding front"

    elif strategy and strategy.name == "Strong Center":
        # Hammers in the dead centre, basic/cannon-fodder as a screen
        # in front, superior on the flanks, shooting on the far flanks,
        # fast units wide.
        if main_type == UnitType.HAMMER:
            # Centre of the line, slightly back so screens are in front
            x = _clamp(avg_enemy_x + random.uniform(-4, 4), x_min, x_max)
            y = random.uniform(y_min + 2, y_min + 5)
            reason = "Centre hammer — the fist of the formation"
        elif main_type == UnitType.ANVIL:
            # Beside the hammers in the centre
            x = _clamp(avg_enemy_x + random.uniform(-6, 6), x_min, x_max)
            y = random.uniform(y_min + 1, y_min + 4)
            reason = "Anvil — anchor next to the centre hammers"
        elif main_type == UnitType.BASIC or main_type == UnitType.CANNON_FODDER:
            # Screen in front of the centre
            x = _clamp(avg_enemy_x + random.uniform(-6, 6), x_min, x_max)
            y = random.uniform(y_min, y_min + 2)
            reason = "Screen/redirector — front of centre to absorb charges"
        elif main_type == UnitType.SUPERIOR:
            # Flanks of the formation
            flank = _pick_weak_flank(enemy_positions, avg_enemy_x)
            # Alternate deployed superior units between left and right
            deployed_sup = [u for u in my_units
                           if u.isDeployed
                           and classifier.classify_from_model(u.unit.model)[0]
                               == UnitType.SUPERIOR]
            if len(deployed_sup) % 2 == 0:
                x = _clamp(flank + random.uniform(-3, 3), x_min, x_max)
            else:
                x = _clamp(-flank + random.uniform(-3, 3), x_min, x_max)
            y = random.uniform(y_min + 1, y_min + 4)
            reason = "Superior — flank guard position"
        elif support_role == SupportRole.SHOOTING:
            # Far flanks to target enemy fast units
            flank = _pick_weak_flank(enemy_positions, avg_enemy_x)
            deployed_shoot = [u for u in my_units
                             if u.isDeployed
                             and classifier.classify_from_model(u.unit.model)[1]
                                 == SupportRole.SHOOTING]
            if len(deployed_shoot) % 2 == 0:
                x = _clamp(25 + random.uniform(-3, 3), x_min, x_max)
            else:
                x = _clamp(-25 + random.uniform(-3, 3), x_min, x_max)
            y = random.uniform(y_max - 4, y_max)
            reason = "Shooting — far flank to cover against fast threats"
        elif support_role == SupportRole.FAST:
            # Wide on a flank to hunt war machines
            flank = _pick_weak_flank(enemy_positions, avg_enemy_x)
            x = _clamp(flank + random.uniform(-2, 4), x_min, x_max)
            y = random.uniform(y_min, y_min + 3)
            reason = "Fast unit — wide flank to hunt enemy shooting"
        else:
            x = _clamp(avg_enemy_x + random.uniform(-8, 8), x_min, x_max)
            y = random.uniform(y_min, y_min + 4)
            reason = "Supporting the centre formation"

    elif strategy and strategy.name == "Cavalry Charge":
        # Cavalry deploy on one flank with stronger units on the outside.
        # Fast chaff units go to both flanks to hunt shooters.
        flank = _pick_weak_flank(enemy_positions, avg_enemy_x)

        if support_role == SupportRole.SHOOTING:
            # Shooting support (bolt throwers etc.) — rear, covering field
            x = _clamp(avg_enemy_x + random.uniform(-6, 6), x_min, x_max)
            y = random.uniform(y_max - 4, y_max)
            reason = "Rear shooting line — strip ranks before the charge"
        elif (main_type == UnitType.CANNON_FODDER
              or (support_role == SupportRole.FAST
                  and main_type not in (UnitType.HAMMER, UnitType.SUPERIOR))):
            # Fast redirectors / chaff — split to both flanks to hunt
            # war machines / screening.  Alternate left/right.
            deployed_chaff = [u for u in my_units
                             if u.isDeployed
                             and classifier.classify_from_model(u.unit.model)[0]
                                 == UnitType.CANNON_FODDER]
            if len(deployed_chaff) % 2 == 0:
                x = _clamp(flank + random.uniform(-3, 3), x_min, x_max)
            else:
                x = _clamp(-flank + random.uniform(-3, 3), x_min, x_max)
            y = random.uniform(y_min, y_min + 3)
            reason = f"Fast chaff — flank sweep to hunt enemy shooters"
        elif main_type == UnitType.HAMMER:
            # Hammer cavalry — outer position on the chosen flank
            # (stronger units further out to protect the flank edge)
            x = _clamp(flank + random.uniform(0, 6), x_min, x_max)
            y = random.uniform(y_min + 1, y_min + 5)
            reason = f"Hammer cavalry — outer-flank (x≈{flank:.0f}) for decisive charge"
        elif main_type == UnitType.SUPERIOR:
            # Superior cav — inner flank next to hammers
            x = _clamp(flank + random.uniform(-4, 2), x_min, x_max)
            y = random.uniform(y_min + 1, y_min + 5)
            reason = f"Superior cavalry — inner-flank support for hammer units"
        elif main_type == UnitType.BASIC:
            # Basic unit in the centre as bait / hold-up
            x = _clamp(avg_enemy_x + random.uniform(-4, 4), x_min, x_max)
            y = random.uniform(y_min, y_min + 3)
            reason = "Basic unit — centre bait to hold up enemy hammer"
        else:
            # Fallback: anchor in the centre
            x = _clamp(avg_enemy_x + random.uniform(-5, 5), x_min, x_max)
            y = random.uniform(y_min, y_min + 4)
            reason = "Centre anchor for cavalry army"

    else:
        # Fallback: generic positioning
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        reason = "Default deployment (no specific strategy match)"

    # ── 5. Avoid overlapping already-deployed friendly units ──────────
    deployed_friendly = [u for u in my_units if u.isDeployed]
    for _ in range(10):  # up to 10 nudge attempts
        collision = False
        for f in deployed_friendly:
            fp = f.bodyNP.getPos()
            if abs(fp.x - x) < 5 and abs(fp.y - y) < 4:
                collision = True
                break
        if not collision:
            break
        x += random.uniform(-4, 4)
        x = _clamp(x, x_min, x_max)

    # ── 6. Print reasoning ────────────────────────────────────────────
    print(f"  DEPLOY {unit.unit.name:25s} [{type_label:18s}] -> ({x:+6.1f}, {y:5.1f})  | {reason}")

    return (x, y)


def _pick_weak_flank(enemy_positions, avg_enemy_x):
    """
    Return an x-coordinate on the flank where the enemy has fewer units.
    Positive = right flank, negative = left flank.
    """
    if not enemy_positions:
        return random.choice([-20, 20])

    left_count  = sum(1 for p in enemy_positions if p[0] < avg_enemy_x)
    right_count = sum(1 for p in enemy_positions if p[0] >= avg_enemy_x)

    if left_count < right_count:
        # Left flank is weaker
        return random.uniform(-30, -15)
    elif right_count < left_count:
        return random.uniform(15, 30)
    else:
        # Even — pick the flank furthest from enemy hammers
        enemy_hammers = [p for p in enemy_positions if p[2] == UnitType.HAMMER]
        if enemy_hammers:
            avg_hammer_x = sum(p[0] for p in enemy_hammers) / len(enemy_hammers)
            return 25.0 if avg_hammer_x < 0 else -25.0
        return random.choice([-22, 22])


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ── Original deploy task (now strategy-aware for AI) ──────────────────

def taskMoveUnit(game,unit,task):
    #game.ignore('mouse1')
    if game.roundCounter.current_player == 2 and game.AIplayer2.active:
        pxy = _get_ai_deploy_position(game, unit)

    else:
        pxy = getMouseXY()

    if pxy is None:
        return task.cont
    x, y = pxy
    #print("Mouse position during move phase:", x, y)
    unit.bodyNP.setPos(x,y,0)
    
    # Notify Bullet that the transform has changed
    unit.bodyNP.node().setTransformDirty()
    # Raise the visual models onto any hill/forest surface under them.
    if hasattr(game, 'movement'):
        game.movement.alignModelsToHillNormal(unit)
    outBounds = False
    c = game.checkUnitContactSmall(unit)
    if c:
        outBounds = True
    unit.bodyNP.node().setTransformDirty()
    if not is_shape_inside(base.world, unit.bodyNP.node(), game.boundary_ghost):
        outBounds = True
    if outBounds:
        unit.model.setColor(.6,0.6,0.6,1)
    else:
        unit.model.setColor(unit.color)

    
    if game.roundCounter.current_player == 2 and game.AIplayer2.active:
        if outBounds:
            return task.cont

        endMoveUnit(game,"dummy")
        return task.done
    return task.cont

def endMoveUnit(game,taskToEnd):
    held = game.unitToMove
    inContact = game.checkUnitContactSmall(held)
    held.bodyNP.node().setTransformDirty()
    inZone = is_shape_inside(base.world, held.bodyNP.node(), game.boundary_ghost)

    # Character dropped onto a friendly unit inside the zone: join its ranks
    # instead of refusing (rather than stacking on top of it).
    if inContact and inZone and is_character(held):
        host = game.getSelectedUnit(inContact.getNode1())
        if (host is not None and host is not held and not is_character(host)
                and same_player(game, held, host) and not has_joined_character(host)):
            print(f"{held.unitName} joins {host.unitName}.")
            join_unit(game, held, host)
            taskMgr.remove(taskToEnd)
            _advance_after_deploy(game)
            return

    if not inZone:
        # Dropped outside the deploy zone: cancel this pickup (leave it
        # undeployed) so the player can change their mind and pick another unit.
        print("Unit is out of bounds, cannot deploy here.")
        held.model.setColor(.6,0.6,0.6,1)
        held.isDeployed = False
        taskMgr.remove(taskToEnd)
        game.accept('mouse1', game.setActiveUnit,
                    [game.setActiveUnitTask, game.setActiveUnitTaskName])
        return

    if inContact:
        # Inside the zone but overlapping another unit: refuse the drop and keep
        # holding so the player can reposition (never place on top of a unit).
        print("Unit is in contact with another unit, cannot deploy here.")
        held.model.setColor(.6,0.6,0.6,1)
        return

    held.model.setColor(held.color)
    taskMgr.remove(taskToEnd)
    held.isDeployed=True
    _advance_after_deploy(game)


def _advance_after_deploy(game):
    """Rebind selection and advance the deploy turn/phase after a placement."""
    game.accept('mouse1', game.setActiveUnit,[game.setActiveUnitTask, game.setActiveUnitTaskName])
    depH = DEPLOY_ZONE_DEPTH
    if game.roundCounter.current_player == 2:
        if not allUnitsDeployed(game.player1Units):
            game.roundCounter.request('PlayerOne')
            #game.boundary_np.setPos(0, -7.5-7.5/2, 0)
            game.boundary_np.setPos(0, -depH-depH/2, 0)

    else:
        if not allUnitsDeployed(game.player2Units):
            game.roundCounter.request('PlayerTwo')
            game.boundary_np.setPos(0, depH+depH/2, 0)
            if game.roundCounter.current_player == 2 and game.AIplayer2.active:
                game.AIplayer2.deployUnits()
            
    if allUnitsDeployed(game.units):
        print("All units deployed, moving to next phase.")
        game.fsm.request("StrategyPhase")
    
    

def getMouseXY():
    if base.mouseWatcherNode.hasMouse():
        pMouse = base.mouseWatcherNode.getMouse()
        pFrom = Point3()
        pTo = Point3()
        base.camLens.extrude(pMouse, pFrom, pTo)
        # Transform to global coordinates
        pFrom = render.getRelativePoint(base.cam, pFrom)
        pTo = render.getRelativePoint(base.cam, pTo)
        result = base.world.rayTestClosest(pFrom, pTo, BitMask32.bit(1))
        if result.hasHit():
            hitPos = result.getHitPos()
            xx = hitPos.getX()
            yy = hitPos.getY()
            return (xx, yy)
        else:
            return None
    else:
        return None
    
def allUnitsDeployed(units):
    for unit in units:
        if not unit.isDeployed:
            return False
    return True

def is_fully_contained(test_node, boundary_ghost):
    """Check if test_node is fully within boundary_ghost"""
    overlapping = boundary_ghost.getOverlappingNodes()
    
    if test_node not in overlapping:
        return False
    
    # For full containment, check bounding box
    test_bounds = test_node.getBounds()
    boundary_bounds = boundary_ghost.getBounds()
    
    return boundary_bounds.contains(test_bounds)

def is_shape_inside(world, inner_node, outer_ghost):
    """Check if inner_node is completely inside outer_ghost"""
    # Get all contact points
    result = world.contactTest(inner_node)
    
    # If there are contacts with the outer boundary, it's not fully inside
    for contact in result.getContacts():
        
        if contact.getNode0() == outer_ghost or contact.getNode1() == outer_ghost:
            """ r = world.contactTestPair(contact.getNode0(), contact.getNode1())
            # Check penetration depth - negative means outside
            for c in r.getContacts():
                 print(f"Contact distance: {c.getManifoldPoint().getDistance()}")
                 #print(f"Contact points: {c.getManifoldPoint().getPositionWorldOnA()}, {c.getManifoldPoint().getPositionWorldOnB()}")
                 #print(f"Contact normal: {c.getManifoldPoint().getNormalWorldOnB()}")
                 if c.getManifoldPoint().getDistance() > 0:
                    return False """
            return False
            
    return True