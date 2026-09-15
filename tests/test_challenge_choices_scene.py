"""Participant and refusal nominations belong to different players (p. 210)."""

from unittest.mock import AsyncMock, patch

import pytest

from challenges import Challenge, duellists
from characters import join_unit
from command_groups import champions
from persistence import load_game_state, save_game_state
from tests.test_faction_rules_scene import members, scene as scene
from tests.test_shieldwall_scene import combat_tasks


def participants(app, baseline):
    load_game_state(app, baseline)
    app.hud._journal.entries.clear()
    army = members(app)
    host, character, enemy = army['Chaos Knight'], army['Aspiring Champion'], army['Mage']
    assert join_unit(app, character, host)
    champion, = champions(host)
    assert duellists(host) == [character, champion]
    return host, character, champion, enemy


def test_issuer_can_choose_unit_champion_instead_of_joined_character(scene):
    app, baseline = scene
    host, character, champion, enemy = participants(app, baseline)
    async def choose(options, *args, owner, **kwargs):
        assert owner is host
        return 'Issue a challenge' if 'Issue a challenge' in options else champion.unitName
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', side_effect=choose), \
            patch.object(app.combat, 'armDuellists', AsyncMock()):
        async def exchange():
            challenge = await app.combat.challengeExchange(host, enemy)
            assert challenge.challenger is champion
            assert challenge.accepter is enemy
        run(exchange())
    assert not character.retiredFromCombat
    history = '\n'.join(entry.text for entry in app.hud._journal.visible())
    assert 'Challenge issued:' in history and 'Challenge accepted:' in history
    assert champion.unit.name in history and enemy.unit.name in history


def test_accepting_player_can_choose_unit_champion(scene):
    app, baseline = scene
    host, character, champion, enemy = participants(app, baseline)
    challenge = Challenge(enemy, enemy)
    async def choose(options, *args, owner, **kwargs):
        assert owner is host
        return 'Accept' if 'Accept' in options else champion.unitName
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', side_effect=choose):
        run(app.combat.answerChallenge(challenge, host))
    assert challenge.accepter is champion and challenge.accepter_host is host
    assert not character.retiredFromCombat
    history = '\n'.join(entry.text for entry in app.hud._journal.visible())
    assert 'Challenge accepted:' in history and champion.unit.name in history


@pytest.mark.parametrize('nomination', ['character', 'champion', 'none'])
def test_challenger_owns_refusal_nomination_and_retirement_survives_reload(scene, tmp_path, nomination):
    app, baseline = scene
    host, character, champion, enemy = participants(app, baseline)
    challenge = Challenge(enemy, enemy)
    nominated = {'character': character, 'champion': champion, 'none': None}[nomination]
    owners = []
    async def choose(options, *args, owner, **kwargs):
        owners.append(owner)
        if 'Refuse' in options:
            assert owner is host
            return 'Refuse'
        assert owner is enemy
        return nominated.unitName if nominated else 'No retirement'
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', side_effect=choose):
        run(app.combat.answerChallenge(challenge, host))
    assert owners == [host, enemy]
    assert challenge.refused and not challenge.answered
    assert challenge.retired is nominated
    assert character.retiredFromCombat is (nomination == 'character')
    assert champion.retiredFromCombat is (nomination == 'champion')
    history = '\n'.join(entry.text for entry in app.hud._journal.visible())
    assert 'Challenge refused:' in history
    if nominated is None:
        assert 'no model nominated to retire' in history
    else:
        assert f'{nominated.unit.name} retires' in history
        assert 'no attacks' in history and 'no Leadership' in history
    path = save_game_state(app, str(tmp_path / 'retirement.json'))
    assert path is not None
    load_game_state(app, path)
    host, character = members(app)['Chaos Knight'], members(app)['Aspiring Champion']
    champion, = champions(host, include_retired=True)
    assert character.retiredFromCombat is (nomination == 'character')
    assert champion.retiredFromCombat is (nomination == 'champion')


def test_multiple_combat_can_nominate_from_a_different_host(scene):
    app, baseline = scene
    host, character, champion, enemy = participants(app, baseline)
    selected = members(app)['Chaos Warrior']
    async def choose(options, *args, owner, **kwargs):
        assert owner is selected
        return 'Issue a challenge' if 'Issue a challenge' in options else champion.unitName
    with combat_tasks(app) as run, patch.object(app.roundCounter, 'current_player', 2), \
            patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', side_effect=choose), \
            patch.object(app.combat, 'armDuellists', AsyncMock()):
        async def exchange():
            challenge = await app.combat.challengeExchange(selected, enemy, hosts=[selected, enemy, host])
            assert challenge.challenger is champion and challenge.host is host
            assert challenge.accepter is enemy
            assert await app.combat.challengeExchange(selected, enemy, hosts=[selected, enemy, host]) is challenge
            assert challenge.rounds == 1
        run(exchange())
    history = '\n'.join(entry.text for entry in app.hud._journal.visible())
    assert 'Challenge continues:' in history and 'duel round 2' in history


def test_unanswered_challenge_is_visible_in_summary(scene):
    app, baseline = scene
    host, character, champion, enemy = participants(app, baseline)
    challenge = Challenge(character, host)
    with combat_tasks(app) as run, patch('challenges.duellists', return_value=[]):
        run(app.combat.answerChallenge(challenge, enemy))
    history = '\n'.join(entry.text for entry in app.hud._journal.visible())
    assert 'Challenge unanswered:' in history and 'no eligible participant' in history


def test_duel_rolls_and_results_reach_game_log_and_history(scene):
    from contextlib import ExitStack
    from battleFunctions import take_last_combat_report
    from rules_log import log_scope
    app, baseline = scene
    host, character, champion, enemy = participants(app, baseline)
    challenge = Challenge(character, host, enemy, enemy)
    app.challenges = [challenge]
    for fighter, initiative in ((character, 5), (enemy, 3)):
        fighter.unit.model.special_rules = []
        fighter.unit.model.characteristics.update(WS=3, S=3, T=3, A=1, I=initiative, W=3)
        fighter.unit.model._base_characteristics = dict(fighter.unit.model.characteristics)
        fighter.unit.model.equipedWeapon = {'name': 'Hand Weapon', 'tag': 'combat'}
        fighter.unit.model.AP = 0
        fighter.woundsOnModel = 0
    app.hud._log_mode = 'Summary'
    app.hud._log_scroll = 0
    app.hud._log_frozen = None
    with ExitStack() as stack:
        for fighter in (character, enemy):
            stack.enter_context(patch.object(fighter.unit.model, 'melee_armour_save', return_value=5))
        dice = stack.enter_context(patch('battleFunctions.random.randint', side_effect=[4, 4, 2, 4, 4, 5]))
        stack.enter_context(log_scope(phase='CombatPhase', combat='History duel'))
        result = app.combat.resolveChallenge(challenge)
    assert result == ((1, 0, 0, 0) if host in app.player1Units else (0, 1, 0, 0))
    assert dice.call_count == 6
    assert take_last_combat_report() is None
    entries = app.hud._journal.visible()
    attacks = [entry for entry in entries if entry.text.startswith('Challenge I')]
    assert len(attacks) == 2
    assert '1 attacks -> 1 hits -> 1 wounds -> 0 saved -> 1 unsaved' in attacks[0].text
    assert 'Armour rolls (after modifiers): [2]' in attacks[0].details
    assert 'Armour rolls (after modifiers): [5]' in attacks[1].details
    assert all(entry.context['combat'] == 'History duel' for entry in attacks)
    visible = app.hud.log_text(app.hud._log_displayed)
    assert 'Challenge result:' in visible and 'Both survive; challenge continues.' in visible
    assert '3 -> 2 Wounds remaining' in visible
    app.hud.open_history()
    try:
        history = app.hud._history
        assert 'Challenge result:' in history.text.getText()
        assert 'Armour rolls' not in history.text.getText()
        history.set_details(True)
        app.graphicsEngine.renderFrame()
        assert 'Armour rolls (after modifiers): [2]' in history.text.getText()
        assert 'Armour rolls (after modifiers): [5]' in history.text.getText()
    finally:
        app.hud.close_history()


@pytest.mark.parametrize('answer', ['Accept', 'Refuse'])
def test_multiple_combat_answer_uses_nominated_models_own_host(scene, answer):
    app, baseline = scene
    host, character, champion, enemy = participants(app, baseline)
    selected = members(app)['Chaos Warrior']
    challenge = Challenge(enemy, enemy)
    async def choose(options, *args, owner, **kwargs):
        if 'Accept' in options:
            assert owner is selected
            return answer
        assert owner is (enemy if answer == 'Refuse' else selected)
        return character.unitName
    with combat_tasks(app) as run, patch.object(app, 'aiControls', return_value=False), \
            patch.object(app, 'makeChoiceNew', side_effect=choose):
        run(app.combat.answerChallenge(challenge, selected, hosts=[selected, host]))
    if answer == 'Accept':
        assert challenge.accepter is character and challenge.accepter_host is host
    else:
        assert challenge.retired is character and character.retiredFromCombat
        assert not champion.retiredFromCombat