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
    path = save_game_state(app, str(tmp_path / 'retirement.json'))
    assert path is not None
    load_game_state(app, path)
    host, character = members(app)['Chaos Knight'], members(app)['Aspiring Champion']
    champion, = champions(host, include_retired=True)
    assert character.retiredFromCombat is (nomination == 'character')
    assert champion.retiredFromCombat is (nomination == 'champion')