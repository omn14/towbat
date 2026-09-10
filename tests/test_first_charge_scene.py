"""First Charge state in the loaded armies, saves and real phase cleanup (p. 169)."""

import json
from pathlib import Path
from unittest.mock import patch

from first_charge import begin_charge_attempt, count_as_charge, finish_charge_attempt, has_first_charge
from persistence import load_game_state, save_game_state
from psychology import combat_rank_bonus
from tests.test_faction_rules_scene import members, scene as scene


def test_actual_roster_sources_and_disruption_expiry(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    armies = members(app)
    for name in ('Silver Helm', 'Dragon Prince', 'Chaos Knight'):
        assert has_first_charge(armies[name]) and armies[name].chargeAttempts == 0
    assert not has_first_charge(armies['Lothern Skycutter'])
    charger, target = armies['Silver Helm'], armies['Chaos Warrior']
    bonus = combat_rank_bonus(target)
    assert bonus > 0
    begin_charge_attempt(charger)
    finish_charge_attempt(charger, target)
    assert combat_rank_bonus(target) == 0
    path = save_game_state(app, str(tmp_path / 'first-charge.json'))
    load_game_state(app, baseline)
    load_game_state(app, path)
    assert charger.chargeAttempts == 1 and combat_rank_bonus(target) == 0
    app.fsm.request('CombatPhase')
    with patch.object(app.fsm, '_spell_origin', 'CombatPhase'):
        app.fsm.exitCombatPhase()
    assert combat_rank_bonus(target) == 0
    app.fsm.exitCombatPhase()
    assert combat_rank_bonus(target) == bonus and charger.chargeAttempts == 1
    begin_charge_attempt(charger)
    finish_charge_attempt(charger, target)
    assert combat_rank_bonus(target) == bonus


def test_failed_attempt_stays_spent_after_reload(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    armies = members(app)
    charger, target = armies['Chaos Knight'], armies['Silver Helm']
    begin_charge_attempt(charger)
    finish_charge_attempt(charger)
    path = save_game_state(app, str(tmp_path / 'failed-first-charge.json'))
    load_game_state(app, baseline)
    load_game_state(app, path)
    begin_charge_attempt(charger)
    finish_charge_attempt(charger, target)
    assert charger.chargeAttempts == 2 and not target.firstChargeDisruptedBy


def test_pending_attempt_resumes_without_double_consumption(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    armies = members(app)
    charger, target = armies['Dragon Prince'], armies['Chaos Warrior']
    begin_charge_attempt(charger)
    path = save_game_state(app, str(tmp_path / 'pending-first-charge.json'))
    load_game_state(app, baseline)
    load_game_state(app, path)
    assert charger.firstChargePending and charger.chargeAttemptPending
    begin_charge_attempt(charger)
    finish_charge_attempt(charger, target)
    finish_charge_attempt(charger, target)
    assert charger.chargeAttempts == 1
    assert target.firstChargeDisruptedBy == [charger.unit.name]


def test_legacy_save_does_not_invent_unused_first_charge(scene, tmp_path, capsys):
    app, baseline = scene
    records = json.loads(Path(baseline).read_text())
    for record in records['units']:
        for key in ('chargeAttempts', 'chargeAttemptPending', 'firstChargePending',
                'firstChargeDisruptedBy', 'firstChargeDisruptedNextTurnBy'):
            record.pop(key, None)
    path = tmp_path / 'legacy-first-charge.json'
    path.write_text(json.dumps(records))
    load_game_state(app, str(path))
    armies = members(app)
    charger, target = armies['Chaos Knight'], armies['Silver Helm']
    begin_charge_attempt(charger)
    finish_charge_attempt(charger, target)
    assert charger.chargeAttempts == 2 and not target.firstChargeDisruptedBy
    assert 'legacy save has no charge history' in capsys.readouterr().out


def test_deferred_pursuit_disruption_survives_reload_and_one_turn(scene, tmp_path):
    app, baseline = scene
    load_game_state(app, baseline)
    armies = members(app)
    charger, target = armies['Dragon Prince'], armies['Chaos Warrior']
    bonus = combat_rank_bonus(target)
    count_as_charge(charger, target, next_turn=True)
    path = save_game_state(app, str(tmp_path / 'pursuit-first-charge.json'))
    load_game_state(app, baseline)
    load_game_state(app, path)
    assert combat_rank_bonus(target) == bonus
    app.fsm.exitCombatPhase()
    assert combat_rank_bonus(target) == 0 and charger.chargeAttempts == 1
    app.fsm.exitCombatPhase()
    assert combat_rank_bonus(target) == bonus