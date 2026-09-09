"""Real base positions survive layout, casualties and restoration."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from direct.showbase.ShowBase import ShowBase
from panda3d.bullet import BulletWorld
from panda3d.core import loadPrcFileData

from models import model
from movement_system import MovementSystem
from scouts import model_base_boxes
from skirmish import coherency_error
from units import unit, unitGraphics


class SkirmishStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loadPrcFileData('', 'window-type offscreen\naudio-library-name null')
        cls.app = ShowBase()
        cls.app.resolvingCombat = False

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()

    def setUp(self):
        self.app.world = BulletWorld()
        profile = model('State Trooper', '')
        profile.special_rules.append(dict(name='Skirmishers', tag='formation', skirmish=True))
        model_path = Path(__file__).resolve().parents[1] / 'models' / 'jade_warrior.bam'
        self.member = unitGraphics(self.app, 'Skirmish test', str(model_path),
                                   unit('Test', profile, 5, 3, 2),
                                   BulletWorld=self.app.world)
        self.member.bodyNP.setPos(3, -5, 0)
        self.member.bodyNP.setH(37)
        self.member.isDeployed = True
        self.app.units = [self.member]
        self.app.player1Units = [self.member]
        self.app.player2Units = []

    def tearDown(self):
        self.app.taskMgr.remove('updateTextNode')
        self.app.world.removeRigidBody(self.member.bodyNP.node())
        self.member.bodyNP.removeNode()

    def test_initial_layout_is_coherent(self):
        self.assertIsNone(coherency_error(model_base_boxes(self.member)))

    def test_leaving_combat_retains_compact_bases_until_phase_end(self):
        for state in ('IsFleeing', 'IsPursuing', 'Moved'):
            with self.subTest(state=state):
                self.member.request('InCombat')
                before = model_base_boxes(self.member)
                identities = self.member.savedSkirmishLayout()
                self.member.request(state)
                self.assertTrue(self.member.skirmishCombat)
                self.assertEqual(model_base_boxes(self.member), before)
                self.assertEqual(self.member.savedSkirmishLayout(), identities)

    def test_rebuilding_footprint_never_shifts_bases(self):
        before = model_base_boxes(self.member)
        self.member.rebuildFootprint()
        self.member.layOutRanks()
        self.assertEqual(before, model_base_boxes(self.member))

    def test_separation_keeps_rank_order_ids_and_world_facing(self):
        self.member.request('InCombat')
        before = model_base_boxes(self.member)
        identities = [record['id'] for record in self.member.skirmishLayout]
        self.member.request('Moved')
        self.member.spreadToSkirmish()
        self.assertFalse(self.member.skirmishCombat)
        self.assertEqual(self.member.state, 'Moved')
        self.assertEqual(identities, [record['id'] for record in self.member.skirmishLayout])
        after = model_base_boxes(self.member)
        self.assertIsNone(coherency_error(after))
        for original, separated in zip(before, after):
            self.assertAlmostEqual(original[0], separated[0], delta=0.001)
            self.assertAlmostEqual(original[1], separated[1], delta=0.001)
            self.assertEqual(original[4], separated[4])
        self.member.spreadToSkirmish()
        self.assertEqual(after, model_base_boxes(self.member))

    def test_engaged_and_fleeing_units_do_not_separate(self):
        self.member.request('InCombat')
        before = model_base_boxes(self.member)
        self.member.spreadToSkirmish()
        self.assertTrue(self.member.skirmishCombat)
        self.assertEqual(before, model_base_boxes(self.member))
        self.member.request('IsFleeing')
        self.member.spreadToSkirmish()
        self.assertTrue(self.member.skirmishCombat)
        self.assertEqual(before, model_base_boxes(self.member))

    def test_casualties_keep_identity_and_world_positions(self):
        before = dict(zip((record['id'] for record in self.member.skirmishLayout),
                          model_base_boxes(self.member)))
        MovementSystem.removeModelsFromUnit(SimpleNamespace(game=self.app), self.member, 2)
        after = dict(zip((record['id'] for record in self.member.skirmishLayout),
                         model_base_boxes(self.member)))
        self.assertEqual(len(after), 3)
        self.assertTrue(all(before[identity] == box for identity, box in after.items()))
        self.assertIsNone(coherency_error(list(after.values())))

    def test_restore_recreates_casualties_at_saved_positions(self):
        saved = self.member.savedSkirmishLayout()
        before = model_base_boxes(self.member)
        MovementSystem.removeModelsFromUnit(SimpleNamespace(game=self.app), self.member, 3)
        self.member.unit.nmodels = 5
        self.member.restoreSkirmishLayout(saved)
        self.assertEqual(before, model_base_boxes(self.member))
        self.assertEqual(saved, self.member.savedSkirmishLayout())

    def test_predecremented_count_still_removes_from_all_live_bases(self):
        for index, record in enumerate(self.member.skirmishLayout):
            record['x'] = index * (self.member.modelWidth + 0.6)
            record['y'] = 0
        self.member.rebuildFootprint()
        self.member.unit.nmodels -= 2
        MovementSystem.removeModelsFromUnit(SimpleNamespace(game=self.app), self.member, 2)
        self.assertEqual(self.member.unit.nmodels, 3)
        self.assertEqual(len(self.member.skirmishLayout), 3)
        self.assertIsNone(coherency_error(model_base_boxes(self.member)))

    def test_legacy_restore_rebuilds_current_model_count(self):
        self.member.unit.nmodels = 3
        self.member.restoreSkirmishLayout()
        self.assertEqual(len(model_base_boxes(self.member)), 3)
        self.assertIsNone(coherency_error(model_base_boxes(self.member)))


if __name__ == '__main__':
    unittest.main()