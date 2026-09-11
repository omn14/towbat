"""Optional campaign resources load on first entry, not battle startup."""

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from panda3d.core import Point3, Vec3

from game import MyApp
from game_fsm import GamePhaseFSM


def test_campaign_entry_loads_once_and_exit_restores_battle():
    country, cloud = Mock(), Mock()
    country.getChildren.return_value = [Mock(), Mock()]
    campaign = SimpleNamespace(CampaignMap=Mock(), CountryFSM=Mock())
    game = Mock(campaign_map=None, units=[Mock(), Mock()])
    game.loader.loadModel.side_effect = [country, cloud]
    game.camera.getPos.return_value = Point3(1, -20, 30)
    game.camera.getHpr.return_value = Vec3(0, -60, 0)
    game.setup_campaign_map = lambda: MyApp.setup_campaign_map(game)
    phase = SimpleNamespace(game=game, ignore=Mock())

    with patch.dict('sys.modules', {'campaignMap': campaign}), patch('game.Shader') as shader:
        for visit in range(2):
            GamePhaseFSM.enterCampaignPhase(phase)
            assert game.campaign_map is campaign.CampaignMap.return_value
            assert game.country_model is country and game.cloud_plane is cloud
            assert game.cloud_nodes == [cloud]
            assert game.campaign_map.show.call_count == visit + 1
            GamePhaseFSM.exitCampaignPhase(phase)
            game.camera.setPos.assert_called_with(Point3(1, -20, 30))
            game.camera.setHpr.assert_called_with(Vec3(0, -60, 0))
            for member in game.units:
                assert member.bodyNP.hide.call_count == visit + 1
                assert member.bodyNP.show.call_count == visit + 1
        shader.load.assert_called_once()

    campaign.CampaignMap.assert_called_once_with(game)
    campaign.CountryFSM.assert_called_once_with(country, game)
    assert game.loader.loadModel.call_args_list == [call('models/blender/maps1.bam'), call('models/box')]
    game.campaign_map.load_heightmap.assert_called_once_with(
        'assets/textures/wals_dem_resized.png', height_scale=25)
    game.campaign_map.set_texture.assert_called_once_with('assets/textures/wals_tex_resized.png')
    assert game.campaign_map.contryCollision.call_count == 2
    assert game.campaign_map.hide.call_count == 3
    assert country.hide.call_count == cloud.hide.call_count == 3
    assert game.taskMgr.add.call_args_list == [
        call(game.update_campaign_terrain, 'update_campaign_terrain'),
        call(game.update_cloud_time, 'update_cloud_time')] * 2
    assert game.taskMgr.remove.call_args_list == [
        call('update_campaign_terrain'), call('update_cloud_time')] * 2