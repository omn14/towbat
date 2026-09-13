"""XML and JSON catalogue inputs must produce the same runtime records."""

import json
import xml.etree.ElementTree as ET

import pytest

from battlescribe import (Catalogue, NS, _catalogue_root, _rule_descriptions,
                          _spell_lores, parse_catalogue_full, parse_weapons)


@pytest.fixture
def catalogue_pair(tmp_path):
    xml = '''<catalogue xmlns="http://www.battlescribe.net/schema/catalogueSchema"
        name="Test Army" id="test-army" library="false" revision="2">
      <sharedProfiles>
        <profile id="model" name="Test Soldier" typeName="Model">
          <characteristics><characteristic name="WS">4</characteristic>
            <characteristic name="I">3</characteristic></characteristics>
        </profile>
        <profile id="weapon" name="Test Bow" typeName="Weapon">
          <characteristics><characteristic name="R">24</characteristic>
            <characteristic name="S">3</characteristic>
            <characteristic name="AP">-1</characteristic></characteristics>
        </profile>
        <profile id="rule" name="Test Rule" typeName="Special Rule">
          <characteristics><characteristic name="Description">Test text.</characteristic></characteristics>
        </profile>
      </sharedProfiles>
      <sharedInfoGroups><infoGroup id="lore" name="Test Lore"><profiles>
        <profile id="spell" name="Test Spell" typeName="Spell"><characteristics>
          <characteristic name="Casting Value">8+</characteristic>
          <characteristic name="Type">Magic Missile</characteristic>
          <characteristic name="Range">24</characteristic>
          <characteristic name="Number">1</characteristic>
        </characteristics></profile>
      </profiles></infoGroup></sharedInfoGroups>
      <sharedSelectionEntries><selectionEntry id="unit" name="Test Soldiers" type="unit">
        <selectionEntries><selectionEntry id="soldier" name="Test Soldier" type="model">
          <infoLinks><infoLink name="Test Soldier" type="profile" targetId="model"/>
            <infoLink name="Base 25x25" type="profile" targetId="base"/></infoLinks>
          <costs><cost name="pts" typeId="points" value="13"/></costs>
        </selectionEntry></selectionEntries>
      </selectionEntry></sharedSelectionEntries>
      <entryLinks><entryLink targetId="unit"><categoryLinks>
        <categoryLink targetId="f0e3-2e32-8866-ea32" primary="true"/>
      </categoryLinks></entryLink></entryLinks>
    </catalogue>'''
    record = {
        'name': 'Test Army', 'id': 'test-army', 'library': False, 'revision': 2,
        'sharedProfiles': [
            {'id': 'model', 'name': 'Test Soldier', 'typeName': 'Model',
             'characteristics': [{'name': 'WS', '$text': 4}, {'name': 'I', '$text': 3}]},
            {'id': 'weapon', 'name': 'Test Bow', 'typeName': 'Weapon',
             'characteristics': [{'name': 'R', '$text': 24}, {'name': 'S', '$text': 3},
                                 {'name': 'AP', '$text': -1}]},
            {'id': 'rule', 'name': 'Test Rule', 'typeName': 'Special Rule', 'alias': ['Alias'],
             'characteristics': [{'name': 'Description', '$text': 'Test text.'}]},
        ],
        'sharedInfoGroups': [{'id': 'lore', 'name': 'Test Lore', 'profiles': [
            {'id': 'spell', 'name': 'Test Spell', 'typeName': 'Spell', 'characteristics': [
                {'name': 'Casting Value', '$text': '8+'}, {'name': 'Type', '$text': 'Magic Missile'},
                {'name': 'Range', '$text': 24}, {'name': 'Number', '$text': 1}]}]}],
        'sharedSelectionEntries': [{'id': 'unit', 'name': 'Test Soldiers', 'type': 'unit',
            'selectionEntries': [{'id': 'soldier', 'name': 'Test Soldier', 'type': 'model',
                'infoLinks': [{'name': 'Test Soldier', 'type': 'profile', 'targetId': 'model'},
                              {'name': 'Base 25x25', 'type': 'profile', 'targetId': 'base'}],
                'costs': [{'name': 'pts', 'typeId': 'points', 'value': 13}]}]}],
        'entryLinks': [{'targetId': 'unit', 'categoryLinks': [
            {'targetId': 'f0e3-2e32-8866-ea32', 'primary': True}]}],
    }
    xml_path, json_path = tmp_path / 'army.cat', tmp_path / 'army.json'
    xml_path.write_text(xml, encoding='utf-8')
    json_path.write_text(json.dumps({'catalogue': record}), encoding='utf-8')
    return xml_path, json_path


def test_xml_json_extraction_parity(catalogue_pair):
    xml_path, json_path = catalogue_pair
    assert parse_catalogue_full(xml_path) == parse_catalogue_full(json_path)
    assert parse_weapons(xml_path) == parse_weapons(json_path)
    assert _rule_descriptions(xml_path) == _rule_descriptions(json_path) == {'Test Rule': 'Test text.'}
    assert _spell_lores(xml_path) == _spell_lores(json_path)
    faction, records, _, weapons = parse_catalogue_full(json_path)
    assert faction == 'Test Army' and len(records) == 1
    assert records[0]['WS'] == '4' and records[0]['Points'] == 13
    assert records[0]['Category'] == 'Core' and records[0]['base_width_mm'] == 25
    assert weapons[0]['ranged_range'] == 24 and weapons[0]['ranged_AP'] == 1
    assert _spell_lores(json_path)['Test Lore'][0]['casting_value'] == 8
    root = _catalogue_root(json_path)
    assert root.get('library') == 'false' and root.get('revision') == '2'
    assert root.find(f'.//{NS}alias').text == 'Alias'


def test_system_namespace_and_shared_weapons(catalogue_pair):
    xml_path, json_path = catalogue_pair
    xml_path.write_text(xml_path.read_text().replace('catalogue', 'gameSystem'))
    record = json.loads(json_path.read_text())['catalogue']
    json_path.write_text(json.dumps({'gameSystem': record}))
    assert parse_weapons(xml_path) == parse_weapons(json_path)
    assert _rule_descriptions(xml_path) == _rule_descriptions(json_path)
    catalogue = Catalogue(str(json_path.parent))
    assert catalogue.weapon('Test Bow')['ranged_strength'] == 3
    assert catalogue.rule_description('Test Rule') == 'Test text.'
    assert not catalogue.by_slug


def test_duplicate_catalogue_ids_prefer_json_and_xml_alone_still_loads(catalogue_pair):
    xml_path, json_path = catalogue_pair
    xml_path.write_text(xml_path.read_text().replace('>4<', '>2<'))
    catalogue = Catalogue(str(json_path.parent))
    assert catalogue.characteristics('Test Soldier')['WS'] == '4'
    assert list(catalogue.iter_models())[0][2]['Category'] == 'Core'
    json_path.unlink()
    assert Catalogue(str(xml_path.parent)).characteristics('Test Soldier')['WS'] == '2'


@pytest.mark.parametrize('document', ['{', '{}', 'null', '{"roster": {}}',
    '{"catalogue": {}, "gameSystem": {}}', '{"catalogue": {"profiles": [3]}}'])
def test_malformed_json_is_reported_and_valid_xml_still_loads(catalogue_pair, document, capsys):
    xml_path, json_path = catalogue_pair
    json_path.write_text(document)
    with pytest.raises(ET.ParseError):
        _catalogue_root(json_path)
    assert Catalogue(str(xml_path.parent)).characteristics('Test Soldier')['WS'] == '4'
    assert 'failed to parse army.json' in capsys.readouterr().out


def test_offline_converter_accepts_json(catalogue_pair, tmp_path):
    from catalogue_converter import convert_catalogue
    _, json_path = catalogue_pair
    output = tmp_path / 'export'
    convert_catalogue(json_path, str(output))
    record = json.loads((output / 'test_army' / 'test_soldier_characteristics.json').read_text())
    assert record['WS'] == '4' and record['Points'] == 13 and record['Category'] == 'Core'