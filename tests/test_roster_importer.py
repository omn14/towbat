"""Selected roster data must survive import without granting unchosen effects."""

import json

import pytest

from roster_importer import import_roster


@pytest.fixture
def roster_path(tmp_path):
    def write(units):
        path = tmp_path / "selected_roster.json"
        path.write_text(json.dumps({"roster": {
            "id": "test-roster", "name": "Import fixture",
            "costLimits": [{"name": "pts", "value": 500}],
            "forces": [{"id": "test-force", "catalogueId": "test-catalogue",
                        "catalogueName": "High Elf Realms", "selections": units}],
        }}), encoding="utf-8")
        return str(path)
    return write


def spell_profile(name, profile_id):
    return {"id": profile_id, "name": name, "typeName": "Spell",
            "characteristics": [{"name": "Type", "$text": "Enchantment"},
                                {"name": "Casting Value", "$text": "8+"},
                                {"name": "Range", "$text": "Self"}]}


def test_lore_profiles_are_options_not_known_spells(roster_path):
    unit = {"id": "mage-unit", "type": "unit", "name": "Mage", "selections": [
        {"id": "mage", "type": "model", "name": "Mage", "number": 1,
         "selections": [
             {"id": "level", "name": "Wizard Level 2", "type": "upgrade"},
             {"id": "high-magic", "name": "High Magic", "type": "upgrade",
              "profiles": [spell_profile("Shield of Saphery", "shield"),
                           spell_profile("Fury of Khaine", "fury")]},
         ]},
    ]}
    imported = import_roster(roster_path([unit]))["units"][0]
    assert imported["spells"] == []
    assert {spell["name"] for spell in imported["spell_pool"]} == {
        "Shield of Saphery", "Fury of Khaine"}
    assert imported["wizard_level"] == 2
    assert imported["spell_generation_pending"] is True


def test_command_promotion_owns_banner_without_adding_models(roster_path):
    association = {"type": "outgoing", "to": "warriors", "name": "Standard Bearer"}
    banner = {"id": "banner", "type": "upgrade", "name": "The Banner Of The Bold",
              "entryId": "banner-entry", "costs": [{"name": "pts", "value": 10}],
              "profiles": [{"id": "banner-definition", "typeName": "Magic Standards",
                            "name": "The Banner Of The Bold"}]}
    unit = {"id": "unit", "type": "unit", "name": "Chaos Warriors", "selections": [
        {"id": "warriors", "type": "model", "name": "Chaos Warrior", "number": 10,
         "costs": [{"name": "pts", "value": 150}],
         "incomingAssociations": [{"type": "incoming", "from": "standard", "amount": 1}]},
        {"id": "standard", "type": "upgrade", "name": "Standard Bearer",
         "group": "Command", "associations": [association],
         "costs": [{"name": "pts", "value": 12}],
         "profiles": [{"id": "standard-profile", "typeName": "Command",
                       "name": "Standard Bearer"}], "selections": [banner]},
    ]}
    imported = import_roster(roster_path([unit]))["units"][0]
    records = {record["id"]: record for record in imported["roster_selections"]}
    assert imported["nmodels"] == 10
    assert imported["points_cost"] == 172
    assert imported["command"][0]["role"] == "standard_bearer"
    assert imported["command"][0]["model_refs"] == [records["warriors"]["ref"]]
    assert records["standard"]["associations"] == [association]
    assert records["warriors"]["incomingAssociations"][0]["amount"] == 1
    item = imported["magic_items"][0]
    assert item["owner_ref"] == records["standard"]["ref"]
    assert item["definition_id"] == "banner-definition"
    assert item["category"] == "Magic Standards"
    assert item["effect_status"] == "unsupported"
    assert imported["special_rules"] == []


def test_equipment_preserves_rider_mount_and_upgrade_model_owners(roster_path):
    def weapon(identity, name):
        return {"id": identity, "type": "upgrade", "name": name,
                "profiles": [{"id": identity + "-profile", "typeName": "Weapon",
                              "name": name, "characteristics": []}]}

    unit = {"id": "unit", "type": "unit", "name": "Rider", "selections": [
        {"id": "rider", "type": "model", "name": "Rider", "number": 1,
         "selections": [weapon("rider-weapon", "Hand Weapon"),
                        {"id": "steed", "type": "mount", "name": "Steed",
                         "selections": [weapon("mount-weapon", "Hand Weapon")]}]},
        {"id": "roc", "type": "upgrade", "name": "Swiftfeather Roc",
         "profiles": [{"typeName": "Model", "name": "Swiftfeather Roc"}],
         "selections": [weapon("claws", "Wicked Claws")]},
    ]}
    imported = import_roster(roster_path([unit]))["units"][0]
    records = {record["id"]: record for record in imported["roster_selections"]}
    assert [entry["owner_ref"] for entry in imported["equipment"]] == [
        records["rider"]["ref"], records["steed"]["ref"], records["roc"]["ref"]]
    assert len(imported["equipment"]) == 3
    assert imported["roster_source"]["catalogueId"] == "test-catalogue"


def test_unselected_and_zero_count_upgrades_are_not_imported(roster_path):
    item = {"id": "unused", "name": "Unselected Relic", "type": "upgrade",
            "costs": [{"name": "pts", "value": 100}],
            "profiles": [{"typeName": "Talismans", "name": "Unselected Relic"}]}
    unit = {"id": "unit", "name": "Mage", "type": "unit",
            "selectionEntries": [item], "selectionEntryGroups": [{"selections": [item]}],
            "selections": [dict(item, number=0)]}
    imported = import_roster(roster_path([unit]))["units"][0]
    assert imported["magic_items"] == []
    assert imported["points_cost"] == 0
    assert len(imported["roster_selections"]) == 1
    assert "selectionEntries" not in imported["roster_selections"][0]


@pytest.fixture
def readiness_units():
    def item(identity, name, category, points):
        return {"id": identity, "type": "upgrade", "name": name, "number": 1,
                "entryId": identity + "-entry", "costs": [{"name": "pts", "value": points}],
                "profiles": [{"id": identity + "-definition", "name": name,
                              "typeName": category}]}

    def unit(identity, name, count, points):
        return {"id": identity, "type": "unit", "name": name, "selections": [
            {"id": identity + "-models", "type": "model", "name": name,
             "number": count, "costs": [{"name": "pts", "value": points}]}]}

    def command(identity, role, target, points):
        return {"id": identity, "name": role, "type": "upgrade", "number": 1,
                "group": "Command", "costs": [{"name": "pts", "value": points}],
                "profiles": [{"name": role, "typeName": "Command"}],
                "associations": [{"type": "outgoing", "to": target, "name": role}]}

    elves = [unit("mage", "Mage", 1, 110), unit("archers", "Elven Archer", 6, 54),
             unit("helms", "Silver Helm", 5, 120), unit("princes", "Dragon Prince", 3, 111),
             unit("skycutter", "Lothern Skycutter", 1, 90)]
    mage = elves[0]["selections"][0]
    mage["profiles"] = [spell_profile(name, str(index)) for index, name in enumerate(
        ["Hand of Khaine", "Courage of Aenarion", "Vaul's Unmaking"])]
    mage["selections"] = [
        {"id": "level", "name": "Wizard Level 2", "type": "upgrade"},
        item("wand", "Silvery Wand", "Arcane Items", 15),
        {"id": "lore", "name": "High Magic", "type": "upgrade", "profiles": [
            spell_profile(name, str(index)) for index, name in enumerate([
                "Walk Between Worlds", "Fiery Convocation", "Tempest", "Corporeal Unmaking",
                "Fury of Khaine", "Shield of Saphery", "Drain Magic"]) ]},
    ]
    chaos = [unit("general", "Aspiring Champion", 1, 74),
             unit("knights", "Chaos Knight", 4, 104), unit("hounds", "Chaos Warhound", 5, 30),
             unit("warriors", "Chaos Warrior", 10, 150),
             unit("horsemen", "Marauder Horsemen", 5, 60)]
    chaos[0]["selections"][0]["selections"] = [
        item("helm", "Helm Of Courage", "Magic Armour", 25)]
    champion = command("champion", "Champion", "knights-models", 10)
    champion["profiles"].append({"name": "Champion", "typeName": "Model",
                                  "characteristics": [{"name": "A", "$text": "2"}]})
    chaos[1]["selections"].extend([
        champion, command("knight-standard", "Standard Bearer", "knights-models", 10),
        command("knight-musician", "Musician", "knights-models", 10)])
    standard = command("warrior-standard", "Standard Bearer", "warriors-models", 6)
    standard["selections"] = [item("banner", "The Banner Of The Bold", "Magic Standards", 10)]
    chaos[3]["selections"].extend([
        standard, command("warrior-musician", "Musician", "warriors-models", 6)])
    chaos[4]["selections"].append(command("horse-musician", "Musician", "horsemen-models", 5))
    return elves, chaos


def test_readiness_rosters_keep_counts_points_items_and_commands(roster_path, readiness_units):
    elves, chaos = [import_roster(roster_path(units)) for units in readiness_units]
    for army in (elves, chaos):
        assert army["budget"] == sum(unit["points_cost"] for unit in army["units"]) == 500
        assert json.loads(json.dumps(army)) == army
        assert army["roster_source"]["id"] == "test-roster"
    assert [unit["nmodels"] for unit in elves["units"]] == [1, 6, 5, 3, 1]
    assert [unit["nmodels"] for unit in chaos["units"]] == [1, 4, 5, 10, 5]
    assert [len(unit["command"]) for unit in chaos["units"]] == [0, 3, 0, 2, 1]
    mage = elves["units"][0]
    assert mage["wizard_level"] == 2 and mage["spells"] == []
    assert len(mage["spell_pool"]) == len(mage["spell_sources"]) == 10
    assert mage["spell_generation_pending"]
    assert all(source["kind"] == "pool" for source in mage["spell_sources"])
    assert mage["magic_items"][0]["name"] == "Silvery Wand"
    assert chaos["units"][0]["magic_items"][0]["name"] == "Helm Of Courage"
    assert chaos["units"][3]["magic_items"][0]["name"] == "The Banner Of The Bold"
    champion = chaos["units"][1]["command"][0]
    assert champion["profiles"][1]["characteristics"] == [{"name": "A", "$text": "2"}]
    assert chaos["units"][0]["armour"] == []


def test_explicit_spell_is_known_without_generating_more(roster_path):
    unit = {"type": "unit", "name": "Mage", "selections": [
        {"type": "upgrade", "name": "Shield of Saphery",
         "profiles": [spell_profile("Shield of Saphery", "shield")]},
    ]}
    imported = import_roster(roster_path([unit]))["units"][0]
    assert [spell["name"] for spell in imported["spells"]] == ["Shield of Saphery"]
    assert imported["spell_pool"] == []
    assert imported["spell_sources"][0]["kind"] == "selected"
    assert not imported["spell_generation_pending"]
    assert imported["wizard_level"] == 1


def test_item_instances_do_not_deduplicate_by_definition(roster_path):
    relic = {"type": "upgrade", "name": "Unknown Relic", "number": 1,
             "profiles": [{"id": "shared-definition", "name": "Unknown Relic",
                           "typeName": "Enchanted Items", "custom": {"retained": True}}]}
    unit = {"id": "unit", "type": "unit", "name": "Mage", "selections": [
        dict(relic, id="first"), dict(relic, id="second")]}
    first_import = import_roster(roster_path([unit]))
    items = first_import["units"][0]["magic_items"]
    assert len(items) == 2
    assert items[0]["selection_ref"] != items[1]["selection_ref"]
    assert items[0]["definition_id"] == items[1]["definition_id"]
    assert all(item["profiles"][0]["custom"]["retained"] for item in items)
    unit["selections"].reverse()
    reordered = import_roster(roster_path([unit]))["units"][0]["magic_items"]
    assert {item["selection_ref"] for item in items} == {item["selection_ref"] for item in reordered}


def test_import_preserves_source_and_is_repeatable(roster_path, readiness_units):
    path = roster_path(readiness_units[0])
    with open(path, encoding="utf-8") as source:
        original = source.read()
    first = import_roster(path)
    assert import_roster(path) == first
    with open(path, encoding="utf-8") as source:
        assert source.read() == original
    source_unit = json.loads(original)["roster"]["forces"][0]["selections"][0]
    records = first["units"][0]["roster_selections"]

    def restore(record):
        selected = {key: value for key, value in record.items()
                    if key not in {"ref", "parent_ref", "owner_ref"}}
        children = [restore(child) for child in records if child["parent_ref"] == record["ref"]]
        if children:
            selected["selections"] = children
        return selected

    assert restore(records[0]) == source_unit


def test_force_scoping_preserves_duplicate_selection_ids(tmp_path):
    unit = {"id": "same-id", "type": "unit", "name": "Model"}
    forces = [{"id": identity, "catalogueName": faction, "selections": [unit]}
              for identity, faction in [("elves", "High Elf Realms"), ("chaos", "Warriors of Chaos")]]
    path = tmp_path / "multiple_forces.json"
    path.write_text(json.dumps({"roster": {"forces": forces}}), encoding="utf-8")
    units = import_roster(str(path))["units"]
    assert units[0]["faction"] == "high_elf_realms"
    assert units[1]["faction"] == "warriors_of_chaos"
    assert units[0]["roster_selections"][0]["ref"] != units[1]["roster_selections"][0]["ref"]


def test_unknown_item_spell_does_not_create_a_wizard(roster_path):
    unit = {"type": "unit", "name": "Captain", "selections": [
        {"id": "relic", "type": "upgrade", "name": "Unknown Relic",
         "profiles": [{"name": "Unknown Relic", "typeName": "Enchanted Items"}],
         "selections": [{"type": "upgrade", "name": "Shield of Saphery",
                         "profiles": [spell_profile("Shield of Saphery", "shield")]}]},
    ]}
    imported = import_roster(roster_path([unit]))["units"][0]
    assert imported["wizard_level"] is None
    assert imported["spells"] == imported["spell_pool"] == []
    assert not imported["spell_generation_pending"]
    assert imported["spell_sources"][0]["kind"] == "item"
    assert imported["spell_sources"][0]["item_ref"] == imported["magic_items"][0]["selection_ref"]