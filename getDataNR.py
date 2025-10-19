import matplotlib.pyplot as plt
import numpy as np
from models import *
from units import *
from toHitAndToWound import *
from battleFunctions import *
from battleGraphs import *


    




    

url_black_orc = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/orc-and-goblin-tribes/907e-90b-b5a5-a8a3/black-orc"
url_man_at_arm = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/kingdom-of-bretonnia/3ddf-271a-aaec-73eb/man-at-arms"
url_saurus_warrior = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/lizardmen/65aee1f-83430cad/saurus-warrior"
url_night_goblin = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/orc-and-goblin-tribes/f241-11e2-3771-3b16/night-goblin"
url_orc_boyz = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/orc-and-goblin-tribes/9d5a-280f-c336-5226/orc-boy"
url_knight_of_the_realm = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/kingdom-of-bretonnia/54ce-96e7-b7e1-3b4b/mounted-knight-of-the-realm"
url_bretonnian_warhorse = "https://www.newrecruit.eu/wiki/tow/warhammer-the-old-world/kingdom-of-bretonnia/71c3-30e-c81-cb64/bretonnian-warhorse"
black_orc = BlackOrc("Black Orc", url_black_orc)
black_orc.armor_save = 3
man_at_arm = model("Man_at_Arm", url_man_at_arm)
man_at_arm.armor_save = 7
saurus_warrior = SaurusWarrior("Saurus Warrior", url_saurus_warrior)
saurus_warrior.armor_save = 4
saurus_warrior.equip_weapon('halberd')
#print(saurus_warrior.weapons)
print(saurus_warrior.special_rules)

night_goblin = NightGoblin("Night Goblin", url_night_goblin)
night_goblin.armor_save = 7
night_goblin.equip_weapon('short bow')

orc_boy = OrcBoyz("Orc Boyz", url_orc_boyz)
orc_boy.armor_save = 5
bretonnian_warhorse = BretonnianWarhorse("Bretonnian Warhorse", url_bretonnian_warhorse)
bretonnian_warhorse.armor_save = 7
bretonnian_warhorse_unit = unit("Bretonnian Warhorse Unit", bretonnian_warhorse, 5,5,1)
mounted_knight_of_the_realm = MountedKnightOfTheRealm("Mounted Knight of the Realm", url_knight_of_the_realm, mountUnit=bretonnian_warhorse_unit)
mounted_knight_of_the_realm.armor_save = 3
mounted_knight_of_the_realm.equip_weapon('lance')




black_orc_unit = unit("Black Orc Unit", black_orc, 10,5,2)
man_at_arm_unit = unit("Man_at_Arm Unit", man_at_arm, 10,5,2)
saurus_warrior_unit = unit("Saurus Warrior Unit", saurus_warrior, 10,5,2)
night_goblin_unit = unit("Night Goblin Unit", night_goblin, 20,20,2)
orc_boy_unit = unit("Orc Boy Unit", orc_boy, 20,5,4)
mounted_knight_of_the_realm_unit = unit("Mounted Knight of the Realm Unit", mounted_knight_of_the_realm, 5,5,1)


print(black_orc.characteristics)
print(man_at_arm.characteristics)
print(saurus_warrior.characteristics)

#store_dict_to_file(black_orc.characteristics, 'black_orc_characteristics.json')
#store_dict_to_file(man_at_arm.characteristics, 'man_at_arm_characteristics.json')



to_hit_result = to_hit(black_orc, man_at_arm)
print(to_hit_result)
to_wound_result = to_wound(black_orc, man_at_arm)
print(to_wound_result)

results_attacker = []
results_defender = []
""" 
attacker = black_orc_unit
defender = saurus_warrior_unit

for i in range(1000):
    defender.nmodels=10
    attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(attacker, defender,charge=True)
    result = [attacks, total_hits, suffered_wounds,  saves_made, total_wounds]
    results_attacker.append(result)
    print(f"Total hits by {attacker.name} on {defender.name}: {total_hits}")
    print(f"suffered wounds by {attacker.name} on {defender.name}: {suffered_wounds}")
    print(f"Saves made by {defender.name}: {saves_made}")
    print(f"Total wounds by {attacker.name} on {defender.name}: {total_wounds}")
    #battle_graph(attacks, total_hits, suffered_wounds, saves_made, total_wounds)
    defender.nmodels-=total_wounds
    attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(defender, attacker,charge=False)
    result = [attacks, total_hits, suffered_wounds,  saves_made, total_wounds]
    results_defender.append(result)
    print(f"Total hits by {defender.name} on {attacker.name}: {total_hits}")
    print(f"suffered wounds by {defender.name} on {attacker.name}: {suffered_wounds}")
    print(f"Saves made by {attacker.name}: {saves_made}")
    print(f"Total wounds by {defender.name} on {attacker.name}: {total_wounds}\n")
    #battle_graph(attacks, total_hits, suffered_wounds, saves_made, total_wounds)
 """

attacker = mounted_knight_of_the_realm_unit
defender = orc_boy_unit

for i in range(1000):
    defender.nmodels=10
    #attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle_ranged(attacker, defender,charge=False)
    attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(attacker, defender,charge=True)
    result = [attacks, total_hits, suffered_wounds,  saves_made, total_wounds]
    results_attacker.append(result)
    print(f"Total hits by {attacker.name} on {defender.name}: {total_hits}")
    print(f"suffered wounds by {attacker.name} on {defender.name}: {suffered_wounds}")
    print(f"Saves made by {defender.name}: {saves_made}")
    print(f"Total wounds by {attacker.name} on {defender.name}: {total_wounds}")
    defender.nmodels-=total_wounds
    #battle_graph(attacks, total_hits, suffered_wounds, saves_made, total_wounds)
    if True: #if unit models has mounts do attacks with the mounts
        attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(attacker.model.special_rules[1]['mountUnit'], defender,charge=True)
        result = [attacks, total_hits, suffered_wounds,  saves_made, total_wounds]
        # Combine mount's attack result with rider's attack result
        if results_attacker:
            last_rider_result = results_attacker.pop()  # Remove the last result (rider's attack)
            sum_result = [(a + b)  for a, b in zip(result, last_rider_result)]
            results_attacker.append(sum_result)
        else:
            results_attacker.append(result)
        print(f"Total hits by {attacker.model.special_rules[1]['mountUnit'].name} on {defender.name}: {total_hits}")
        print(f"suffered wounds by {attacker.model.special_rules[1]['mountUnit'].name} on {defender.name}: {suffered_wounds}")
        print(f"Saves made by {defender.name}: {saves_made}")
        print(f"Total wounds by {attacker.model.special_rules[1]['mountUnit'].name} on {defender.name}: {total_wounds}")
        #battle_graph(attacks, total_hits, suffered_wounds, saves_made, total_wounds)
        defender.nmodels-=total_wounds

    
    attacks, total_hits, suffered_wounds,  saves_made, total_wounds = simulate_battle(defender, attacker,charge=False)
    result = [attacks, total_hits, suffered_wounds,  saves_made, total_wounds]
    results_defender.append(result)
    print(f"Total hits by {defender.name} on {attacker.name}: {total_hits}")
    print(f"suffered wounds by {defender.name} on {attacker.name}: {suffered_wounds}")
    print(f"Saves made by {attacker.name}: {saves_made}")
    print(f"Total wounds by {defender.name} on {attacker.name}: {total_wounds}\n")
    #battle_graph(attacks, total_hits, suffered_wounds, saves_made, total_wounds)

def analyze_results(unit, results):
    results_attacker = np.array(results)
    attacks=results_attacker[:,0].mean()
    hits=results_attacker[:,1].mean()
    suffered_wounds=results_attacker[:,2].mean()
    saves=results_attacker[:,3].mean()
    total_wounds=results_attacker[:,4].mean()
    battle_graph(unit, attacks, hits, suffered_wounds, saves, total_wounds)
    return results_attacker

results_attacker = analyze_results(attacker, results_attacker)
results_defender = analyze_results(defender, results_defender)

""" results_defender = np.array(results_defender)
attacks=results_defender[:,0].mean()
hits=results_defender[:,1].mean()
suffered_wounds=results_defender[:,2].mean()
saves=results_defender[:,3].mean()
total_wounds=results_defender[:,4].mean()
battle_graph(attacks, hits, suffered_wounds, saves, total_wounds) """

attacker_wins = results_attacker[:,4]-results_defender[:,4]
#plt.figure()
#plt.plot(attacker_wins)
#plt.title('Attacker Wins (Positive means attacker wins)')
plt.figure()
#plt.hist(attacker_wins, bins=10, edgecolor='black', align='mid')
plt.title('Histogram of Attacker Wins')

# Annotate each bar with its value
# Center bins on integer numbers
min_win = int(np.floor(attacker_wins.min()))
max_win = int(np.ceil(attacker_wins.max()))
num_attacker_wins = np.sum(attacker_wins > 0)
print(f"Number of times attacker wins: {num_attacker_wins}")
num_attaker_losts = np.sum(attacker_wins < 0)
print(f"Number of times attacker losts: {num_attaker_losts}")
num_attacker_draws = np.sum(attacker_wins == 0)
print(f"Number of times attacker draws: {num_attacker_draws}")
bins = np.arange(min_win - 0.5, max_win + 1.5, 1)
counts, bins, patches = plt.hist(attacker_wins, bins=bins, edgecolor='black', align='mid')
for count, bin_left, patch in zip(counts, bins[:-1], patches):
    plt.text(bin_left + (bins[1] - bins[0]) / 2, count, f"{int(count)}", ha='center', va='bottom')
#print(results_attacker)
#print(results_attacker[:,0])
#print(results_defender)
plt.show()