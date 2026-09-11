"""Declare each model's divided attacks before rolling To Hit (pp. 147, 199, 209)."""

from dataclasses import dataclass, field

from panda3d.core import Vec3

from rules_log import rule_log


def nearest_targets(candidates, *, epsilon=0.01):
    """Contact takes precedence; otherwise only the nearest enemy unit is legal."""
    nearest = min((distance for _, distance in candidates), default=float('inf'))
    return [target for target, distance in candidates if distance <= nearest + epsilon]


@dataclass
class AttackAllocation:
    owner: object
    profile: object
    batches: list
    attacks: list = field(default_factory=list)

    async def resolve(self, game):
        """Mandatory attacks may be divided only between the model's legal targets."""
        allocated = {}
        for slot, count, targets in self.batches:
            if not targets or count <= 0:
                continue
            options = {target.unitName: target for target in targets}
            if len(targets) == 1 or game.aiControls(self.owner):
                assignments = [(targets[0], count)]
            else:
                assignments = []
                for attack in range(count):
                    selected = await game.makeChoiceNew(list(options), Vec3(0, 0, 10), owner=self.owner,
                        prompt=f'{self.profile.name}, model {slot + 1}: attack {attack + 1}/{count}')
                    assignments.append((options.get(selected, targets[0]), 1))
            for target, attacks in assignments:
                identity = id(target)
                previous = allocated.get(identity, (target, 0))[1]
                allocated[identity] = (target, previous + attacks)
        self.attacks = list(allocated.values())
        for target, attacks in self.attacks:
            rule_log('Dividing Attacks', self.owner,
                     f'{self.profile.name}: {attacks} attack(s) allocated to {target.unitName} before rolling (p. 147)')