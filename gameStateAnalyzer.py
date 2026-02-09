class GameStateAnalyzer:
    """Heuristic functions to evaluate the current game state"""
    
    def __init__(self, game):
        self.game = game
    
    def calculate_army_strength(self, player_units):
        """
        Calculate total army strength based on surviving models and their stats.
        Returns a numerical score representing army power.
        """
        total_strength = 0
        
        for unit in player_units:
            if unit.bodyNP.isEmpty():
                continue
                
            # Base strength from number of models
            model_strength = unit.unit.nmodels
            
            # Multiply by combat effectiveness (WS, S, T, A)
            ws = int(unit.unit.model.characteristics.get('WS', 3))
            strength = int(unit.unit.model.characteristics.get('S', 3))
            toughness = int(unit.unit.model.characteristics.get('T', 3))
            attacks = int(unit.unit.model.characteristics.get('A', 1))
            
            combat_multiplier = (ws + strength + toughness + attacks) / 10.0
            
            # Factor in armor save (lower is better)
            armor = unit.unit.model.armor_save
            armor_multiplier = 1 + (7 - armor) * 0.1 if armor < 7 else 1.0
            
            # Leadership factor for staying power
            leadership = int(unit.unit.model.characteristics.get('Ld', 7))
            ld_multiplier = leadership / 7.0
            
            unit_strength = model_strength * combat_multiplier * armor_multiplier * ld_multiplier
            
            # Bonus for mounted units
            for rule in unit.unit.model.special_rules:
                if rule.get('mountUnit'):
                    unit_strength *= 1.5
            
            total_strength += unit_strength
        
        return total_strength
    
    def calculate_position_advantage(self, player_units, enemy_units):
        """
        Evaluate positional advantage based on flanking opportunities,
        formation integrity, and battlefield control.
        """
        position_score = 0
        
        for unit in player_units:
            if unit.bodyNP.isEmpty():
                continue
            
            unit_pos = unit.bodyNP.getPos()
            
            # Check for flanking positions on enemies
            for enemy in enemy_units:
                if enemy.bodyNP.isEmpty():
                    continue
                    
                enemy_pos = enemy.bodyNP.getPos()
                distance = (unit_pos - enemy_pos).length()
                
                if distance < 20:  # Within threatening range
                    # Calculate if unit is flanking
                    enemy_heading = enemy.bodyNP.getH()
                    angle_to_unit = self._get_angle_to_target(enemy_pos, enemy_heading, unit_pos)
                    
                    if 45 < abs(angle_to_unit) < 135:  # Flank position
                        position_score += 15
                    elif abs(angle_to_unit) > 135:  # Rear position
                        position_score += 30
            
            # Formation integrity bonus
            if unit.unit.ranks >= 3:
                position_score += 5 * unit.unit.ranks
            
            # Center field control
            if abs(unit_pos.x) < 10 and abs(unit_pos.y) < 10:
                position_score += 10
        
        return position_score
    
    def calculate_momentum(self, player_units):
        """
        Assess army momentum based on movement, charges, and morale.
        Positive values indicate offensive momentum.
        """
        momentum = 0
        
        for unit in player_units:
            if unit.bodyNP.isEmpty():
                continue
            
            # Units that haven't moved yet have potential
            if not unit.hasMovedThisTurn and unit.state == "Idle":
                momentum += 5
            
            # Charging units have high momentum
            if unit.state == "InCombat" and unit.unit.model.charging:
                momentum += 20
            
            # Pursuing units
            if unit.state == "IsPursuing":
                momentum += 15
            
            # Fleeing units are negative momentum
            if unit.state == "IsFleeing":
                momentum -= 30
            
            # Fresh units (not attacked yet)
            if not unit.hasAttackedThisTurn and unit.state != "IsFleeing":
                momentum += 3
        
        return momentum
    
    def calculate_combat_potential(self, player_units):
        """
        Evaluate the potential damage output of all units.
        """
        combat_potential = 0
        
        for unit in player_units:
            if unit.bodyNP.isEmpty():
                continue
            
            attacks = int(unit.unit.model.characteristics.get('A', 1))
            strength = int(unit.unit.model.characteristics.get('S', 3))
            ws = int(unit.unit.model.characteristics.get('WS', 3))
            
            # Base combat value
            unit_combat = unit.unit.nmodels * attacks * (ws / 3.0) * (strength / 3.0)
            
            # Weapon multiplier
            if unit.unit.model.equipedWeapon:
                weapon_strength = unit.unit.model.equipedWeapon.get('strength_bonus', 0)
                unit_combat *= (1 + weapon_strength * 0.1)
            
            # Charge bonus potential
            if not unit.hasMovedThisTurn:
                unit_combat *= 1.3  # Potential to charge
            
            combat_potential += unit_combat
        
        return combat_potential
    
    def evaluate_overall_state(self, player_num):
        """
        Comprehensive evaluation returning a score and assessment string.
        Positive values favor the player, negative values favor the opponent.
        """
        player_units = self.game.player1Units if player_num == 1 else self.game.player2Units
        enemy_units = self.game.player2Units if player_num == 1 else self.game.player1Units
        
        # Calculate all metrics
        army_strength = self.calculate_army_strength(player_units)
        enemy_strength = self.calculate_army_strength(enemy_units)
        
        position = self.calculate_position_advantage(player_units, enemy_units)
        enemy_position = self.calculate_position_advantage(enemy_units, player_units)
        
        momentum = self.calculate_momentum(player_units)
        enemy_momentum = self.calculate_momentum(enemy_units)
        
        combat_potential = self.calculate_combat_potential(player_units)
        enemy_combat_potential = self.calculate_combat_potential(enemy_units)
        
        # Weighted total score
        strength_diff = (army_strength - enemy_strength) * 2.0
        position_diff = (position - enemy_position) * 1.0
        momentum_diff = (momentum - enemy_momentum) * 0.5
        combat_diff = (combat_potential - enemy_combat_potential) * 1.5
        
        total_score = strength_diff + position_diff + momentum_diff + combat_diff
        
        # Generate assessment
        assessment = self._generate_assessment(
            total_score, 
            army_strength, 
            enemy_strength,
            momentum,
            position
        )
        
        return {
            'total_score': total_score,
            'assessment': assessment,
            'metrics': {
                'army_strength': army_strength,
                'enemy_strength': enemy_strength,
                'position_advantage': position,
                'momentum': momentum,
                'combat_potential': combat_potential
            }
        }
    
    def _get_angle_to_target(self, enemy_pos, enemy_heading, unit_pos):
        """Helper to calculate angle from enemy facing to unit position"""
        import math
        
        direction = unit_pos - enemy_pos
        angle_to_target = math.degrees(math.atan2(direction.y, direction.x))
        relative_angle = angle_to_target - enemy_heading
        
        # Normalize to -180 to 180
        while relative_angle > 180:
            relative_angle -= 360
        while relative_angle < -180:
            relative_angle += 360
            
        return relative_angle
    
    def _generate_assessment(self, score, strength, enemy_strength, momentum, position):
        """Generate human-readable assessment"""
        if score > 100:
            return "Dominant Position - Victory Likely"
        elif score > 50:
            return "Strong Advantage"
        elif score > 20:
            return "Slight Advantage"
        elif score > -20:
            return "Balanced - Tactical Decisions Critical"
        elif score > -50:
            return "Slight Disadvantage"
        elif score > -100:
            return "Difficult Position"
        else:
            return "Critical Situation - Survival at Stake"
    
    def suggest_strategy(self, player_num):
        """
        Suggest strategic approach based on current state.
        """
        evaluation = self.evaluate_overall_state(player_num)
        score = evaluation['total_score']
        metrics = evaluation['metrics']
        
        player_units = self.game.player1Units if player_num == 1 else self.game.player2Units
        
        if score > 50:
            return "AGGRESSIVE: Press the advantage with charges and flanking maneuvers"
        elif score > 0:
            return "OPPORTUNISTIC: Seek favorable engagements, avoid unfavorable trades"
        elif metrics['momentum'] > 0:
            return "DEFENSIVE: Consolidate position, use terrain and formations"
        else:
            # Count fleeing units
            fleeing_count = sum(1 for u in player_units if u.state == "IsFleeing")
            if fleeing_count > len(player_units) / 3:
                return "DESPERATE: Rally fleeing units, defensive formations critical"
            else:
                return "CAUTIOUS: Minimize losses, look for rally opportunities"


# Usage in game.py:
# Add to MyApp.__init__:
# self.state_analyzer = GameStateAnalyzer(self)

# Then you can call:
# evaluation = self.state_analyzer.evaluate_overall_state(player_num=1)
# print(f"Player 1 Assessment: {evaluation['assessment']}")
# print(f"Total Score: {evaluation['total_score']:.1f}")
# strategy = self.state_analyzer.suggest_strategy(player_num=1)
# print(f"Suggested Strategy: {strategy}")