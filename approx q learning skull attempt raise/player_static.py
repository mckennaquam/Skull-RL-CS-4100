import random
import math

# totally random choice of actions
class player_rand:
    
    def __init__(self) -> None:
        pass

    def chose_action(self, known):
        return random.choice(known["action_set"])
    

# player always places their skull and always passes
# hops to get other people  
class player_mean:

    def __init__(self) -> None:
        pass
        
    def chose_action(self, known):
        if (known["game_phase"] == "first_add"):
            if (1 in known["hand"]):
                return "place skill"
            else:
                return "place flower"
        elif (known["game_phase"] == "add_or_bet"):
            if (len(known["hand"]) > 0):
                return "place flower"
            else:
                return "raise"
        elif (known["game_phase"] == "raise"):
            return "pass"
        # this should not happen
        elif (known["game_phase"] == "attempt"):
            return random.choice(known["action_set"])
        
# player only wants to place flowers and is excited to bet
class player_nice:
    def __init__(self) -> None:
        pass

    def chose_action(self, known):
        if (known["game_phase"] == "first_add"):
            if (0 in known["hand"]):
                return "place flower"
            else:
                return "place skull"

        elif (known["game_phase"] == "add_or_bet"):
            return "raise"

        elif (known["game_phase"] == "raise"):
            return "raise"

        # no theory of mind for attempt
        elif (known["game_phase"] == "attempt"):
            return random.choice(known["action_set"])


class player_smart():

    def __init__(self, skull_threshfold, raise_threshold, player_num) -> None:
        # lower skull threshold means more likely to place skull
        # lower raise thresholf means more likely to raise bet
        self.skull_threshfold = skull_threshfold
        self.raise_threshold = raise_threshold
        self.player_num = player_num

    # covers when there is an action which is forced by
    # the game state
    # note the raise phase doesn't have any action restictions
    def forced_actions(self, known):
        action = "unforced"
        if (known["game_phase"] == "first_add"):
            if (0 not in known["hand"]):
                action = "place skill"
            elif (1 not in known["hand"]):
                action = "place flower"

        elif (known["game_phase"] == "add_or_bet"):
            if (len(known["hand"]) == 0):
                action = "raise"

        elif (known["game_phase"] == "attempt"):
            if (len(known["played"]) > 0):
                action =  f"turn {self.player_num}"
            
        return action

    # func assumes we have a hand to choose from
    def choose_flower_or_skull(self, hand):
        if (1 in hand):
            place_skull = random.random()
            if (place_skull > self.skull_threshfold):
                return "place skull"
            else:
                return "place flower"
        else:
            return "place flower"


    def chose_action(self, known):
        # is there an action forced by the game state
        # ie turning your own cards in an attempt
        if (self.forced_actions(known) != "unforced"):
            return self.forced_actions(known)
        
        # first add logic
        # place skull if random num > skull threshold
        if (known["game_phase"] == "first_add"):
            return self.choose_flower_or_skull(known["hand"])
        elif (known["game_phase"] == "add_or_bet"):
            start_bet = random.random()
            if (start_bet > 0.5):
                return "raise"
            else:
                return self.choose_flower_or_skull(known["hand"])
                
        # raise logic
        # always raise is half # of played cards is greater than the current bet
        # if half # of played cards is less than current bet raise if rand higher than raise threshold
        elif (known["game_phase"] == "raise"):
            if (math.ceil(sum(known["length_of_played"]) / 2) >= (known["max_bet"] + 1)):
                return "raise"
            else:
                raise_bet = random.random()
                if (raise_bet > self.raise_threshold):
                    return "raise"
                else:
                    return "pass"
                
        # attempt logic
        # this version still has random
        else:
            return random.choice(known["action_set"])
        

# this player determins skull threshold via the number of cards 
# left in their hand rather than a flat number 
class player_smart_2:
    def __init__(self, skull_factor, raise_threshold, player_num) -> None:
        # lower skull threshold means more likely to place skull
        # lower raise thresholf means more likely to raise bet
        self.skull_factor = skull_factor
        self.raise_threshold = raise_threshold
        self.player_num = player_num

    # covers when there is an action which is forced by
    # the game state
    # note the raise phase doesn't have any action restictions
    def forced_actions(self, known):
        action = "unforced"
        if (known["game_phase"] == "first_add"):
            if (0 not in known["hand"]):
                action = "place skull"
            elif (1 not in known["hand"]):
                action = "place flower"

        elif (known["game_phase"] == "add_or_bet"):
            if (len(known["hand"]) == 0):
                action = "raise"

        elif (known["game_phase"] == "attempt"):
            if (len(known["played"]) > 0):
                action =  f"turn {self.player_num}"
            
        return action

        # func assumes we have a hand to choose from
    def choose_flower_or_skull(self, hand):
        if (1 in hand):
            place_skull = random.random()
            skull_threshold = (1 + self.skull_factor) / (len(hand) + self.skull_factor)
            if (place_skull > skull_threshold):
                return "place skull"
            else:
                return "place flower"
        else:
            return "place flower"
        
    def chose_action(self, known):
        # is there an action forced by the game state
        # ie turning your own cards in an attempt
        if (self.forced_actions(known) != "unforced"):
            return self.forced_actions(known)
        
        # first add logic
        # place skull if random num > skull threshold
        if (known["game_phase"] == "first_add"):
            return self.choose_flower_or_skull(known["hand"])
        elif (known["game_phase"] == "add_or_bet"):
            start_bet = random.random()
            if (start_bet > 0.5):
                return "raise"
            else:
                return self.choose_flower_or_skull(known["hand"])
                
        # raise logic
        # always raise is half # of played cards is greater than the current bet
        # if half # of played cards is less than current bet raise if rand higher than raise threshold
        elif (known["game_phase"] == "raise"):
            if (math.ceil(sum(known["length_of_played"]) / 2) >= (known["max_bet"] + 1)):
                return "raise"
            else:
                raise_bet = random.random()
                if (raise_bet > self.raise_threshold):
                    return "raise"
                else:
                    return "pass"
                
        # attempt logic
        # this version still has random
        else:
            return random.choice(known["action_set"])