import random

class skull_env():

    def __init__(self, player_num) -> None:
        super().__init__()
        # represents if players are still alive (have a hand)
        self.players = [True] * player_num

        # scores of the players, start at 0, win at 2
        self.scores = [0] * player_num

        # a 1 is a skull, a 0 is a flower
        self.hands = [[0, 0, 0, 1] for i in range(player_num)]
        # cards will go from hand to
        # treat at stack!
        self.played_space = [[] for i in range(player_num)]
        # tracks players bets for betting phase
        # acculuator for the attempt phase
        self.max_bet = -1
        self.current_bet = 0
        self.player_in_attempt = -1
        self.passed = [False] * player_num
        self.player_starts_next_round = -1

        # game phases
        # first add - every player must *add a card* to their play stack
        # add or bet - player may *add a card* or *raise*
        # raise - player muse *raise* or *pass*
        # attempt - player whose bet went through must *flip cards* starting with their own 
        self.game_phases = {
            "first_add": True, 
            "add_or_bet": False, 
            "raise": False,
            "attempt": False
            }

        # rewards
        '''
        self.rewards = {
            "place skull": 10,
            "flip flower": 100,
            "win challange": 1000,
            "lose challange": -100,
            "win": 10000,
            "lose": -10000
        }
        '''

        self.action_space = ["place flower", "place skull", "raise", "pass"]
        for i in range(player_num):
            self.action_space.append(f"turn {i}")

    # print who env for debugging
    def print_env(self):
        print("players: ", self.players)
        print("scores: ", self.scores)
        print("hands: ", self.hands)
        print("play area: ", self.played_space)
        print("passed: ", self.passed)
        print("max bet: ", self.max_bet)
        print("player in attempt: ", self.player_in_attempt)
        print("current bet: ", self.current_bet)
        print("game phases: ", self.game_phases)


    # returns all observable information for a specific player
    # will be used to determine agent actions
    def get_player_knowlage(self, player):
        # get a string representing game phase
        for key, val in self.game_phases.items():
            if (val): game_phase = key

        players_cards = []
        for i in range(len(self.hands)):
            players_cards.append(len(self.hands[i]) + len(self.played_space[i]))

        if (self.game_phases["first_add"]): 
            actions = ["place flower", "place skull"]
        elif (self.game_phases["add_or_bet"]):
            actions = ["place flower", "place skull", "raise"]
        elif (self.game_phases["raise"]): 
            actions = ["raise", "pass"]
        else:
            actions = []

            if (player == self.player_in_attempt and len(self.played_space[player]) != 0):
                actions.append(f"turn {player}")
            else:
                for i in range(len(self.played_space)):
                    if (len(self.played_space[i]) != 0):
                        actions.append(f"turn {i}")
                


        # note: not all of these things will be relevant to all action decsisions
        known = {
            "players": self.players,
            "scores": self.scores,
            "hand": self.hands[player],
            "played": self.played_space[player],
            "length_of_hands": [len(x) for x in self.hands],
            "length_of_played": [len(x) for x in self.played_space],
            "length_of_cards": players_cards, 
            "game_phase":game_phase,
            "passed": self.passed,
            "max_bet":self.max_bet,
            "player_in_attempt": self.player_in_attempt,
            "current_bet":self.current_bet,
            "action_set": actions
        }

        return known

    # checking for the termial case
    def players_left(self):
        #return [key for key, value in self.players.items() if value == True]
        players_int = []
        for i in range(len(self.players)):
            if (self.players[i]): players_int.append(i)
        return players_int
    
    def player_won(self):
        player = None
        for i in range(len(self.scores)):
            if self.scores[i] == 2:
                player = i
                break

        return player

    def is_terminal(self):
        if (2 in self.scores):
            #self.print_env()
            return [self.player_won(), "win via score", True]
        elif (len(self.players_left()) == 1):
            #self.print_env()
            return [self.players_left()[0], "win via elim", True]
        return False
        
    # actions in the place phase

    # card type 0 for flower, 1 for skull
    # player is an int
    def place_card(self, player, card_type):
        # dose the player have this card type in their hand
        if (card_type not in self.hands[player]):
            return [player, "fail: none of card type avaliable", 0]
        
        # remove card from hand
        self.hands[player].remove(card_type)

        # place card into play
        self.played_space[player].append(card_type)

        return [player, "played card", 0]
    
    # actions in the bet phase
    def raise_bet(self, player):

        if (self.max_bet == -1):
            self.max_bet = 1      
        else:
            self.max_bet += 1

        return [player, f"raised to {self.max_bet}", 0]
    
    def pass_bet(self, player):

        self.passed[player] = True

        return [player, "passed", 0]
    
    # actions in attempt phase

    # this assumes players will want to keep their skull 
    # unless it would lose them the game
    def remove_card_self(self, player):
        # player removes their last card, player has lost
        if (len(self.hands[player]) == 1):
            self.hands[player] = []
            self.players[player] = False
        # if player has 2 cards and 1 is a skull remove the skull
        # player cannot win with only a skull
        elif (len(self.hands[player]) == 2 and 1 in self.hands[player]):
            self.hands[player].remove(1)
        # player should remove a flower
        else:
            self.hands[player].remove(0)

    # failing on opp stack means they remove random card
    def remove_card_opp(self, player):
        if (len(self.hands[player]) == 1):
            self.hands[player] = []
            self.players[player] = False
        else:
            random_card = random.choice(self.hands[player])
            self.hands[player].remove(random_card)


    def turn_card(self, player, turned_player):
        # player has not turned their own cards first
        if (len(self.played_space[player])!= 0 and player != turned_player):
            return [player, "fail: must turn own cards first", 0]
        
        # player tried to turn stack with no cards
        if (len(self.played_space[turned_player]) == 0):
            return [player, "fail: no card to turn", 0]
        
        card = self.played_space[turned_player].pop()
        self.hands[turned_player].append(card)

        # card is a flower
        if (card == 0):
            self.current_bet += 1
            # check if player has won their challange

            if self.current_bet == self.max_bet:
                self.scores[player] += 1
                self.round_cleanup(player)
                return [player, "win challange", 100]
            else:
                return [player, "turned flower", 10]
            
        # card is a skull
        else:
            # if they turned own skull
            if (player == turned_player):
                self.round_cleanup(player)
                self.remove_card_self(player)
                
                if (len(self.hands[player]) == 0):
                    self.players[player] = False
                    self.passed[player] = True
                    self.player_starts_next_round = -1
                    return [player, "self lose game", -1000]
                else:
                    return [player, "self lose challange", -100]
            # if they turned opp skull
            else:
                self.round_cleanup(player)
                self.remove_card_opp(player)

                if (len(self.hands[player]) == 0):
                    self.players[player] = False
                    self.passed[player] = True
                    self.player_starts_next_round = -1
                    return [player, "lose game", -100]
                else:
                    return [player, "lose challange", -10]


    def choose_random_action(self):
        return random.choice(self.action_space)
    
    def chose_random_action_smart(self):
        if (self.game_phases["first_add"]):
            return random.choice(["place flower", "place skull"])
        elif (self.game_phases["add_or_bet"]):
            return random.choice(["place flower", "place skull", "raise"])
        elif (self.game_phases["raise"]):
            return random.choice(["raise", "pass"])
        elif (self.game_phases["attempt"]):
            turn_array = []
            for i in range(len(self.passed)):
                turn_array.append("turn "+str(i))
            return random.choice(turn_array)
        
    def move_to_add_or_bet(self):
        all_played = True
        for i in range(len(self.played_space)):
            if (len(self.played_space[i]) == 0 and (self.players[i])):
                all_played = False
                break

        return all_played
    
    def game_phase_first_add(self, player, action):
        # only legal action is place

        if ("place" not in action):
            action_result = [player, "fail: player must place card", 0]
        else:
            card = 0 if ("flower" in action) else 1
            action_result = self.place_card(player, card)
        
        #checking if we have moved to next game phase
        if (self.move_to_add_or_bet()):
            self.game_phases["first_add"] = False
            self.game_phases["add_or_bet"] = True
            
        return action_result

    def game_phase_add_or_bet(self, player, action):

        if (action not in ["place flower", "place skull", "raise"]):
            action_result = [player, "fail: player must place card or raise", 0]
        elif ("place" in action):
            card = 0 if ("flower" in action) else 1
            action_result = self.place_card(player, card)
        else:
            # if a player chooses to bet enter next phase
            self.game_phases["add_or_bet"] = False
            self.game_phases["raise"] = True
            action_result = self.raise_bet(player)

        return action_result

    def game_phase_raise(self, player, action):
        if (action not in ["raise", "pass"]):
            action_result = [player, "fail: must raise or pass", 0]
        elif (action == "raise"):
            action_result = self.raise_bet(player)
            self.player_in_attempt = player
        else:
            # player has chosen to pass
            action_result = self.pass_bet(player)
        
        # checking if bet is now raised to the number of cards on the board
        # if so go to attempt with the player who placed the bet in attempt
        if (self.max_bet == sum(len(p) for p in self.played_space)):
            self.game_phases["raise"] = False
            self.game_phases["attempt"] = True
            self.player_in_attempt = player
        # check if all but 1 players have passed
        # the unpassed player is now the attempting player
        elif (sum(self.passed) == (len(self.passed)-1)):
            self.game_phases["raise"] = False
            self.game_phases["attempt"] = True
            self.player_in_attempt = self.passed.index(False)

        return action_result
    
    def round_cleanup(self, player):
        # reset bet varibles
        self.max_bet = -1
        self.current_bet = 0
        self.player_in_attempt = -1
        self.player_starts_next_round = player

        # reset passed array
        for i in range(len(self.passed)):
            if (not self.players[i]):
                self.passed[i] = True
            else:
                self.passed[i] = False

        # put all cards back into hands
        for i in range(len(self.played_space)):
            for j in range(len(self.played_space[i])):
                card = self.played_space[i][j]
                self.hands[i].append(card)

            self.played_space[i] = []


    def game_phase_attempt(self, player, action):

        if ("turn" not in action):
            action_result = [player, "fail: must turn in attempt", 0]
        else:
            turned_player = int(action.split()[1])
            action_result = self.turn_card(player, turned_player)

        
        # has the attempt ended?
        if (action_result[1] in ["win challange", "self lose game", "self lose challange", "lose game", "lose challange"]):
            self.game_phases["attempt"] = False
            self.game_phases["first_add"] = True

        return action_result

    # exacute an action
    # this function redirects to a sub function for each game phase
    def play_action(self, player, action):

        # is the player out of the game
        if (not self.players[player]):
            return [player, "skip: out of cards", 0, False]
            
        
        # if starting a new round, is this the player who made the last attempt?
        if (self.player_starts_next_round != -1 and self.player_starts_next_round != player):
            return [player, f"skip: {self.player_starts_next_round} starts this round", 0, False]
        if (self.player_starts_next_round == player):
            self.player_starts_next_round = -1

        # otherwise perform action as normal
        if (self.game_phases["first_add"]):
            action_result = self.game_phase_first_add(player, action)
        elif (self.game_phases["add_or_bet"]):
            action_result = self.game_phase_add_or_bet(player, action)
        elif (self.game_phases["raise"]):
            # if player is passed skip their turn
            if (self.passed[player]):
                action_result = [player, "skip: currently passed", 0]
            else:
                action_result = self.game_phase_raise(player, action)
        elif (self.game_phases["attempt"]):
            # when in attempt phase only player in attempt takes actions
            # this is not a failed action but rather only 1 player is currenlty taking actions
            if (player != self.player_in_attempt):
                action_result = [player, "skip: not in attempt", 0]
            else:
                action_result = self.game_phase_attempt(player, action)

        # check is the game is over
        if (not self.is_terminal()):
            action_result.append(False)
        else:
            action_result.append(True)

        # action results in the form
        # [player num, descption of result, reward, is the game over?]
        return action_result


        

        


