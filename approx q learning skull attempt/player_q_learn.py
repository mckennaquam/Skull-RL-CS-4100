
import random
import numpy as np

def dot_product(w, f):
    if (len(w) != len(f)):
        print("w: ", w)
        print("f: ", f)
        raise Exception("weights and features are different lengths!!")
    else:
        acc = 0
        for i in range(len(w)):
            acc += w[i] * f[i]

        return acc


class player_qlearn_attempt_aprox:

    def __init__(self, num_players) -> None:
        self.train = True

        # will tinker with these later
        self.skull_threshold = 1
        self.raise_threshold = 0

        # we are creating features for the attempt phase
        # feature 1 = how many cards are in player i's pile
        # feature 2 = how many cards are in player i's hand
        # we have actions = # players being "turn i"
        # so for action turn 0 in a 4 player game the features look like
        # [# cards in pile 0, # cards in hand 0, 0, 0, 0, 0, 0] 
        # for turn 3 that looks like
        # [0, 0, 0, 0, 0, 0, # cards in pile 3, # cards in hand 3] 
        
        # define weights initalize all to 0
        self.w = [0] * 2 * num_players

        # for updating weights
        # will be in the from player turned, # of cards in play, # of cards in hand
        self.last_state = [0,0,0]
        self.last_q_value = 0

        self.num_players = num_players

        self.epsilon = 1
        self.num_episodes = 0
        self.gamma = 0.9
        self.alpha = 0.5


    def choose_card_to_play(self, hand):
        if (1 in hand):
            place_skull = random.random()
            if (place_skull > self.skull_threshold):
                return "place skull"
            else:
                return "place flower"
        else:
            return "place flower"
        
    def inc_num_eps(self):
        self.num_episodes += 1
        self.epsilon -= (1/(self.num_episodes+1))

    def training_done(self):
        self.train = False

    # for debugging
    def print_weights(self):
        print(self.w)
        
    def evaluate_q_values(self, actions, played_length, hand_length):
        # get player nums from turn i
        players_avaliable = []

        for action in actions:
            player = int(action.split()[1])
            players_avaliable.append(player)

        #for each action avaible!
        q_values = {}
        for p in players_avaliable:
            # generate feature vector
            feature_vector = []
            for i in range(self.num_players):
                # if the player we are looking at is the player in the action
                if (i == p):
                    # feature 1 length of played
                    feature_vector.append(played_length[p])
                    # feature 2 length of hand
                    feature_vector.append(hand_length[p])
                else:
                    feature_vector.append(0)
                    feature_vector.append(0)

            q_value_for_action = dot_product(self.w, feature_vector)
            q_values[p] = q_value_for_action

        player_to_turn = max(q_values, key=q_values.get)
        #self.last_state = [player_to_turn, played_length[player_to_turn], hand_length[player_to_turn]]
        #self.last_q_value = max(q_values.values())
        max_q_val_info = [player_to_turn, played_length[player_to_turn], hand_length[player_to_turn], max(q_values.values())]

        return max_q_val_info
    
    def update_q_values(self, reward, actions, played_length, hand_length, game_state):
        # get the important information from last q values
        player_turned = self.last_state[0]
        turned_played_length = self.last_state[1]
        turned_hand_length = self.last_state[2]

        # we are still int the attempt get the next q value
        if (game_state == "attempt"):
            next_q_info = self.evaluate_q_values(actions, played_length, hand_length)
            next_q = next_q_info[3]
        # we are out of the attempt, q value = 0
        else:
            next_q = 0

        # find the difference in approx q value
        difference = (reward + self.gamma * next_q) - self.last_q_value

        # update our weights
        # update # of played for player i weight
        self.w[player_turned * 2] = self.w[player_turned * 2] + self.alpha * difference * turned_played_length
        # update # in hand for player i weight
        self.w[(player_turned * 2) + 1] = self.w[(player_turned * 2) + 1] + self.alpha * difference * turned_hand_length

    def forced_actions(self, game_phase, hand):
        if (game_phase == "first_add"):
            if (1 not in hand):
                return "place flower"
            elif (0 not in hand):
                return "place skull"
        elif (game_phase == "add_or_raise"):
            if (len(hand) == 0):
                return "raise"
        
        return "unforced"


    def chose_action(self, known):
        game_phase = known["game_phase"]

        if (self.forced_actions(game_phase, known["hand"]) != "unforced"):
            return self.forced_actions(game_phase, known["hand"])

        if (game_phase == "first_add"):
            return self.choose_card_to_play(known["hand"])
        elif(game_phase == "add_or_bet"):
            start_bet = random.random()
            if (start_bet > 0.5):
                return "raise"
            else:
                return self.choose_card_to_play(known["hand"])
        elif(game_phase == "raise"):
            raise_bet = random.random()
            # will likely change
            if (raise_bet > self.raise_threshold):
                return "raise"
            else:
                return "pass"
        else:
            played_length = known["length_of_played"]
            hand_length = known["length_of_cards"]
            if (self.train):
                if (random.random() > self.epsilon):
                    q_value_info = self.evaluate_q_values(known["action_set"], played_length, hand_length)
                    self.last_state = q_value_info[0:3]
                    self.last_q_value = q_value_info[3]
                    return "turn " + str(q_value_info[0])
                else:
                    next_turn = random.choice(known["action_set"])
                    q_value_info = self.evaluate_q_values([next_turn], played_length, hand_length)
                    self.last_state = q_value_info[0:3]
                    self.last_q_value = q_value_info[3]
                    return next_turn
            else:
                player_to_turn = self.evaluate_q_values(known["action_set"], played_length, hand_length)
                return "turn " + str(player_to_turn[0])
            