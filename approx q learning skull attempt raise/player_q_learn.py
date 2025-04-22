import random

# helper func for dot product
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
    
def weights_diff(weight_old, weights_current):
    if (len(weight_old) != len(weights_current)):
        print("weight_old: ", weight_old)
        print("weights_current: ", weights_current)
        raise Exception("weights old and weights current are different lengths!!")
    else:
        acc = 0
        for i in range(len(weights_current)):
            acc += abs(weight_old[i] - weights_current[i])

        return acc


class player_qlearn_attempt_aprox:

    def __init__(self, num_players, player_num, skull_threshold=0.5) -> None:
        self.train_attempt = True
        self.train_raise = True

        # will tinker with these later
        self.skull_threshold = skull_threshold
        #self.skull_factor = 0
        #self.raise_threshold = 0

        # APPROX Q LEARNING FOR RAISE PHASE
        # version 1
        # feature 0 = 0/1 do we have a skull down
        # feature 1 = if we havent played skull # of flowers down for us, else 0
        # feature 2 = 0/1 are we currently the player in attempt
        # feature 3 = # of opp cards down
        # feature 4 = current max bet

        # version 2
        # feature 0 = 0/1 do we have a skull down
        # feature 1 = the number of cards played
        # feature 2 = current max bet

        # version 3
        # feature 0 = 0/1 do we have a skull down
        # feture 1 = the % we have to turn for the next bet 
        # (max bet + 1) / cards on the table

        # 5 features for version 1
        # 3 features for version 2
        # 2 features for version 3
        self.num_features_raise = 2

        # initalize vector of 0 for weights
        # 2 actions (raise, pass) with 5 features each
        self.w_raise = [0] * 2 * self.num_features_raise
        # for determining convergance
        self.w_old_raise = []

        # for updating weights
        # last action is the string prepresenting last action pass/raise
        # state vector features 1-5
        self.last_action_raise = ""
        self.last_state_raise = [0,0,0]
        self.last_q_value_raise = 0

        self.player_num = player_num

        self.epsilon_raise = 1
        self.num_episodes_raise = 0
        self.gamma_raise = 0.9
        self.alpha_raise = 0.5

        self.raise_rewards = {
            "other_skull": 10,
            "other_no_skull": -1,
            "self_win": 100,
            "self_lose": 0,
            "non_terminal": 0
        }


        # APPROX Q LEARNING FOR ATTEMPT PHASE

        # we are creating features for the attempt phase
        # feature 0 = how many cards are in player i's pile
        # feature 1 = how many cards are in player i's hand
        # we have actions = # players being "turn i"
        # so for action turn 0 in a 4 player game the features look like
        # [# cards in pile 0, # cards in hand 0, 0, 0, 0, 0, 0] 
        # for turn 3 that looks like
        # [0, 0, 0, 0, 0, 0, # cards in pile 3, # cards in hand 3] 
        
        # define weights initalize all to 0
        self.w_attempt = [0] * 2 * num_players

        self.w_old_attempt = []

        # for updating weights
        # will be in the from player turned, # of cards in play, # of cards in hand
        self.last_state_attempt = [0,0,0]
        self.last_q_value_attempt = 0

        self.num_players = num_players

        self.epsilon_attempt = 1
        self.num_episodes_attempt = 0
        self.gamma_attempt = 0.9
        self.alpha_attempt = 0.5


    def choose_card_to_play(self, hand):
        if (1 in hand):
            place_skull = random.random()
            if (place_skull > self.skull_threshold):
                return "place skull"
            else:
                return "place flower"
        else:
            return "place flower"
        
    def training_done(self):
        self.train_attempt = False
        self.train_raise = False
        

    #-------------------------APPROX Q FOR ATTEMPT---------------------------------    
    def inc_num_eps_attempt(self):
        self.num_episodes_attempt += 1
        self.epsilon_attempt -= (1/(self.num_episodes_attempt+1))

    # for debugging
    def print_weights_attempt(self):
        print(self.w_attempt)
        
    def evaluate_q_values_attempt(self, actions, played_length, hand_length):
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

            q_value_for_action = dot_product(self.w_attempt, feature_vector)
            q_values[p] = q_value_for_action

        player_to_turn = max(q_values, key=q_values.get)
        max_q_val_info = [player_to_turn, played_length[player_to_turn], hand_length[player_to_turn], max(q_values.values())]

        return max_q_val_info
    
    def update_q_values_attempt(self, reward, actions, played_length, hand_length, game_state):
        # get the important information from last q values
        player_turned = self.last_state_attempt[0]
        turned_played_length = self.last_state_attempt[1]
        turned_hand_length = self.last_state_attempt[2]

        # save old weights for finding difference for convergance
        self.w_old_attempt = self.w_attempt.copy()

        # we are still int the attempt get the next q value
        if (game_state == "attempt"):
            next_q_info = self.evaluate_q_values_attempt(actions, played_length, hand_length)
            next_q = next_q_info[3]
        # we are out of the attempt, q value = 0
        else:
            next_q = 0

        # find the difference in approx q value
        difference = (reward + self.gamma_attempt * next_q) - self.last_q_value_attempt

        # update our weights
        # update # of played for player i weight
        self.w_attempt[player_turned * 2] = self.w_attempt[player_turned * 2] + self.alpha_attempt * difference * turned_played_length
        # update # in hand for player i weight
        self.w_attempt[(player_turned * 2) + 1] = self.w_attempt[(player_turned * 2) + 1] + self.alpha_attempt * difference * turned_hand_length

        w_diff = weights_diff(self.w_old_attempt, self.w_attempt)

        # returns the eps number with the difference between the old and new weights
        return [self.num_episodes_attempt, w_diff]

    #-------------------------APPROX Q FOR RAISE---------------------------------
    def inc_num_eps_raise(self):
        self.num_episodes_raise += 1
        self.epsilon_raise -= (1/(self.num_episodes_raise+1))

    # for debugging
    def print_weights_raise(self):
        print(self.w_raise)   
        
    def evaluate_q_values_raise(self, actions, played, player_in_attempt, played_length, max_bet):
        # calculate feature vector
        features = []

        '''
        # feature 0 = 1/0 if we have a skull down
        # feature 1 = if we have a skull down 0 else # of flowers played
        if (1 in played):
            features.append(1)
            features.append(0)
        else:
            features.append(0)
            features.append(len(played))
        # feature 2 = 0/1 are we currently the player in attempt
        if (self.player_num == player_in_attempt):
            features.append(1)
        else:
            features.append(0)
        # feature 3 = # of opp cards down
        acc = 0
        for i in range(len(played_length)):
            if (i != self.player_num):
                acc += played_length[i]
        features.append(acc)
        # feature 4 = current max bet
        features.append(max_bet)
        '''

        '''
        # version 2
        # feature 0 = 0/1 do we have a skull down
        if (1 in played):
            features.append(1)
        else:
            features.append(0)
        # feature 1 = the number of cards played
        acc = 0
        for i in range(len(played_length)):
            acc += played_length[i]
        features.append(acc)
        # feature 2 = current max bet
        features.append(max_bet)
        '''
        
        # version 3
        # feature 0 = 0/1 do we have a skull down
        if (1 in played):
            features.append(1)
        else:
            features.append(0)
        # feature 1 = the % we have to turn of others piles for the next bet
        # (max bet + 1) / cards on the table
        features.append((max_bet+1)/sum(played_length))

        # evaluate Q values for each action
        q_values = {}
        for a in actions:
            if (a == "raise"):
                feature_vector = features + ([0] * self.num_features_raise)
                q_values["raise"] = dot_product(self.w_raise, feature_vector)
            else:
                feature_vector =  ([0] * self.num_features_raise) + features 
                q_values["pass"] = dot_product(self.w_raise, feature_vector)

        # return the action with max Q value as well as the state and its q value
        action_to_take = max(q_values, key=q_values.get)
        max_q_val_info = [action_to_take, features, max(q_values.values())]
        return max_q_val_info

    
    def update_q_values_raise(self, actions, played, player_in_attempt, played_length, max_bet, game_phase, attempt_outcome):
        # feature 0 = 0/1 do we have a skull down
        # feature 1 = if we havent played skull # of flowers down for us, else 0
        # feature 2 = 0/1 are we currently the player in attempt
        # feature 3 = # of opp cards down
        # feature 4 = current max bet

        #save the old weights for fiding the difference
        self.w_old_raise = self.w_raise.copy()
        
        # rewards is state dependent so we are calcuating it here
        # did we reach a terminal state?
        if (game_phase == "attempt"):
            # other play is in attempt and we have a skull down
            if (self.last_state_raise[0] == 1):
                reward = self.raise_rewards["other_skull"]
                reward_result = "other player skull"
            # other player is in attempt and we do not have a skull down
            else:
                reward_result = "other player no skull"
                reward = self.raise_rewards["other_no_skull"]
            next_q = 0
        elif (game_phase == "first_add"):
            # we have completed a attempt and won
            if (attempt_outcome == 1):
                reward = self.raise_rewards["self_win"]
                reward_result = "attempt won"
            # we have completed an attempt and lost
            else:
                reward_result = "attempt loss"
                reward = self.raise_rewards["self_lose"]
            next_q = 0
        # non terminal state
        else:
            reward_result = self.raise_rewards["non_terminal"]
            next_q_info = self.evaluate_q_values_raise(actions, played, player_in_attempt, played_length, max_bet)
            next_q = next_q_info[2]
            reward = 0

        # find the difference in approx q value
        difference = (reward + self.gamma_raise * next_q) - self.last_q_value_raise

        # update our weights
        for f in range(len(self.last_state_raise)):
            if (self.last_action_raise == "raise"):
                self.w_raise[f] = self.w_raise[f] + self.alpha_raise * difference * self.last_state_raise[f]
            else:
                self.w_raise[f + self.num_features_raise] = self.w_raise[f + self.num_features_raise] + self.alpha_raise * difference * self.last_state_raise[f]
        
        # find the difference for plotting convergance
        w_diff = weights_diff(self.w_old_raise, self.w_raise)

        return [self.num_episodes_raise, w_diff, reward_result]

    #-------------------------APPROX Q PLAY ACTION MANAGER---------------------------------
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
            '''
            raise_bet = random.random()
            # will likely change
            if (raise_bet > self.raise_threshold):
                return "raise"
            else:
                return "pass"
            '''
            played = known["played"]
            player_in_attempt = known["player_in_attempt"]
            played_length = known["length_of_played"]
            max_bet = known["max_bet"]
            actions = known["action_set"]


            if (self.train_raise):
                if (random.random() > self.epsilon_raise):
                    q_value_info = self.evaluate_q_values_raise(actions, played, player_in_attempt, played_length, max_bet)
                    self.last_action_raise = q_value_info[0]
                    self.last_state_raise = q_value_info[1]
                    self.last_q_value_raise = q_value_info[2]
                    return self.last_action_raise
                else:
                    next_action = random.choice(known["action_set"])
                    q_value_info = self.evaluate_q_values_raise([next_action], played, player_in_attempt, played_length, max_bet)
                    self.last_action_raise = q_value_info[0]
                    self.last_state_raise = q_value_info[1]
                    self.last_q_value_raise = q_value_info[2]
                    return self.last_action_raise
            else:
                action_to_take = self.evaluate_q_values_raise(actions, played, player_in_attempt, played_length, max_bet)
                return action_to_take[0]
        else:
            played_length = known["length_of_played"]
            hand_length = known["length_of_cards"]
            if (self.train_attempt):
                if (random.random() > self.epsilon_attempt):
                    q_value_info = self.evaluate_q_values_attempt(known["action_set"], played_length, hand_length)
                    self.last_state_attempt = q_value_info[0:3]
                    self.last_q_value_attempt = q_value_info[3]
                    return "turn " + str(q_value_info[0])
                else:
                    next_turn = random.choice(known["action_set"])
                    q_value_info = self.evaluate_q_values_attempt([next_turn], played_length, hand_length)
                    self.last_state_attempt = q_value_info[0:3]
                    self.last_q_value_attempt = q_value_info[3]
                    return next_turn
            else:
                player_to_turn = self.evaluate_q_values_attempt(known["action_set"], played_length, hand_length)
                return "turn " + str(player_to_turn[0])
            