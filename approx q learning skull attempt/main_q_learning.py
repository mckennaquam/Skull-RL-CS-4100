from skull_env import skull_env
from player_static import player_smart_2
from player_q_learn import player_qlearn_attempt_aprox
import pandas as pd
import time


# set up the enviorment
players = [
    player_qlearn_attempt_aprox(4),
    player_smart_2(0, 0.4, 1),
    player_smart_2(2, 0.6, 2),
    player_smart_2(0, 0.6, 3)
]

# for recoding game results to pd
players_types = ["approx q learning",
                 "smart 2: 0, 0.4",
                 "smart 2: 2, 0.6",
                 "smart 2: 0, 0.6"]

# set up the number of train and test episodes
train_iter = 1000
test_iter = 1000

# train the agent
for i in range(train_iter):
    # create the skull env
    env = skull_env(len(players))
    game_over = False

    # while the game isnt over
    while(not game_over):

        # each players take turns
        for player in range(len(players)):

            # ensures no turns skipped due to illegal action choice
            action_result = [player, "turn not taken", 0, False]
            while ((action_result[1] == "turn not taken") or ("fail" in action_result[1])):
                # get the information which the player knows about the envirment
                player_knowlage = env.get_player_knowlage(player)
                player_in_attempt = player_knowlage["player_in_attempt"]
                game_phase = player_knowlage["game_phase"]

                # player chooses which action to take based on env state varibles
                action = players[player].chose_action(player_knowlage)
                
                # if the player in attempt is the q learning player perforem Q learning
                if (player == 0 and player_in_attempt == 0 and game_phase == "attempt"):
                    # play the chosen action (greedy epsilon)
                    action_result = env.play_action(0, action)
                    # get the resulting state
                    next_state = env.get_player_knowlage(0)
                    # update the weights for the feature vector
                    # reward, actions, played_length, hand_length, game_state
                    players[0].update_q_values(action_result[2], 
                                                    next_state["action_set"], 
                                                    next_state["length_of_played"],
                                                    next_state["length_of_hands"],
                                                    next_state["game_phase"])
                    # increase the number of eps, decrease the epsilon
                    players[0].inc_num_eps()
                # otherwise play action    
                else:
                    action_result = env.play_action(player, action)

                # check if game is over
                game_over = action_result[3]

            # if game over terminate game
            if (game_over): break

        if (game_over): break


# q agent has finished training and will no longer update weights
print("finished training")
players[0].training_done()

# set up data frame for recoding results
df = pd.DataFrame(columns=['player_type', 'player', 'result'])
df_attempts = pd.DataFrame(columns=['player type', 'player', 'result'])

# test the agent
for i in range(test_iter):
    # create the skull env
    env = skull_env(len(players))
    game_over = False

    # while the game isnt over
    while(not game_over):

        # each players take turns
        for player in range(len(players)):

            # ensures no turns skipped due to illegal action choice
            action_result = [player, "turn not taken", 0, False]

            # ensures no turns skipped due to illegal action choice
            while ((action_result[1] == "turn not taken") or ("fail" in action_result[1])):
                # get the information which the player knows about the envirment
                player_knowlage = env.get_player_knowlage(player)

                # player chooses which action to take based on env state varibles
                # (q player handles state eval under the hood, dont need to invoke any specaized functions here)
                action = players[player].chose_action(player_knowlage)
                action_result = env.play_action(player, action)

                # check if game is over
                game_over = action_result[3]

            # if the action was in the attempt phase record results
            if (action_result[1] in ["turned flower", "self lose game", "self lose challange", "lose game", "lose challange", "win challange"]):
                    df_attempts.loc[len(df_attempts)] = [players_types[action_result[0]], action_result[0], action_result[1]]

            # if game over terminate game
            if (game_over): break

        if (game_over): break

    # record game outcome
    game_result = env.is_terminal()
    df.loc[len(df)] = [players_types[game_result[0]], game_result[0], game_result[1]]

# save results to dataframe for analysis
df.to_csv("qlearn_approx_attempt_game_outcomes.csv")
df_attempts.to_csv("qlearn_approx_attempt_attempt_outcomes.csv")
