from skull_env import skull_env
from player_static import player_smart_2
from player_q_learn import player_qlearn_attempt_aprox
import pandas as pd
import time
import numpy as np

# version of the main file for testing different skull thresholds for the Q agent with feature set 3

skull_thresholds = np.linspace(0, 1, 100)

df_results = pd.DataFrame(columns=["player_type", "count", "skull_threshold"])


# for each skull threshold (testing 100 values equally spaced from 0-1)
for st in skull_thresholds:

    # set up the enviorment
    players = [
        player_qlearn_attempt_aprox(4, 0, st),
        player_smart_2(0, 0.4, 1),
        player_smart_2(2, 0.6, 2),
        player_smart_2(0, 0.6, 3)
    ]

    # helper function for recoding the outcome of the q players attempt
    q_outcome = 0
    q_learn_raise_update = False
    def attempt_outcome_update(action_outcome):
        if ("lose" in action_outcome):
            return -1
        elif ("win" in action_outcome):
            return 1
        else:
            return 0 

    # for recoding game results to csv
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
                    # get the information which the player knows about the enviorment
                    player_knowlage = env.get_player_knowlage(player)
                    player_in_attempt = player_knowlage["player_in_attempt"]
                    game_phase = player_knowlage["game_phase"]

                    # if statement for updating the weights for the raise phase
                    # we need to update weights when
                    # non terminal = q agent still in raise
                    # terminal - opponante in attempt
                    # terminal - q agent lost or won attempt
                    if (q_learn_raise_update and player == 0 and
                        ((game_phase == "raise" and (not player_knowlage["passed"][0])) or
                        (game_phase == "attempt" and player_in_attempt != 0) or
                        (game_phase == "first_add" and q_outcome != 0))):
                        
                        # update the weights for the feature vector
                        update_q_results = players[0].update_q_values_raise(player_knowlage["action_set"], 
                                                        player_knowlage["played"], 
                                                        player_in_attempt, 
                                                        player_knowlage["length_of_played"], 
                                                        player_knowlage["max_bet"], 
                                                        game_phase, 
                                                        q_outcome)
                        
                        # inc number of episodes, desc epsilon
                        players[0].inc_num_eps_raise()
                        # we no longer need to update weights
                        q_learn_raise_update = False
                        q_outcome = 0

                    # player chooses which action to take based on env state varibles
                    action = players[player].chose_action(player_knowlage)

                    # if player is in the raise phase and is the q player
                    if (player == 0 and game_phase == "raise"):
                        # play the chosen action (greedy epsilon)
                        action_result = env.play_action(0, action)

                        # only update weights if we perform an action
                        # player "performs" an action but is skipped by the env
                        # so only turn flag to true if action does not have skip
                        if ("skip" not in action_result[1]):
                            q_learn_raise_update  = True

                     # if the player in attempt is the q learning player
                    elif (player == 0 and player_in_attempt == 0 and game_phase == "attempt"):
                        # play the chosen action (greedy epsilon)
                        action_result = env.play_action(0, action)
                        # get the resulting state
                        next_state = env.get_player_knowlage(0)
                        # reward, actions, played_length, hand_length, game_state
                        # update the weights for the feature vector 
                        update_q_results = players[0].update_q_values_attempt(action_result[2], 
                                                        next_state["action_set"], 
                                                        next_state["length_of_played"],
                                                        next_state["length_of_hands"],
                                                        next_state["game_phase"])
                        
                        # increase the number of eps, decrease the epsilon
                        players[0].inc_num_eps_attempt()
                        # update the attempt outcome for updating the raise phase weights
                        q_outcome = attempt_outcome_update(action_result[1])
                    #otherwise play action     
                    else:
                        action_result = env.play_action(player, action)

                    # check if game is over
                    game_over = action_result[3]

                if (game_over): break

            if (game_over): break

    # q agent has finished training and will no longer update weights
    print("finished training")
    players[0].training_done()

    # set up data frame for recoding results
    df = pd.DataFrame(columns=['player_type', 'player', 'result'])

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
                while ((action_result[1] == "turn not taken") or ("fail" in action_result[1])):
                    # get the information which the player knows about the envirment
                    player_knowlage = env.get_player_knowlage(player)
                    player_in_attempt = player_knowlage["player_in_attempt"]
                    game_phase = player_knowlage["game_phase"]

                    # player chooses which action to take based on env state varibles
                    # (q player handles state eval under the hood, dont need to invoke any specaized functions here)
                    action = players[player].chose_action(player_knowlage)
                    action_result = env.play_action(player, action)

                    # check if game is over
                    game_over = action_result[3]

                # if game over terminate game
                if (game_over): break

            if (game_over): break

        # record game outcome
        game_result = env.is_terminal()
        df.loc[len(df)] = [players_types[game_result[0]], game_result[0], game_result[1]]


    # sum game outcomes for the skull threshold train
    df_temp = df.groupby('player_type')[['player_type']].value_counts().reset_index(name='count')
    df_temp["skull_threshold"] = st

    # record game win counts to result dataframe
    df_results = pd.concat([df_results, df_temp], axis=0)
    print(f"finished testing: {st}")

# save result dataframe
df_results.to_csv("simulation_data/q_learn_approx_v3/st_v3.csv")
   
