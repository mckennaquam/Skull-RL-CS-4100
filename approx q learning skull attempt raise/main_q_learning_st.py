from skull_env import skull_env
from player_static import player_smart_2
from player_q_learn import player_qlearn_attempt_aprox
import pandas as pd
import time
import numpy as np

skull_thresholds = np.linspace(0, 1, 100)


df_results = pd.DataFrame(columns=["player_type", "count", "skull_threshold"])

for st in skull_thresholds:

    # set up the enviorment
    players = [
        player_qlearn_attempt_aprox(4, 0, st),
        player_smart_2(0, 0.4, 1),
        player_smart_2(2, 0.6, 2),
        player_smart_2(0, 0.6, 3)
    ]

    q_outcome = 0
    q_learn_raise_update = False
    def attempt_outcome_update(action_outcome):
        if ("lose" in action_outcome):
            return -1
        elif ("win" in action_outcome):
            return 1
        else:
            return 0 

    players_types = ["approx q learning",
                    "smart 2: 0, 0.4",
                    "smart 2: 2, 0.6",
                    "smart 2: 0, 0.6"]

    train_iter = 1000
    test_iter = 1000

    # train the agent
    for i in range(train_iter):
        env = skull_env(len(players))
        game_over = False

        while(not game_over):

            for player in range(len(players)):

                action_result = [player, "turn not taken", 0, False]
                while ((action_result[1] == "turn not taken") or ("fail" in action_result[1])):
                    player_knowlage = env.get_player_knowlage(player)
                    player_in_attempt = player_knowlage["player_in_attempt"]
                    game_phase = player_knowlage["game_phase"]

                    #if (player == 0): print(game_phase)

                    if (q_learn_raise_update and player == 0 and
                        ((game_phase == "raise" and (not player_knowlage["passed"][0])) or
                        (game_phase == "attempt" and player_in_attempt != 0) or
                        (game_phase == "first_add" and q_outcome != 0))):
                        
                        
                        update_q_results = players[0].update_q_values_raise(player_knowlage["action_set"], 
                                                        player_knowlage["played"], 
                                                        player_in_attempt, 
                                                        player_knowlage["length_of_played"], 
                                                        player_knowlage["max_bet"], 
                                                        game_phase, 
                                                        q_outcome)
                        
                        players[0].inc_num_eps_raise()
                        #players[0].print_weights_raise()
                        q_learn_raise_update = False
                        q_outcome = 0

                    action = players[player].chose_action(player_knowlage)

                    
                    # if the player in attempt is the q learning player
                    if (player == 0 and game_phase == "raise"):
                        action_result = env.play_action(0, action)

                        if ("skip" not in action_result[1]):
                            q_learn_raise_update  = True

                    elif (player == 0 and player_in_attempt == 0 and game_phase == "attempt"):
                        action_result = env.play_action(0, action)
                        next_state = env.get_player_knowlage(0)
                        # reward, actions, played_length, hand_length, game_state
                        update_q_results = players[0].update_q_values_attempt(action_result[2], 
                                                        next_state["action_set"], 
                                                        next_state["length_of_played"],
                                                        next_state["length_of_hands"],
                                                        next_state["game_phase"])
                        
                        players[0].inc_num_eps_attempt()
                        q_outcome = attempt_outcome_update(action_result[1])
                        
                    else:
                        
                        action_result = env.play_action(player, action)

                    game_over = action_result[3]

                if (game_over): break

            if (game_over): break

    print("finished training")
    players[0].training_done()


    df = pd.DataFrame(columns=['player_type', 'player', 'result'])

    # test the agent
    for i in range(test_iter):
        env = skull_env(len(players))
        game_over = False

        while(not game_over):

            for player in range(len(players)):

                action_result = [player, "turn not taken", 0, False]
                while ((action_result[1] == "turn not taken") or ("fail" in action_result[1])):
                    player_knowlage = env.get_player_knowlage(player)
                    player_in_attempt = player_knowlage["player_in_attempt"]
                    game_phase = player_knowlage["game_phase"]

                    action = players[player].chose_action(player_knowlage)
                    action_result = env.play_action(player, action)

                    game_over = action_result[3]


                if (game_over): break

            if (game_over): break

        game_result = env.is_terminal()
        df.loc[len(df)] = [players_types[game_result[0]], game_result[0], game_result[1]]


    df_temp = df.groupby('player_type')[['player_type']].value_counts().reset_index(name='count')
    df_temp["skull_threshold"] = st

    df_results = pd.concat([df_results, df_temp], axis=0)
    print(f"finished testing: {st}")


df_results.to_csv("simulation_data/q_learn_approx_v3/st_v3.csv")
   
