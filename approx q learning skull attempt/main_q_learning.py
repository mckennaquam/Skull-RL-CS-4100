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

                action = players[player].chose_action(player_knowlage)

                #print(game_phase)
                
                # if the player in attempt is the q learning player
                if (player == 0 and player_in_attempt == 0 and game_phase == "attempt"):
                    action_result = env.play_action(0, action)
                    next_state = env.get_player_knowlage(0)
                    # reward, actions, played_length, hand_length, game_state
                    players[0].update_q_values(action_result[2], 
                                                    next_state["action_set"], 
                                                    next_state["length_of_played"],
                                                    next_state["length_of_hands"],
                                                    next_state["game_phase"])
                    players[0].inc_num_eps()
                    
                else:
                    
                    action_result = env.play_action(player, action)

                game_over = action_result[3]

                ''' 
                print(action, action_result)
                print(env.get_player_knowlage(player)["game_phase"])
                print("-------------------")

                time.sleep(1)
                '''

            if (game_over): break

        if (game_over): break


print("finished training")
players[0].training_done()


df = pd.DataFrame(columns=['player_type', 'player', 'result'])
df_attempts = pd.DataFrame(columns=['player type', 'player', 'result'])


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

            if (action_result[1] in ["turned flower", "self lose game", "self lose challange", "lose game", "lose challange", "win challange"]):
                    df_attempts.loc[len(df_attempts)] = [players_types[action_result[0]], action_result[0], action_result[1]]

            if (game_over): break

        if (game_over): break

    game_result = env.is_terminal()
    df.loc[len(df)] = [players_types[game_result[0]], game_result[0], game_result[1]]

df.to_csv("qlearn_approx_attempt_game_outcomes.csv")
df_attempts.to_csv("qlearn_approx_attempt_attempt_outcomes.csv")
