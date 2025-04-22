from skull_env import skull_env
from player_static import player_rand
import time

print("turn 3".split()[1])

'''

players = [player_rand(), player_rand(), player_rand(), player_rand()]

for iter in range(1):
    
    env = skull_env(len(players))
    game_ended = False

    while (not game_ended):

        for i in range(len(players)):

            action_result = [i, "turn not taken", 0, False]
            while ((action_result[1] == "turn not taken") or ("fail" in action_result[1])):
                player_knowlage = env.get_player_knowlage(i)
                player_in_attempt = player_knowlage["player_in_attempt"]
                action = players[i].chose_action(player_knowlage)


                if (i == player_in_attempt):
                    print("------------------")
                    env.print_env()
                    action_result = env.play_action(i, action)
                    print("played: ", action)
                    env.print_env()
                    print("------------------")
                else:
                    action_result = env.play_action(i, action)
                    #print(action, action_result)
                
                
                game_ended = action_result[3]

                
                if(game_ended): break

            if(game_ended): break
'''

