from skull_env import skull_env
import time
import pandas as pd
import sys
from player_static import player_rand 
from player_static import player_mean
from player_static import player_nice
from player_static import player_smart
from player_static import player_smart_2


iterations = 10000
skull_factors = [0, 1, 2, 3]

df_total = pd.DataFrame(columns=["skull_factor", "raise_thershold", "win_count_score", "win_count_elim"])

for sf in skull_factors:

    df = pd.DataFrame(columns=['player_type', 'player', 'result'])

    players = [player_smart_2(sf, 0.6, 0),
            player_rand(),
            player_rand(),
            player_rand()]

    players_types = [f"smart 2: {sf}, 0.6", "rand", "rand", "smart: rand"]

    for i in range(iterations):

        env = skull_env(len(players))
        game_over = False

        while (not game_over):
            for player in range(len(players)):

                action_result = [player, "turn not taken", False]
                while ((action_result[1] == "turn not taken") or ("fail" in action_result[1])):
                    action = players[player].chose_action(env.get_player_knowlage(player))
                    action_result = env.play_action(player, action)
                    game_over = action_result[2]

                if (game_over): break

            if (game_over): break

        df.loc[len(df)] = [players_types[action_result[0]], action_result[0], action_result[1]]

    # record the win values for the smart player
    df = df[df["player_type"] == f"smart 2: {sf}, 0.6"].groupby("result").value_counts().reset_index(name='count')
    
    win_via_score = df[df["result"] == "win via score"]
    win_via_score_count = win_via_score["count"].iloc[0] if not win_via_score.empty else 0
    
    win_via_elim = df[df["result"] == "win via elim"]
    win_via_elim_count =  win_via_elim["count"].iloc[0] if not win_via_elim.empty else 0
    
    df_total.loc[len(df_total)] = [sf, 0.6, win_via_score_count, win_via_elim_count]

    print(f"done with player: sf = {sf}")
    df_total.to_csv("smart2_sf.csv")
        
    #df.to_csv("simulation_data/four_smart.csv")




