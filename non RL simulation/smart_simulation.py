from skull_env import skull_env
import pandas as pd
import numpy as np
from player_static import player_rand 
from player_static import player_smart
from player_static import player_smart_2

#skull_probs = np.linspace(0, 1, 25)
skull_factor = np.linspace(-0.5, 3, 10)
raise_probs = np.linspace(0, 1, 10)
iterations = 10000

#skull_factor = [2]
#raise_probs = [0.6]

df_total = pd.DataFrame(columns=["skull_factor", "raise_thershold", "win_count_score", "win_count_elim"])

for sf in skull_factor:
    for rp in raise_probs:
        # for each combo of skull probabilty and bet probaility
        df = pd.DataFrame(columns=['player_type', 'player', 'result'])

        players = [player_smart_2(sf, rp, 0),
           player_rand(), 
           player_rand(), 
           player_rand()]
        
        players_types = [f"smart 2: {sf}, {rp}", "rand", "rand", "rand"]

        # run # iterations of games (10,000)
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
        df = df[df["player_type"] == f"smart 2: {sf}, {rp}"].groupby("result").value_counts().reset_index(name='count')
        
        win_via_score = df[df["result"] == "win via score"]
        win_via_score_count = win_via_score["count"].iloc[0] if not win_via_score.empty else 0
        
        win_via_elim = df[df["result"] == "win via elim"]
        win_via_elim_count =  win_via_elim["count"].iloc[0] if not win_via_elim.empty else 0
        
        df_total.loc[len(df_total)] = [sf, rp, win_via_score_count, win_via_elim_count]

        print(f"done with player: sf = {sf}, rp = {rp}")
        df_total.to_csv("smart2_sf_rp.csv")

