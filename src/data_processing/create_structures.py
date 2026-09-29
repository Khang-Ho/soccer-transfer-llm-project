import pandas as pd

transfer_data = pd.read_csv("data/transfer_data.csv")
player_data = pd.read_csv("data/player_data.csv")
player_info = player_data[
    ["player_id",
     "country_of_citizenship",
     "date_of_birth",
     "position",
     "height_in_cm",
     "current_club_name"
     ]
]

enriched_transfer = transfer_data.merge(player_info, how = "left", left_on = "player_id", right_on = "player_id")

def to_key_value(row):
    return {
        "player_id": row["player_id"],
        "player_name": row["player_name"],
        "position": row["position"],
        "nationality": row["country_of_citizenship"],
        "current_club": row["current_club_name"],
        "market_value": row["market_value"],
        "transfer_fee": row["transfer_fee"],
    }