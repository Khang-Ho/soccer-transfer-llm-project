import pandas as pd
import numpy as np

transfer_data = pd.read_csv("data/transfers.csv")
player_data = pd.read_csv("data/players.csv")
player_info = player_data[
    ["player_id",
     "country_of_citizenship",
     "date_of_birth",
     "position",
     "height_in_cm",
     "current_club_name"
     ]
]

enriched_transfer = transfer_data.merge(player_info, how = "left", on = "player_id")

def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value

integer_columns = ["player_id", 
                   "height_in_cm",
                   "transfer_fee",
                   "market_value_in_eur"]
for col in integer_columns:
    enriched_transfer[col] = enriched_transfer[col].astype("Int64")

def to_key_value(row):
    return {
        "Player ID": clean_value(row["player_id"]),
        "Player Name": clean_value(row["player_name"]),
        "Date of Birth": clean_value(row["date_of_birth"]),
        "Country of Citizenship": clean_value(row["country_of_citizenship"]),
        "Height (cm)": clean_value(row["height_in_cm"]),
        "Position": clean_value(row["position"]),
        "Current Club": clean_value(row["current_club_name"]),
        "Transfer Date": clean_value(row["transfer_date"]),
        "From Club": clean_value(row["from_club_name"]),
        "To Club": clean_value(row["to_club_name"]),
        "Transfer Season": clean_value(row["transfer_season"]),
        "Transfer Fee": clean_value(row["transfer_fee"]),
        "Market Value (EUR)": clean_value(row["market_value_in_eur"])
    }

def to_flat(row):
    return (
        
    )

test_row = enriched_transfer.iloc[0]
print(to_key_value(test_row))  