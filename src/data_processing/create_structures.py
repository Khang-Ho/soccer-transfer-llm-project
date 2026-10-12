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

tabular_columns = {
    "player_id": "Player ID",
    "player_name": "Player Name",
    "date_of_birth": "Date of Birth",
    "country_of_citizenship": "Country of Citizenship",
    "height_in_cm": "Height (cm)",
    "position": "Position",
    "current_club_name": "Current Club",
    "transfer_date": "Transfer Date",
    "from_club_name": "From Club",
    "to_club_name": "To Club",
    "transfer_season": "Transfer Season",
    "transfer_fee": "Transfer Fee",
    "market_value_in_eur": "Market Value (EUR)"
}
tabular_data = enriched_transfer[list(tabular_columns)].copy().rename(
    columns=tabular_columns
)

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
    values = to_key_value(row)
    values = {
        key: "unknown" if value is None else value
        for key, value in values.items()
    }
    return (
        f"{values['Player Name']} has player ID {values['Player ID']}. "
        f"The player's date of birth is {values['Date of Birth']}, "
        f"country of citizenship is {values['Country of Citizenship']}, "
        f"height in cm is {values['Height (cm)']}, "
        f"and position is {values['Position']}. "
        f"The player's current club is {values['Current Club']}. "
        f"On {values['Transfer Date']}, they transferred from "
        f"{values['From Club']} to {values['To Club']} "
        f"in the {values['Transfer Season']} season. "
        f"The transfer fee is {values['Transfer Fee']}, "
        f"and the market value in EUR is {values['Market Value (EUR)']}."
    )

def to_hier_nested(row):
    values = to_key_value(row)
    return {
        "Player": {
            "Player ID": values["Player ID"],
            "Player Name": values["Player Name"],
            "Personal Information": {
                "Date of Birth": values["Date of Birth"],
                "Country of Citizenship": values["Country of Citizenship"],
                "Height (cm)": values["Height (cm)"]
            },
            "Career Information": {
                "Position": values["Position"],
                "Current Club": values["Current Club"]
            }
        },
        "Transfer": {
            "Transfer Date": values["Transfer Date"],
            "Transfer Season": values["Transfer Season"],
            "Clubs": {
                "From Club": values["From Club"],
                "To Club": values["To Club"]
            },
            "Financial Information": {
                "Transfer Fee": values["Transfer Fee"],
                "Market Value (EUR)": values["Market Value (EUR)"]
            }
        }
    }

