import pandas as pd
from util.constants import *

def performance(original_df: pd.DataFrame, normalized_df: pd.DataFrame, normalized_last_row: pd.Series) -> float:
    activities = original_df["event"].values
    occurrences_nc = normalized_df[normalized_df['event'] == 'Admission NC'][ACTIVITY_OCCURRENCE].iloc[-1] if 'Admission NC' in normalized_df['event'].values else 0
    occurrences_ic = normalized_df[normalized_df['event'] == 'Admission IC'][ACTIVITY_OCCURRENCE].iloc[-1] if 'Admission IC' in normalized_df['event'].values else 0
    occurrences_returns = normalized_df[normalized_df['event'] == 'Return ER'][ACTIVITY_OCCURRENCE].iloc[-1] if 'Return ER' in normalized_df['event'].values else 0
    performance = 1 - (normalized_last_row[TRACE_LENGTH] + normalized_last_row[TRACE_DURATION] + occurrences_returns) / 3
    if not (0 <= performance <= 1):
        raise ValueError(f"Performance={performance} not within the [0,1] range.")
    return performance