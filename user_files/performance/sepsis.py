import pandas as pd
from util.constants import *

def performance(original_df: pd.DataFrame, normalized_df: pd.DataFrame, normalized_last_row: pd.Series) -> float:
    occurrences_returns = normalized_df[normalized_df['event'] == 'Return ER'][ACTIVITY_OCCURRENCE].iloc[-1] if 'Return ER' in normalized_df['event'].values else 0
    if occurrences_returns:
        return 0
    performance = 1 - (normalized_last_row[TRACE_LENGTH] + normalized_last_row[TRACE_DURATION]) / 2
    if not (0 <= performance <= 1):
        raise ValueError(f"Performance={performance} not within the [0,1] range.")
    return performance