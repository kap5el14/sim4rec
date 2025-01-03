import pandas as pd

def performance(original_df: pd.DataFrame, normalized_df: pd.DataFrame, normalized_last_row: pd.Series) -> float:
    activity_col = 'concept:name'
    o_accepted = 'O_Accepted'
    if o_accepted not in original_df[activity_col].values:
        return 0
    index_o_accepted = original_df[activity_col].eq(o_accepted).idxmax()
    offer_id = original_df['OfferID'].loc[index_o_accepted]
    offered_amount = normalized_df[(normalized_df[activity_col] == 'O_Create Offer') & (normalized_df['EventID'] == offer_id)]['OfferedAmount'].iloc[0]
    if not (0 <= offered_amount <= 1):
        return ValueError("Performance must be within the [0,1] range.")
    return offered_amount