import pandas as pd
import sys

log_name = sys.argv[1]
size = int(sys.argv[2])
case_id_col = sys.argv[3]
df = pd.read_csv(f'user_files/log/{log_name}.csv')
unique_case_ids = set()
while len(unique_case_ids) < size:
    sampled_ids = df[case_id_col].sample(n=10, replace=False, random_state=None)
    unique_case_ids.update(sampled_ids)
unique_case_ids = list(unique_case_ids)[:size]
filtered_df = df[df[case_id_col].isin(unique_case_ids)]
filtered_df.to_csv(f'user_files/log/{log_name}_{size}.csv')