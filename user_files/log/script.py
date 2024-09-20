import pandas as pd
df = pd.read_csv('bpic.csv')
unique_case_ids = set()
while len(unique_case_ids) < 100:
    sampled_ids = df['case:concept:name'].sample(n=10, replace=False, random_state=None)
    unique_case_ids.update(sampled_ids)
unique_case_ids = list(unique_case_ids)[:100]
filtered_df = df[df['case:concept:name'].isin(unique_case_ids)]
filtered_df.to_csv('bpic_mini.csv')