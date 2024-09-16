from util.common import *
from algo.recommendation import make_recommendations
from algo.sampling import sample_peers

def recommend(folds: list[Common]):
    Common.set_instance(folds[0])
    common = Common.instance
    sampled_test_case_id = random.choice(list(common.test_df[common.event_log_specs.case_id]))
    df = common.test_df[common.test_df[common.event_log_specs.case_id] == sampled_test_case_id]
    idx = random.choice(list(range(1, len(df) + 1)))
    df = df.head(idx)
    print(df[common.event_log_specs.activity])
    print('\n\n')
    recommendations = make_recommendations(dfs=sample_peers(df=df), df=df)
    for r in recommendations:
        print(r)
