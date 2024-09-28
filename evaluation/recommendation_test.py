from util.common import *
from algo.recommendation import make_recommendations, RecommendationPackage
from algo.sampling import sample_peers

def recommend(folds: list[Common]):
    Common.set_instance(folds[0])
    common = Common.instance
    sampled_test_case_id = random.choice(list(common.test_df[common.conf.event_log_specs.case_id]))
    df = common.test_df[common.test_df[common.conf.event_log_specs.case_id] == sampled_test_case_id]
    idx = random.choice(list(range(1, len(df) + 1)))
    df = df.head(idx)
    original_df = common.conf.df.loc[df.index]
    print(original_df[common.conf.output_format.numerical_attributes + common.conf.output_format.categorical_attributes + common.conf.output_format.timestamp_attributes])
    print('\n\n')
    recommendation_package = make_recommendations(dfs=sample_peers(df=df), df=df)
    print(recommendation_package)

