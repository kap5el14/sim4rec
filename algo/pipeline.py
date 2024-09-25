from util.common import *
from algo.sampling import sample_peers
from algo.recommendation import recommend_scenarios, Scenario
import pyperclip

class PreprocessingException(Exception):
    def __init__(self, message):
        super().__init__(f"Couldn't preprocess trace due to the following exception:\n{message}")

class SamplingException(Exception):
    def __init__(self, message):
        super().__init__(f"Couldn't sample peers due to the following exception:\n{message}")

class RecommendationException(Exception):
    def __init__(self, message):
        super().__init__(f"Couldn't recommend scenarios due to the following exception:\n{message}")

def recommendation_pipeline(df: pd.DataFrame):
    common = Common.instance
    try:
        df.columns = df.columns.str.strip()
        df[common.conf.event_log_specs.timestamp] = pd.to_datetime(df[common.conf.event_log_specs.timestamp], format='mixed')
        df = common.preprocess(df=df)
    except Exception as e:
        raise PreprocessingException(str(e))
    try:
        peers = sample_peers(df)
    except Exception as e:
        raise SamplingException(str(e))
    try:
        scenarios = recommend_scenarios(dfs=peers, df=df)
        json_output = Scenario.to_json(scenarios=scenarios)
        pyperclip.copy(json_output)
    except Exception as e:
        raise RecommendationException(str(e))
    return scenarios