from util.common import *
from algo.sampling import sample_peers
from algo.recommendation import make_recommendation
import pyperclip
from json2html import *

class PreprocessingException(Exception):
    def __init__(self, message):
        super().__init__(f"Couldn't preprocess trace due to the following exception:\n{message}")

class SamplingException(Exception):
    def __init__(self, message):
        super().__init__(f"Couldn't sample peers due to the following exception:\n{message}")

class RecommendationException(Exception):
    def __init__(self, message):
        super().__init__(f"Couldn't compute recommendations due to the following exception:\n{message}")

class VisualizationException(Exception):
    def __init__(self, message):
        super().__init__(f"Couldn't visualize recommendations due to the following exception:\n{message}")

def recommendation_pipeline(df: pd.DataFrame, interactive=True):
    common = Common.instance
    try:
        df.columns = df.columns.str.strip()
        df[common.conf.event_log_specs.timestamp] = pd.to_datetime(df[common.conf.event_log_specs.timestamp], format='mixed')
        df, _ = common.preprocess(df=df)
    except Exception as e:
        raise PreprocessingException(str(e))
    try:
        peers = sample_peers(df)
    except Exception as e:
        raise SamplingException(str(e))
    try:
        recommendation = make_recommendation(dfs=peers, df=df)
        if not recommendation:
            return
        if interactive:
            json_output = recommendation.to_json()
            pyperclip.copy(json_output)
    except Exception as e:
        raise RecommendationException(str(e))
    try:
        if interactive:
            html_content = json2html.convert(json=json_output)
            with open("output.html", "w") as file:
                file.write(html_content)
    except Exception as e:
        raise VisualizationException(str(e))
    return recommendation