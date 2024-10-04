from common import *
from algo.sampling import sample_peers
from algo.recommendation import make_recommendation, Recommendation
import pyperclip
from json2html import *

class Pipeline:
    def __init__(self, df: pd.DataFrame):
        common = Common.instance
        self.df = df
        self.df.columns = self.df.columns.str.strip()
        self.df[common.conf.event_log_specs.timestamp] = pd.to_datetime(self.df[common.conf.event_log_specs.timestamp], format='mixed')
        self.df, _ = common.preprocess(df=self.df)
        self.peers = sample_peers(self.df)
        self.recommendations = make_recommendation(dfs=self.peers, df=self.df)
        
    def __str__(self, n=None):
        result = ''
        common = Common.instance
        if n is None:
            n = len(self.recommendations)
        for i, p in enumerate(self.peers):
           complete, past, future = common.get_original(p[1])
           result += f'Peer no. {i}: \nPast:\n{past[[common.conf.event_log_specs.case_id, common.conf.event_log_specs.activity, common.conf.event_log_specs.timestamp]]}\nFuture:\n{future[[common.conf.event_log_specs.case_id, common.conf.event_log_specs.activity, common.conf.event_log_specs.timestamp]]}\n\n'
        complete, past, future = common.get_original(self.df)
        result += f'Recommendee: \nPast:\n{past[[common.conf.event_log_specs.case_id, common.conf.event_log_specs.activity, common.conf.event_log_specs.timestamp]]}\nFuture:\n{future[[common.conf.event_log_specs.case_id, common.conf.event_log_specs.activity, common.conf.event_log_specs.timestamp]]}\n\n'
        for i, rec in enumerate(self.recommendations[:min(len(self.recommendations), n)]):
            result += f'Recommendation no. {i}:\n{rec}\n'
        return result
    
    def get_best_recommendation(self, interactive: bool = False) -> Recommendation:
        recommendations = self.get_n_recommendations(n=1, interactive=interactive)
        if recommendations:
            return recommendations[0]
        
    def get_all_recommendations(self, interactive: bool = False) -> list[Recommendation]:
        return self.get_n_recommendations(n=len(self.recommendations), interactive=interactive)
    
    def get_n_recommendations(self, n: int, interactive: bool = False) -> list[Recommendation]:
        recommendations = self.recommendations[:min(len(self.recommendations), n)]
        if interactive:
            print(self.__str__(n=n))
            json_output = Recommendation.to_json(recommendations)
            pyperclip.copy(json_output)
            html_content = json2html.convert(json=json_output)
            with open("recommendation.html", "w") as file:
                file.write(html_content)
        return recommendations