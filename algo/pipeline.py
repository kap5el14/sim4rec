from common import *
from algo.sampling import sample_peers
from algo.recommendation import make_recommendation, Recommendation
from algo.performance import KPIUtils
import pyperclip
from json2html import *

class Pipeline:
    def __init__(self, df: pd.DataFrame):
        common = Common.instance
        self.df = df
        self.df.columns = self.df.columns.str.strip()
        self.df[common.conf.event_log_specs.timestamp] = pd.to_datetime(self.df[common.conf.event_log_specs.timestamp], format='mixed')
        last_timestamp = pd.to_datetime(self.df[common.conf.event_log_specs.timestamp], unit='s').iloc[-1]
        self.df, _ = common.preprocess(df=self.df)
        self.peers = sample_peers(self.df)
        self.recommendations = make_recommendation(dfs=self.peers, df=self.df, last_timestamp=last_timestamp)
        
    def __str__(self, n=None, interactive=False, very_interactive=False):
        result = ''
        common = Common.instance
        if n is None:
            n = len(self.recommendations)
        if very_interactive:
            dirs=['peers', 'peers', 'recommendee']
            for dir in dirs:
                for path in glob.glob(os.path.join(dir, '*.csv')):
                    os.remove(path)   
            all_refined_attributes = common.conf.get_all_normalized_attributes()
            all_original_attributes = common.conf.get_all_original_attributes()
        
        for i, p in enumerate(self.peers):
           complete_original, past_original, future_original = common.get_original(p[1])
           complete_refined, past_refined, future_refined = common.get_normalized(p[1])
           if very_interactive:
                past_refined[all_refined_attributes].to_csv(f'peers/refined_peer_{past_refined[common.conf.event_log_specs.case_id].iloc[0]}_past.csv')
                future_refined[all_refined_attributes].to_csv(f'peers/refined_peer_{past_refined[common.conf.event_log_specs.case_id].iloc[0]}_future.csv')
                past_original[all_original_attributes].to_csv(f'peers/original_peer_{past_original[common.conf.event_log_specs.case_id].iloc[0]}_past.csv', date_format='%Y-%m-%d %H:%M:%S', index=False)
                future_original[all_original_attributes].to_csv(f'peers/original_peer_{past_original[common.conf.event_log_specs.case_id].iloc[0]}_future.csv', date_format='%Y-%m-%d %H:%M:%S', index=False)
                
           result += f'Peer no. {i}: \nPast:\n{past_original[[common.conf.event_log_specs.case_id, common.conf.event_log_specs.activity, common.conf.event_log_specs.timestamp]]}\nFuture:\n{future_original[[common.conf.event_log_specs.case_id, common.conf.event_log_specs.activity, common.conf.event_log_specs.timestamp]]}\nperformance: {KPIUtils.instance.compute_kpi(complete_original)}\n'
        complete_original, past_original, future_original = common.get_original(self.df)
        result += f'Recommendee: \nPast:\n{past_original[[common.conf.event_log_specs.case_id, common.conf.event_log_specs.activity, common.conf.event_log_specs.timestamp]]}\nFuture:\n{future_original[[common.conf.event_log_specs.case_id, common.conf.event_log_specs.activity, common.conf.event_log_specs.timestamp]]}\n\n'
        if very_interactive:
            self.df[all_refined_attributes].to_csv(f'recommendee/refined_recommendee_{self.df[common.conf.event_log_specs.case_id].iloc[0]}.csv')
            past_original[all_original_attributes].to_csv(f'recommendee/original_recommendee_{self.df[common.conf.event_log_specs.case_id].iloc[0]}.csv', date_format='%Y-%m-%d %H:%M:%S', index=False)
        for i, rec in enumerate(self.recommendations[:min(len(self.recommendations), n)]):
            result += f'Recommendation no. {i}:\n{rec}\n\n'
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
            print(self.__str__(n=n, interactive=True))
            json_output = Recommendation.to_json(recommendations)
            #pyperclip.copy(json_output)
            html_content = json2html.convert(json=json_output)
            with open("recommendation.html", "w") as file:
                file.write(html_content)
            #input()
        return recommendations