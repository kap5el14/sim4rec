from algo.performance import KPIUtils
from common import *
from algo.similarity import similarity_between_events, similarity_between_trace_headers

class RecommendationUtils:
    instance: 'RecommendationUtils' = None
    def __init__(self, common: Common):
        counter1 = Counter(common.train_df[common.conf.event_log_specs.activity])
        self.novelty_perc_values = np.percentile(list(counter1.values()), [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        self.activity_novelties = {k: self._normalize(v) for k, v in counter1.items()}
        performance_dict = KPIUtils.instance.performance_dict
        self.activity_performance_dict = {act: [] for act in common.train_df[common.conf.event_log_specs.activity].unique()}
        sums = {act: 0 for act in common.train_df[common.conf.event_log_specs.activity].unique()}
        for case_id, performance in performance_dict.items():
            df = common.train_df[common.train_df[common.conf.event_log_specs.case_id] == case_id]
            for act in self.activity_performance_dict.keys():
                occurrences = len(df[df[common.conf.event_log_specs.activity] == act])
                if occurrences:
                    self.activity_performance_dict[act].append(float(occurrences * performance[1]))
                    sums[act] += occurrences
        self.activity_performance_dict = {k: np.sum(v) / sums[k] if v and sums[k] else 0.5 for k, v in self.activity_performance_dict.items()}

    def _normalize(self, v) -> float:
        if v <= self.novelty_perc_values[0]:
            return 1.0
        elif v >= self.novelty_perc_values[-1]:
            return 0.0
        for i in range(1, len(self.novelty_perc_values)):
            if v <= self.novelty_perc_values[i]:
                lower_bound = self.novelty_perc_values[i-1]
                upper_bound = self.novelty_perc_values[i]
                if upper_bound == lower_bound:
                    return 1 - (i-1)/10 + 0.05
                else:
                    return 1 - (i-1)/10 + (v - lower_bound) / (upper_bound - lower_bound) * 0.1
        return 0.5
    
    def get_novelty(self, activity):
        if activity not in self.activity_novelties:
            return 0.5
        return self.activity_novelties[activity]
    
    def get_activity_kpi(self, activity):
        if activity not in self.activity_performance_dict:
            return 0.5
        return self.activity_performance_dict[activity]

@dataclass
class RecommendationCandidate:
    row: pd.Series
    timeliness: float
    peer_performance: float
    kpi_dict: dict[str, float]
    temporal_offset: timedelta

    def __post_init__(self):
        self.similar_candidates = []

    def __hash__(self):
        return hash(self.row.name)

    @classmethod
    def generate_candidates(cls, peer_df: pd.DataFrame, df: pd.DataFrame, sim: float):
        common = Common.instance
        complete_peer_df = common.train_df[common.train_df[common.conf.event_log_specs.case_id] == peer_df[common.conf.event_log_specs.case_id].iloc[0]]
        peer_future_df = complete_peer_df.iloc[len(peer_df):]
        peer_future_df = peer_future_df[:min(len(peer_future_df), common.conf.horizon)]
        min_index = peer_future_df[INDEX].min()
        proximities = peer_future_df[INDEX].transform(lambda x: max(0, 1 - 2 * (x - min_index)))
        kpi_dict, fp = KPIUtils.instance.compute_kpi(complete_peer_df)
        candidates = []
        for i in range(len(peer_future_df)):
            is_output_activity = False
            acted_on = False
            row = peer_future_df.iloc[i]
            for j in range(len(df)):
                is_output_activity = row[common.conf.event_log_specs.activity] in common.conf.output_format.activities
                if not is_output_activity:
                    break
                event_sim = similarity_between_events(row, df.iloc[j])
                if event_sim >= sim:
                    acted_on = True
                    break
            if is_output_activity and not acted_on:
                temporal_offset = common.conf.df.loc[row.name][common.conf.event_log_specs.timestamp] - common.conf.df.loc[peer_df.iloc[-1].name][common.conf.event_log_specs.timestamp]
                candidates.append(RecommendationCandidate(row=row, timeliness=proximities.iloc[i], peer_performance=fp, kpi_dict=kpi_dict, temporal_offset=temporal_offset))
        return candidates

@dataclass
class Recommendation:
    cluster: list[RecommendationCandidate]
    all_peers: list[tuple[float, pd.DataFrame]]
    id: int
    last_timestamp: datetime
    supporting_peers: set[str] = field(init=False, default=None)
    event: pd.Series = field(init=False, default=None)
    support: float = field(init=False, default=None)
    peer_kpi: float = field(init=False, default=None)
    kpi_dict: dict[str, float] = field(init=False, default=None)
    timeliness: float = field(init=False, default=None)
    coherence: float = field(init=False, default=None)
    novelty: float = field(init=False, default=None)
    score: float = field(init=False, default=0)
    normalized_event: pd.Series = field(init=False, default=None)

    def __eq__(self, value: 'Recommendation') -> bool:
        return self.id == value.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __str__(self) -> str:
        result = ''
        for k, v in self.__dict__().items():
            if k == 'event':
                result += 'event:\n'
                for k2, v2 in v.items():
                    result += f'\t{k2}: {v2}\n'
            else:
                result += f'{k}: {v}\n'
        result += '\n'
        return result
    
    def __dict__(self) -> dict:
        result = {
            "overall score": round(float(self.score), 2),
            "support among peers": round(float(self.support), 2),
            "supporting peers": list(self.supporting_peers),
            "timeliness": round(float(self.timeliness), 2),
            "agreement on attributes": round(float(self.coherence), 2),
            "activity novelty": round(float(self.novelty), 2),
            "KPI (average performance of supporting peers)": round(float(self.peer_kpi), 2),
            "KPI gain (relative to non-supporting peers)": round(float(self.kpi_gain), 2),
            "activity KPI": round(float(self.activity_kpi), 2),
            "component KPIs:": {k: round(float(v), 2) for k, v in self.kpi_dict.items()},
            "event": {k: str(v) for k, v in self.event.to_dict().items()}
        }
        return result
    
    @staticmethod
    def to_json(recommendations):
        result = []
        for rec in recommendations:
            result.append(rec.__dict__())
        return json.dumps(result)


    def __post_init__(self):
        common = Common.instance
        self.supporting_peers = set()
        timeliness_sum = 0
        peer_performance_sum = 0
        kpi_sums = {}
        values, counts = np.unique([can.row[common.conf.event_log_specs.activity] for can in self.cluster], return_counts=True)
        main_activity = values[np.argmax(counts)]
        self.cluster = [can for can in self.cluster if can.row[common.conf.event_log_specs.activity] == main_activity]
        for candidate in self.cluster:
            case_id = candidate.row[common.conf.event_log_specs.case_id]
            self.supporting_peers.add(case_id)
            timeliness_sum += candidate.timeliness
            peer_performance_sum += candidate.peer_performance
            for k, v in candidate.kpi_dict.items():
                if k in kpi_sums:
                    kpi_sums[k] += v
                else:
                    kpi_sums[k] = v
        self.support = len(self.supporting_peers) / len(self.all_peers)
        self.timeliness = timeliness_sum / len(self.cluster)
        self.peer_kpi = peer_performance_sum / len(self.cluster)
        peer_case_ids = [c.row[common.conf.event_log_specs.case_id] for c in self.cluster]
        non_supporting_peers = [p[1] for p in self.all_peers if p[1][common.conf.event_log_specs.case_id].iloc[0] not in peer_case_ids]
        non_supporting_kpi = np.mean([KPIUtils.instance.compute_kpi(p)[1] for p in non_supporting_peers]) if len(non_supporting_peers) else self.peer_kpi
        self.kpi_gain = 0.5 + (self.peer_kpi - non_supporting_kpi) / 2
        self.activity_kpi = RecommendationUtils.instance.get_activity_kpi(main_activity)
        self.kpi_dict = {k: v / len(self.cluster) for k, v in kpi_sums.items()}
        candidates_df = pd.DataFrame([common.conf.df.loc[c.row.name] for c in self.cluster])
        numerical_cols = candidates_df[common.conf.output_format.numerical_attributes]
        categorical_cols = candidates_df[common.conf.output_format.categorical_attributes]
        temporal_cols = candidates_df[common.conf.output_format.timestamp_attributes]
        attributes = []
        if not numerical_cols.empty:
            attributes.append(numerical_cols.median())
        if not categorical_cols.empty:
            attributes.append(categorical_cols.mode().iloc[0])
        if not temporal_cols.empty:
            if pd.api.types.is_datetime64tz_dtype(common.conf.df[common.conf.event_log_specs.timestamp]):
                unix_epoch = pd.Timestamp("1970-01-01", tz='UTC')
            else:
                unix_epoch = pd.Timestamp("1970-01-01")
            attributes.append(temporal_cols.apply(lambda x: x.median()).apply(lambda x: unix_epoch + pd.to_timedelta(x, unit='s')))
        self.event = pd.concat(attributes).dropna()
        self.normalized_event = common.normalize(pd.DataFrame([self.event])).iloc[0]
        self.event[common.conf.event_log_specs.timestamp] = self.last_timestamp + np.median([c.temporal_offset for c in self.cluster])
        sims = []
        for i, can1 in enumerate(self.cluster):
            for j, can2 in enumerate(self.cluster):
                if j > i:
                    sims.append(similarity_between_events(can1.row, can2.row))
        self.coherence = np.mean(sims) if len(sims) else 1
        self.novelty = RecommendationUtils.instance.get_novelty(self.event[common.conf.event_log_specs.activity])
        #self.score = max(0, min(1, np.power(self.kpi * self.support * self.timeliness * self.coherence, 0.25)))
        #self.score = self.timeliness
        self.score = self.timeliness * common.conf.optimization_goals.timeliness + np.power(self.peer_kpi * self.kpi_gain * self.activity_kpi, 1/3) * common.conf.optimization_goals.performance + self.support * common.conf.optimization_goals.support + self.novelty * common.conf.optimization_goals.novelty + self.coherence * common.conf.optimization_goals.coherence
        

    @classmethod
    def generate_recommendations(cls, candidates: list[RecommendationCandidate], dfs, last_timestamp) -> list['Recommendation']:
        common = Common.instance
        n = len(candidates)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if i != j:
                    distance_matrix[j, i] = distance_matrix[i, j] = 1 - similarity_between_events(candidates[i].row, candidates[j].row)
        clusters = {can.row[common.conf.event_log_specs.activity]: set() for can in candidates}
        for can in candidates:
            clusters[can.row[common.conf.event_log_specs.activity]].add(can)
        recommendations = [Recommendation(cluster=cluster, all_peers=dfs, id=rec_index, last_timestamp=last_timestamp) for rec_index, cluster in enumerate(clusters.values())]
        recommendations = [r for r in recommendations if r.score]
        return list(sorted(recommendations, key=lambda rec: rec.score, reverse=True))

def make_recommendation(dfs: list[tuple[float, pd.DataFrame]], df: pd.DataFrame, last_timestamp) -> list[Recommendation]:
    n = len(dfs)
    candidates_map = {}
    average_sim = np.mean([sim for sim, df in dfs])
    for i in range(n):
        df1 = dfs[i][1]
        candidates_map[i] = RecommendationCandidate.generate_candidates(peer_df=df1, df=df, sim=average_sim)
    candidates = []
    for _, vals in candidates_map.items():
        candidates += vals
    if candidates:
        recommendations = Recommendation.generate_recommendations(candidates=candidates, dfs=dfs, last_timestamp=last_timestamp)
        if recommendations:
            return recommendations
    return []
