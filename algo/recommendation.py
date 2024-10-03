from algo.performance import KPIUtils
from common import *
from algo.similarity import similarity_between_events, similarity_between_trace_headers

class RecommendationUtils:
    instance: 'RecommendationUtils' = None
    def __init__(self, common: Common):
        counter = Counter(common.train_df[common.conf.event_log_specs.activity])
        self.perc_values = np.percentile([v for v in counter.values()], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        self.activity_rarenesss = {k: self._normalize_rareness(v) for k, v in counter.items()}

    def _normalize_rareness(self, count: int) -> float:
        if count <= self.perc_values[0]:
            return 1.0
        elif count >= self.perc_values[-1]:
            return 0.0
        for i in range(1, len(self.perc_values)):
            if count <= self.perc_values[i]:
                lower_bound = self.perc_values[i-1]
                upper_bound = self.perc_values[i]
                if upper_bound == lower_bound:
                    return 1 - ((i-1)/10 + 0.05)
                else:
                    return 1 - ((i-1)/10 + (count - lower_bound) / (upper_bound - lower_bound) * 0.1)
        return 1 - count
    
    def get_rareness(self, activity):
        if activity not in self.activity_rarenesss:
            return 0.5
        return self.activity_rarenesss[activity]
    

@dataclass
class RecommendationCandidate:
    row: pd.Series
    urgency: float
    peer_performance: float
    kpi_dict: dict[str, float]

    def __post_init__(self):
        self.similar_candidates = []

    def __hash__(self):
        return hash(self.row.name)

    @classmethod
    def generate_candidates(cls, peer_df: pd.DataFrame, df: pd.DataFrame, sim: float):
        common = Common.instance
        complete_peer_df = common.train_df[common.train_df[common.conf.event_log_specs.case_id] == peer_df[common.conf.event_log_specs.case_id].iloc[0]]
        peer_future_df = complete_peer_df.iloc[len(peer_df):]
        last_values = df[[TIME_FROM_TRACE_START, INDEX]].iloc[-1]
        peer_future_df_filtered = peer_future_df[[TIME_FROM_TRACE_START, INDEX]]
        difference_df = peer_future_df_filtered - last_values
        proximities = 1 - difference_df.abs().mean(axis=1)
        kpi_dict, fp = KPIUtils.instance.compute_kpi(complete_peer_df)
        candidates = []
        for i in range(len(peer_future_df)):
            is_output_activity = False
            acted_on = False
            for j in range(len(df)):
                row = peer_future_df.iloc[i]
                is_output_activity = row[common.conf.event_log_specs.activity] in common.conf.output_format.activities
                if not is_output_activity:
                    break
                event_sim = similarity_between_events(row, df.iloc[j])
                if event_sim >= sim:
                    acted_on = True
                    break
            if is_output_activity and not acted_on:
                candidates.append(RecommendationCandidate(row=row, urgency=proximities.iloc[i], peer_performance=fp, kpi_dict=kpi_dict))
        return candidates

@dataclass
class Recommendation:
    cluster: list[RecommendationCandidate] = field(init=True, default=None)
    all_peers: list[str] = field(init=True, default=None)
    id: int = field(init=True, default=None)
    peers: set[str] = field(init=False, default=None)
    event: pd.Series = field(init=False, default=None)
    support: float = field(init=False, default=None)
    kpi: float = field(init=False, default=None)
    kpi_dict: dict[str, float] = field(init=False, default=None)
    urgency: float = field(init=False, default=None)
    coherence: float = field(init=False, default=None)
    rareness: float = field(init=False, default=None)
    score: float = field(init=False, default=None)
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
            "supporting peers": list(self.peers),
            "urgency": round(float(self.urgency), 2),
            "agreement on attributes": round(float(self.coherence), 2),
            "activity rareness": round(float(self.rareness), 2),
            "KPI (average performance of supporting peers)": round(float(self.kpi), 2),
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
        self.peers = set()
        urgency_sum = 0
        peer_performance_sum = 0
        kpi_sums = {}
        values, counts = np.unique([can.row[common.conf.event_log_specs.activity] for can in self.cluster], return_counts=True)
        main_activity = values[np.argmax(counts)]
        self.cluster = [can for can in self.cluster if can.row[common.conf.event_log_specs.activity] == main_activity]
        for candidate in self.cluster:
            case_id = candidate.row[common.conf.event_log_specs.case_id]
            self.peers.add(case_id)
            urgency_sum += candidate.urgency
            peer_performance_sum += candidate.peer_performance
            for k, v in candidate.kpi_dict.items():
                if k in kpi_sums:
                    kpi_sums[k] += v
                else:
                    kpi_sums[k] = v
        self.support = len(self.peers) / len(self.all_peers)
        self.urgency = urgency_sum / len(self.cluster)
        self.kpi = peer_performance_sum / len(self.cluster)
        self.kpi_dict = {k: v / len(self.cluster) for k, v in kpi_sums.items()}
        candidates_df = pd.DataFrame([common.conf.df.loc[c.row.name] for c in self.cluster])
        numerical_cols = candidates_df[common.conf.output_format.numerical_attributes]
        categorical_cols = candidates_df[common.conf.output_format.categorical_attributes]
        timestamp_cols = candidates_df[common.conf.output_format.timestamp_attributes]
        attributes = []
        if not numerical_cols.empty:
            attributes.append(numerical_cols.median())
        if not categorical_cols.empty:
            attributes.append(categorical_cols.mode().iloc[0])
        if not timestamp_cols.empty:
            if pd.api.types.is_datetime64tz_dtype(common.conf.df[common.conf.event_log_specs.timestamp]):
                unix_epoch = pd.Timestamp("1970-01-01", tz='UTC')
            else:
                unix_epoch = pd.Timestamp("1970-01-01")
            attributes.append(timestamp_cols.apply(lambda x: x.median()).apply(lambda x: unix_epoch + pd.to_timedelta(x, unit='s')))
        self.event = pd.concat(attributes).dropna()
        self.normalized_event = common.normalize(pd.DataFrame([self.event])).iloc[0]
        sims = []
        for i, can1 in enumerate(self.cluster):
            for j, can2 in enumerate(self.cluster):
                if j > i:
                    sims.append(similarity_between_events(can1.row, can2.row, just_attributes=True))
        self.coherence = np.mean(sims) if len(sims) else 1
        self.rareness = RecommendationUtils.instance.get_rareness(self.event[common.conf.event_log_specs.activity])
        #self.score = max(0, min(1, np.power(self.kpi * self.support * self.urgency * self.coherence, 0.25)))
        #self.score = self.urgency
        self.score = np.power(self.urgency * self.kpi * self.support * self.rareness * self.coherence, 0.2)

    @classmethod
    def generate_recommendations(cls, candidates: list[RecommendationCandidate]) -> list['Recommendation']:
        common = Common.instance
        all_peers = set()
        n = len(candidates)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            all_peers.add(candidates[i].row[common.conf.event_log_specs.case_id])
            for j in range(i + 1, n):
                if i != j:
                    distance_matrix[j, i] = distance_matrix[i, j] = 1 - similarity_between_events(candidates[i].row, candidates[j].row)
        clusters = {can.row[common.conf.event_log_specs.activity]: set() for can in candidates}
        for can in candidates:
            clusters[can.row[common.conf.event_log_specs.activity]].add(can)
        recommendations = [Recommendation(cluster=cluster, all_peers=all_peers, id=rec_index) for rec_index, cluster in enumerate(clusters.values())]
        return list(sorted(recommendations, key=lambda rec: rec.score, reverse=True))

def make_recommendation(dfs: list[tuple[float, pd.DataFrame]], df: pd.DataFrame) -> list[Recommendation]:
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
        recommendations = Recommendation.generate_recommendations(candidates=candidates)
        if recommendations:
            return recommendations
    return []
