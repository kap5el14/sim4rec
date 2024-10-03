from algo.performance import compute_kpi
from util.common import *
from algo.similarity import similarity_between_events, similarity_between_trace_headers


@dataclass
class RecommendationCandidate:
    row: pd.Series
    proximity: float
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
        kpi_dict, fp = compute_kpi(complete_peer_df)
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
                candidates.append(RecommendationCandidate(row=row, proximity=proximities.iloc[i], peer_performance=fp, kpi_dict=kpi_dict))
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
    proximity: float = field(init=False, default=None)
    coherence: float = field(init=False, default=None)
    score: float = field(init=False, default=None)
    normalized_event: pd.Series = field(init=False, default=None)

    def __eq__(self, value: 'Recommendation') -> bool:
        return self.id == value.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __str__(self) -> str:
        return f"score: {self.score}, peers: {self.peers}\nsupport: {self.support}\nproximity: {self.proximity} coherence: {self.coherence}\nKPI: {self.kpi}\nKPI_dict: {self.kpi_dict}\n\n{self.event}\n\n"
    
    def __dict__(self) -> dict:
        result = {
            "score": self.score,
            "peers": list(self.peers),
            "support": self.support,
            "proximity": self.proximity,
            "coherence": self.coherence,
            "KPI": self.kpi,
            "component KPIs:": self.kpi_dict,
            "event": {k: str(v) for k, v in self.event.to_dict().items()}
        }
        return result
    
    def to_json(self):
        return json.dumps(self.__dict__())


    def __post_init__(self):
        common = Common.instance
        self.peers = set()
        proximity_sum = 0
        peer_performance_sum = 0
        kpi_sums = {}
        values, counts = np.unique([can.row[common.conf.event_log_specs.activity] for can in self.cluster], return_counts=True)
        main_activity = values[np.argmax(counts)]
        self.cluster = [can for can in self.cluster if can.row[common.conf.event_log_specs.activity] == main_activity]
        for candidate in self.cluster:
            case_id = candidate.row[common.conf.event_log_specs.case_id]
            self.peers.add(case_id)
            proximity_sum += candidate.proximity
            peer_performance_sum += candidate.peer_performance
            for k, v in candidate.kpi_dict.items():
                if k in kpi_sums:
                    kpi_sums[k] += v
                else:
                    kpi_sums[k] = v
        self.support = len(self.peers) / len(self.all_peers)
        self.proximity = proximity_sum / len(self.cluster)
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
                    sims.append(similarity_between_events(can1.row, can2.row))
        self.coherence = np.mean(sims) if len(sims) else 1
        #self.score = max(0, min(1, np.power(self.kpi * self.support * self.proximity * self.coherence, 0.25)))
        #self.score = self.proximity
        self.score = 0.5 * self.proximity + 0.25 * self.kpi + 0.20 * self.support + 0.05 * self.coherence

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

def make_recommendation(dfs: list[tuple[float, pd.DataFrame]], df: pd.DataFrame, all=False) -> Recommendation:
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
            if all:
                return recommendations
            return recommendations[0]
