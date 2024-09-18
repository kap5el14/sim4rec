from util.common import *
from algo.similarity import similarity_between_events, similarity_between_trace_headers
from data.success import is_successful


def compute_kpi(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    common = Common.instance
    if not is_successful(df):
        return {}, 0
    kpi_dict = {}
    kpi_weights = {}
    if common.conf.performance_weights.trace_length:
        kpi_dict[TRACE_LENGTH] = 1 - df[TRACE_LENGTH].iloc[-1]
        kpi_weights[TRACE_LENGTH] = common.conf.performance_weights.trace_length
    if common.conf.performance_weights.trace_duration:
        kpi_dict[TRACE_DURATION] = 1 - df[TRACE_DURATION].iloc[-1]
        kpi_weights[TRACE_DURATION] = common.conf.performance_weights.trace_duration
    for k, v in common.conf.performance_weights.numerical_trace_attributes.items():
        if 'min' in v:
            kpi_dict[k] = 1 - df[k].iloc[-1]
        elif 'max' in v:
            kpi_dict[k] = df[k].iloc[-1]
        else:
            raise ValueError
        kpi_weights[k] = v[1]
    for k, v in common.conf.performance_weights.categorical_trace_attributes.items():
        value = v[0]
        name = f"{k}=={value}"
        kpi_dict[name] = 1 if df[k] == value else 0
        kpi_weights[name] = v[1]
    for k, v in common.conf.performance_weights.numerical_event_attributes.items():
        if 'sum' in v:
            t = CUMSUM
        elif 'avg' in v:
            t = CUMAVG
        else:
            raise ValueError
        name = f"{k}{t}"
        if 'min' in v:
            kpi_dict[name] = 1 - df[name].iloc[-1]
        elif 'max' in v:
            kpi_dict[name] = df[name].iloc[-1]
        else:
            raise ValueError
        kpi_weights[name] = v[2]
    future_performance = sum([kpi_dict[k] * kpi_weights[k] for k in kpi_dict.keys()])
    return kpi_dict, future_performance  


@dataclass
class RecommendationCandidate:
    row: pd.Series
    proximity: float
    similarity: float
    future_performance: float
    kpi_dict: dict[str, float]
    similar_candidates: list['RecommendationCandidate'] = field(init=False, default=None)
    marked: bool = field(init=False, default=False)

    def __post_init__(self):
        self.similar_candidates = []

    def __hash__(self):
        return hash(self.row.name)

    @classmethod
    def generate_candidates(cls, peer_df: pd.DataFrame, df: pd.DataFrame, sim: float):
        common = Common.instance
        complete_peer_df = common.train_df[common.train_df[common.conf.event_log_specs.case_id] == peer_df[common.conf.event_log_specs.case_id].iloc[0]]
        last_values = df[[TIME_FROM_TRACE_START, INDEX]].iloc[-1]
        complete_peer_df_filtered = complete_peer_df[[TIME_FROM_TRACE_START, INDEX]]
        difference_df = complete_peer_df_filtered - last_values
        proximities = 1 - difference_df.abs().mean(axis=1)
        kpi_dict, fp = compute_kpi(complete_peer_df)
        candidates = []
        for i in range(len(complete_peer_df)):
            acted_on = False
            for j in range(len(df)):
                row = complete_peer_df.iloc[i]
                event_sim = similarity_between_events(row, df.iloc[j])
                if event_sim >= sim:
                    acted_on = True
                    break
            if not acted_on:
                candidates.append(RecommendationCandidate(row=row, proximity=proximities.iloc[i], similarity=sim, future_performance=fp, kpi_dict=kpi_dict))
        return candidates

    @classmethod
    def connect_candidates(cls, candidates1: list['RecommendationCandidate'], candidates2: list['RecommendationCandidate'], sim):
        for c1 in candidates1:
            for c2 in candidates2:
                event_sim = similarity_between_events(c1.row, c2.row)
                if event_sim >= sim:
                    c1.similar_candidates.append(c2)
                    c2.similar_candidates.append(c1)

@dataclass
class Recommendation:
    cluster: list[RecommendationCandidate] = field(init=True, default=None)
    all_peers: list[str] = field(init=True, default=None)
    id: int = field(init=True, default=None)
    peers: set[str] = field(init=False, default=None)
    event: pd.Series = field(init=False, default=None)
    support: float = field(init=False, default=None)
    future_performance: float = field(init=False, default=None)
    kpi_dict: dict[str, float] = field(init=False, default=None)
    proximity: float = field(init=False, default=None)
    similarity: float = field(init=False, default=None)
    marked: bool = field(init=False, default=False)

    def __eq__(self, value: 'Recommendation') -> bool:
        return self.id == value.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __str__(self) -> str:
        return f"peers: {self.peers}\nsupport: {self.support}\nsimilarity: {self.similarity}\nproximity: {self.proximity}\nKPI: {self.future_performance}\n\n{self.event}\n"

    def __post_init__(self):
        common = Common.instance
        self.peers = set()
        similarity_sum = 0
        proximity_sum = 0
        future_performance_sum = 0
        kpi_sums = {}
        for candidate in self.cluster:
            case_id = candidate.row[common.conf.event_log_specs.case_id]
            self.peers.add(case_id)
            similarity_sum += candidate.similarity
            proximity_sum += candidate.proximity
            future_performance_sum += candidate.future_performance
            for k, v in candidate.kpi_dict.items():
                if k in kpi_sums:
                    kpi_sums[k] += v
                else:
                    kpi_sums[k] = v
        self.support = len(self.peers) / len(self.all_peers)
        self.similarity = similarity_sum / len(self.cluster)
        self.proximity = proximity_sum / len(self.cluster)
        self.future_performance = future_performance_sum / len(self.cluster)
        self.kpi_dict = {k: v / len(self.cluster) for k, v in kpi_sums.items()}
        candidates_df = pd.DataFrame([common.conf.df.loc[c.row.name] for c in self.cluster])
        numerical_cols = candidates_df[common.conf.output_format.numerical_attributes]
        categorical_cols = candidates_df[common.conf.output_format.categorical_attributes]
        timestamp_cols = candidates_df[common.conf.output_format.timestamp_attributes]
        numerical_avg = numerical_cols.mean()
        categorical_mode = categorical_cols.mode().iloc[0]
        timestamp_avg = timestamp_cols.apply(lambda x: x.mean())
        self.event = pd.concat([numerical_avg, categorical_mode, timestamp_avg])
    
    def score(self):
        return self.future_performance + (self.support + self.similarity + self.proximity) / 3
    def dominates(self, rec: 'Recommendation'):
        not_worse = self.support >= rec.support and self.future_performance >= rec.future_performance and self.proximity >= rec.proximity and self.similarity >= rec.similarity
        better = self.support > rec.support or self.future_performance > rec.future_performance or self.proximity > rec.proximity or self.similarity > rec.similarity
        return not_worse and better

    @classmethod
    def generate_recommendations(cls, candidates: list[RecommendationCandidate]) -> list['Recommendation']:
        common = Common.instance
        clusters: list[set[RecommendationCandidate]] = []
        all_peers = set()
        for c in candidates:
            if c.marked:
                continue
            all_peers.add(c.row[common.conf.event_log_specs.case_id])
            cluster = set([c])
            to_visit = set(c.similar_candidates)
            while to_visit:
                current = to_visit.pop()
                cluster.add(current)
                current.marked = True
                for n in current.similar_candidates:
                    if not n.marked:
                        to_visit.add(n)
            clusters.append(cluster)
        recommendations: list[Recommendation] = []
        rec_index = 0
        for cluster in clusters:
            recommendations.append(Recommendation(cluster=cluster, all_peers=all_peers, id=rec_index))
            rec_index +=1
        pareto: set[Recommendation] = set()
        for rec1 in recommendations:
            dominated = set()
            flag = False
            for rec2 in pareto:
                if rec1.dominates(rec2):
                    dominated.add(rec2)
                elif rec2.dominates(rec1):
                    dominated.add(rec1)
                    flag = True
                    break
            pareto = pareto.difference(dominated)
            if not flag:
                pareto.add(rec1)
        return list(pareto)
    
    @classmethod
    def generate_scenarios(cls, recommendations: list['Recommendation']) -> list[list['Recommendation']]:
        n = len(recommendations)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    set1 = recommendations[i].peers
                    set2 = recommendations[j].peers
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    distance_matrix[i, j] = 1 - (intersection / union if union != 0 else 0)
        clustering = AgglomerativeClustering(linkage='average')
        cluster_labels = clustering.fit_predict(distance_matrix)
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(recommendations[idx])
        scenarios = list(clusters.values())
        sorted_scenarios = list(sorted(scenarios, key=lambda scenario: np.mean([rec.score() for rec in scenario]), reverse=True))
        return [list(sorted(scenario, key=lambda x: x.score(), reverse=True)) for scenario in sorted_scenarios]

def recommend_scenarios(dfs: list[tuple[float, pd.DataFrame]], df: pd.DataFrame, sample_size=3) -> list[Recommendation]:
    n = len(dfs)
    candidates_map = {}
    for i in range(n):
        df1 = dfs[i][1]
        candidates_map[i] = RecommendationCandidate.generate_candidates(peer_df=df1, df=df, sim=dfs[i][0])
    for i in range(n):
        df1 = dfs[i][1]
        candidates1 = candidates_map[i]
        for j in range(i + 1, n):
            df2 = dfs[j][1]
            candidates2 = candidates_map[j]
            RecommendationCandidate.connect_candidates(candidates1=candidates1, candidates2=candidates2, sim=similarity_between_trace_headers(df1, df2))
    candidates = []
    for _, vals in candidates_map.items():
        candidates += vals
    scenarios = Recommendation.generate_scenarios(recommendations=Recommendation.generate_recommendations(candidates=candidates))
    return scenarios[:min(sample_size, len(scenarios))]

