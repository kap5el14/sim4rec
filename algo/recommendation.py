from util.common import *
from algo.similarity import similarity_between_events, similarity_between_trace_headers


def compute_kpi(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    common = Common.instance
    kpi_dict = {}
    kpi_weights = {}
    normalized_last_row = common.future_train_df[common.future_train_df[common.conf.event_log_specs.case_id] == df[common.conf.event_log_specs.case_id].iloc[0]].iloc[0]
    if common.conf.custom_performance_function:
        original_df = common.conf.df.loc[df.index]
        custom_performance = common.conf.custom_performance_function(original_df, df, normalized_last_row)
        kpi_dict['custom'] = custom_performance
        return kpi_dict, custom_performance
    if common.conf.performance_weights.trace_length:
        kpi_dict[TRACE_LENGTH] = 1 - normalized_last_row[TRACE_LENGTH]
        kpi_weights[TRACE_LENGTH] = common.conf.performance_weights.trace_length
    if common.conf.performance_weights.trace_duration:
        kpi_dict[TRACE_DURATION] = 1 - normalized_last_row[TRACE_DURATION]
        kpi_weights[TRACE_DURATION] = common.conf.performance_weights.trace_duration
    for k, v in common.conf.performance_weights.numerical_trace_attributes.items():
        if 'min' in v:
            kpi_dict[k] = 1 - normalized_last_row[k]
        elif 'max' in v:
            kpi_dict[k] = normalized_last_row[k]
        else:
            raise ValueError
        kpi_weights[k] = v[1]
    for k, v in common.conf.performance_weights.categorical_trace_attributes.items():
        value = v[0]
        name = f"{k}=={value}"
        kpi_dict[name] = 1 if normalized_last_row[k] == value else 0
        kpi_weights[name] = v[1]
    for k, v in common.conf.performance_weights.numerical_event_attributes.items():
        for v2 in v:
            if 'sum' in v2:
                t = CUMSUM
            elif 'avg' in v2:
                t = CUMAVG
            else:
                raise ValueError
            name = f"{k}{t}"
            if 'min' in v2:
                kpi_dict[name] = 1 - normalized_last_row[name]
            elif 'max' in v2:
                kpi_dict[name] = normalized_last_row[name]
            else:
                raise ValueError
            kpi_weights[name] = v2[2]
    performance = sum([kpi_dict[k] * kpi_weights[k] for k in kpi_dict.keys()])
    return kpi_dict, performance  


@dataclass
class RecommendationCandidate:
    row: pd.Series
    proximity: float
    similarity: float
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
        peer_future_df = complete_peer_df[len(peer_df):]
        last_values = df[[TIME_FROM_TRACE_START, INDEX]].iloc[-1]
        peer_future_df_filtered = peer_future_df[[TIME_FROM_TRACE_START, INDEX]]
        difference_df = peer_future_df_filtered - last_values
        proximities = 1 - difference_df.abs().mean(axis=1)
        kpi_dict, fp = compute_kpi(complete_peer_df)
        candidates = []
        for i in range(len(peer_future_df)):
            acted_on = False
            for j in range(len(df)):
                row = peer_future_df.iloc[i]
                event_sim = similarity_between_events(row, df.iloc[j])
                if event_sim >= sim:
                    acted_on = True
                    break
            if not acted_on:
                candidates.append(RecommendationCandidate(row=row, proximity=proximities.iloc[i], similarity=sim, peer_performance=fp, kpi_dict=kpi_dict))
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
    similarity: float = field(init=False, default=None)
    normalized_event: pd.Series = field(init=False, default=None)

    def __eq__(self, value: 'Recommendation') -> bool:
        return self.id == value.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __str__(self) -> str:
        return f"peers: {self.peers}\nsupport: {self.support}\nsimilarity: {self.similarity}\nproximity: {self.proximity}\nKPI: {self.kpi}\nKPI_dict: {self.kpi_dict}\n\n{self.event}\n\n"
    
    def __dict__(self) -> dict:
        result = {
            "peers": list(self.peers),
            "support": self.support,
            "similarity": self.similarity,
            "proximity": self.proximity,
            "KPI": self.kpi,
            "component KPIs:": self.kpi_dict,
            "event": {k: str(v) for k, v in self.event.to_dict().items()}
        }
        return result


    def __post_init__(self):
        common = Common.instance
        self.peers = set()
        similarity_sum = 0
        proximity_sum = 0
        peer_performance_sum = 0
        kpi_sums = {}
        for candidate in self.cluster:
            case_id = candidate.row[common.conf.event_log_specs.case_id]
            self.peers.add(case_id)
            similarity_sum += candidate.similarity
            proximity_sum += candidate.proximity
            peer_performance_sum += candidate.peer_performance
            for k, v in candidate.kpi_dict.items():
                if k in kpi_sums:
                    kpi_sums[k] += v
                else:
                    kpi_sums[k] = v
        self.support = len(self.peers) / len(self.all_peers)
        self.similarity = similarity_sum / len(self.cluster)
        self.proximity = proximity_sum / len(self.cluster)
        self.kpi = peer_performance_sum / len(self.cluster)
        self.kpi_dict = {k: v / len(self.cluster) for k, v in kpi_sums.items()}
        candidates_df = pd.DataFrame([common.conf.df.loc[c.row.name] for c in self.cluster])
        numerical_cols = candidates_df[common.conf.output_format.numerical_attributes]
        categorical_cols = candidates_df[common.conf.output_format.categorical_attributes]
        timestamp_cols = candidates_df[common.conf.output_format.timestamp_attributes]
        attributes = []
        if not numerical_cols.empty:
            attributes.append(numerical_cols.mean())
        if not categorical_cols.empty:
            attributes.append(categorical_cols.mode().iloc[0])
        if not timestamp_cols.empty:
            if pd.api.types.is_datetime64tz_dtype(common.conf.df[common.conf.event_log_specs.timestamp]):
                unix_epoch = pd.Timestamp("1970-01-01", tz='UTC')
            else:
                unix_epoch = pd.Timestamp("1970-01-01")
            attributes.append(timestamp_cols.apply(lambda x: x.mean()).apply(lambda x: unix_epoch + pd.to_timedelta(x, unit='s')))
        self.event = pd.concat(attributes).dropna()
        attributes = []
        candidates_df = pd.DataFrame([common.train_df.loc[c.row.name] for c in self.cluster])
        numerical_cols = candidates_df[common.conf.performance_weights.get_all_numerical_attributes()]
        categorical_cols = candidates_df[list(common.conf.performance_weights.categorical_trace_attributes.keys())]
        if not numerical_cols.empty:
            attributes.append(numerical_cols.mean())
        if not categorical_cols.empty:
            attributes.append(categorical_cols.mode().iloc[0])
        self.normalized_event = pd.concat(attributes).dropna()
    
    def score(self):
        return self.kpi * self.support
    
    def dominates(self, rec: 'Recommendation'):
        not_worse = self.support >= rec.support and self.kpi >= rec.kpi and self.proximity >= rec.proximity and self.similarity >= rec.similarity
        better = self.support > rec.support or self.kpi > rec.kpi or self.proximity > rec.proximity or self.similarity > rec.similarity
        return not_worse and better

    @classmethod
    def generate_recommendations(cls, candidates: list[RecommendationCandidate]) -> list['Recommendation']:
        common = Common.instance
        clusters: list[set[RecommendationCandidate]] = []
        all_peers = set()
        n = len(candidates)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            all_peers.add(candidates[i].row[common.conf.event_log_specs.case_id])
            for j in range(i + 1, n):
                if i != j:
                    distance_matrix[j, i] = distance_matrix[i, j] = 1 - similarity_between_events(candidates[i].row, candidates[j].row)
        clustering = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        clusters = {}
        no_labels = len(set(cluster_labels))
        for idx, label in enumerate(cluster_labels):
            if label == -1:
                clusters[no_labels] = [candidates[idx]]
                no_labels += 1
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(candidates[idx])
        recommendations: list[Recommendation] = []
        recommendations = [Recommendation(cluster=cluster, all_peers=all_peers, id=rec_index) for rec_index, cluster in enumerate(clusters.values())]
        recommendations = list(filter(lambda rec: rec.event[common.conf.event_log_specs.activity] in common.conf.output_format.activities, recommendations))
        means = []
        for rec in recommendations:
            sims = []
            for i, can1 in enumerate(rec.cluster):
                for j, can2 in enumerate(rec.cluster):
                    if j > i:
                        sims.append(similarity_between_events(can1.row, can2.row))
            means.append(np.mean(sims))
        median = np.median([m for m in means if not np.isnan(m)])
        mask = [np.isnan(mean) or mean >= median for mean in means]
        recommendations = [rec for rec, m in zip(recommendations, mask) if m]
        pareto: set[Recommendation] = set()
        for rec1 in recommendations:
            dominated = set()
            flag = False
            for rec2 in pareto:
                if rec1.dominates(rec2):
                    dominated.add(rec2)
                elif rec2.dominates(rec1):
                    flag = True
                    break
            pareto = pareto.difference(dominated)
            if not flag:
                pareto.add(rec1)
        return list(pareto)
    
@dataclass
class RecommendationPackage:
    highest_kpi_rec: Recommendation = field(init=False, default=None)
    highest_support_rec: Recommendation = field(init=False, default=None)
    optimal_rec: Recommendation = field(init=False, default=None)

    def __init__(self, recommendations: list[Recommendation]):
        max_kpi = max([rec.kpi for rec in recommendations])
        self.highest_kpi_rec = next((rec for rec in recommendations if rec.kpi == max_kpi), None)
        max_support = max([rec.support for rec in recommendations])
        self.highest_support_rec = next((rec for rec in recommendations if rec.support == max_support), None)
        max_score = max([rec.score() for rec in recommendations])
        self.optimal_rec = next((rec for rec in recommendations if rec.score() == max_score), None)

    def __dict__(self):
        d = {
            self.highest_kpi_rec: [],
            self.highest_support_rec: [],
            self.optimal_rec: []
        }
        d[self.highest_kpi_rec].append('highest KPI')
        d[self.highest_support_rec].append('highest support')
        d[self.optimal_rec].append('optimal')
        result = {}
        for k, v in d.items():
            name = ', '.join(v)
            result[name] = k.__dict__()
        return result
    
    def __str__(self):
        return str(self.__dict__())

    def to_json(self):
        return json.dumps(self.__dict__())

def make_recommendations(dfs: list[tuple[float, pd.DataFrame]], df: pd.DataFrame) -> RecommendationPackage:
    n = len(dfs)
    candidates_map = {}
    average_sim = np.mean([sim for sim, df in dfs])
    for i in range(n):
        df1 = dfs[i][1]
        candidates_map[i] = RecommendationCandidate.generate_candidates(peer_df=df1, df=df, sim=average_sim)
    candidates = []
    for _, vals in candidates_map.items():
        candidates += vals
    recommendations = Recommendation.generate_recommendations(candidates=candidates)
    return RecommendationPackage(recommendations=recommendations)
