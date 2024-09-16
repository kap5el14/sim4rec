from util.common import *
from algo.similarity import similarity_between_events, similarity_between_trace_headers
from data.success import is_successful


def compute_kpi(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    common = Common.instance
    if not is_successful(df):
        return {}, 0
    kpi_dict = {}
    kpi_weights = {}
    if common.performance_weights.trace_length:
        kpi_dict[TRACE_LENGTH] = 1 - df[TRACE_LENGTH].iloc[-1]
        kpi_weights[TRACE_LENGTH] = common.performance_weights.trace_length
    if common.performance_weights.trace_duration:
        kpi_dict[TRACE_DURATION] = 1 - df[TRACE_DURATION].iloc[-1]
        kpi_weights[TRACE_DURATION] = common.performance_weights.trace_duration
    for k, v in common.performance_weights.numerical_trace_attributes.items():
        if 'min' in v:
            kpi_dict[k] = 1 - df[k].iloc[-1]
        elif 'max' in v:
            kpi_dict[k] = df[k].iloc[-1]
        else:
            raise ValueError
        kpi_weights[k] = v[1]
    for k, v in common.performance_weights.categorical_trace_attributes.items():
        value = v[0]
        name = f"{k}=={value}"
        kpi_dict[name] = 1 if df[k] == value else 0
        kpi_weights[name] = v[1]
    for k, v in common.performance_weights.numerical_event_attributes.items():
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
    similar_candidates: list['RecommendationCandidate'] = field(init=False, default=[])
    marked: bool = field(init=False, default=False)

    @classmethod
    def generate_candidates(cls, peer_df: pd.DataFrame, df: pd.DataFrame, sim: float):
        common = Common.instance
        complete_peer_df = common.train_df[common.train_df[common.event_log_specs.case_id] == peer_df[common.event_log_specs.case_id].iloc[0]]
        last_values = df[[TIME_FROM_TRACE_START, INDEX]].iloc[-1]
        complete_peer_df_filtered = complete_peer_df[[TIME_FROM_TRACE_START, INDEX]]
        difference_df = complete_peer_df_filtered - last_values
        proximities = 1 - difference_df.abs().sum(axis=1)
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
                candidates.append(RecommendationCandidate(row=row, proximity=proximities[i], similarity=sim, future_performance=fp, kpi_dict=kpi_dict))

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
    event: pd.Series
    peers: set[str]
    support: float
    future_performance: float
    kpi_dict: dict[str, float]
    proximity: float
    similarity: float

    @classmethod
    def generate_recommendations(cls, candidates: list[RecommendationCandidate]) -> list['Recommendation']:
        clusters: list[set[RecommendationCandidate]] = []
        for c in candidates:
            if c.marked:
                continue
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
        common = Common.instance
        all_peers = set()
        recommendations = []
        for cluster in clusters:
            peers = set()
            similarity_sum = 0
            proximity_sum = 0
            future_performance_sum = 0
            kpi_sums = {}
            for candidate in cluster:
                case_id = candidate.row[common.event_log_specs.case_id].iloc[0]
                peers.add(case_id)
                all_peers.add(case_id)
                similarity_sum += candidate.similarity
                proximity_sum += candidate.proximity
                future_performance_sum += candidate.future_performance
                for k, v in candidate.kpi_dict.items():
                    if k in kpi_sums:
                        kpi_sums[k] += v
                    else:
                        kpi_sums[k] = v
            support = len(peers) / len(all_peers)
            similarity = similarity_sum / len(peers)
            proximity = proximity_sum / len(peers)
            future_performance = future_performance_sum / len(peers)
            kpi_dict = {k: v / len(peers) for k, v in kpi_sums.items()}
            candidates_df = pd.DataFrame([c.row for c in cluster])
            numerical_cols = candidates_df.select_dtypes(include=[np.number])
            categorical_cols = candidates_df.select_dtypes(exclude=[np.number])
            numerical_avg = numerical_cols.mean()
            categorical_mode = categorical_cols.mode().iloc[0]
            result_event = pd.concat([numerical_avg, categorical_mode])
            rec = Recommendation(
                event=result_event,
                peers=peers,
                support=support,
                future_performance=future_performance,
                kpi_dict=kpi_dict,
                proximity=proximity,
                similarity=similarity
            )
            recommendations.append(rec)
        return recommendations

def make_recommendations(dfs: list[tuple[pd.DataFrame, float]], df: pd.DataFrame) -> list[Recommendation]:
    n = len(dfs)
    candidates_map = {}
    for i in range(n):
        df1 = dfs[i][0]
        candidates_map[i] = RecommendationCandidate.generate_candidates(peer_df=df1, df=df, sim=dfs[i][1])
    for i in range(n):
        df1 = dfs[i][0]
        candidates1 = candidates_map[i]
        for j in range(i + 1, n):
            df2 = dfs[j][0]
            candidates2 = candidates_map[j]
            RecommendationCandidate.connect_candidates(candidates1=candidates1, candidates2=candidates2, sim=similarity_between_trace_headers(df1, df2))
    candidates = []
    for _, vals in candidates_map.items():
        candidates += vals
    return Recommendation.generate_recommendations(candidates=candidates)

