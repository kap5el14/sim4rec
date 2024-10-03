from util.common import *
from algo.similarity import similarity_between_trace_headers, similarity_between_traces
from algo.performance import compute_kpi

common = None
successful_dfs = None

def first_pass() -> list[pd.DataFrame]:
    global successful_dfs, common
    if common is not None and common == Common.instance:
        return successful_dfs
    common = Common.instance
    case_ids = common.train_df[common.conf.event_log_specs.case_id].unique()
    dfs = []
    for case_id, df in common.train_df.groupby(common.conf.event_log_specs.case_id):
        if case_id in case_ids:
            dfs.append(df)
    successful_dfs = [df for df in dfs if compute_kpi(df)[1]]
    return successful_dfs

def second_pass(dfs: list[pd.DataFrame], df: pd.DataFrame, sample_size=80, log=False) -> list[pd.DataFrame]:
    best_dfs: list[tuple[float, pd.DataFrame]] = []
    if log:
        log_results = []
    for peer_df in dfs:
        last_values = df[[TIME_FROM_TRACE_START, INDEX]].iloc[-1]
        peer_df_filtered = peer_df[[TIME_FROM_TRACE_START, INDEX]]
        difference_df = peer_df_filtered - last_values
        row_differences = difference_df.abs().sum(axis=1)
        min_diff_index = row_differences.idxmin() - row_differences.index[0] + 1
        queue = []
        heapq.heappush(queue, (-float('inf'), -float('inf'), min_diff_index))
        visited = set([min_diff_index])
        best_sim = -float('inf')
        best_df = None
        max_iterations = 10
        new_log_result = []
        while queue and max_iterations:
            sim, priority, i = heapq.heappop(queue)
            sim = -sim
            if sim < best_sim:
                continue
            trimmed_df = peer_df.head(i)
            current_sim = similarity_between_trace_headers(df, trimmed_df)
            if current_sim >= best_sim:
                best_sim = current_sim
                best_df = trimmed_df
                if log:
                    new_log_result.append(i)
                neighbors = [(i + 3, 1), (i - 3, 1), (i + 1, 2), (i - 1, 2)]
                for neighbor, priority in neighbors:
                    if neighbor not in visited and 0 < neighbor < len(peer_df) + 1:
                        heapq.heappush(queue, (-current_sim, priority, neighbor))
                        visited.add(neighbor)
            max_iterations -= 1
        best_dfs.append((best_sim, best_df))
        if log:
            log_results.append(new_log_result)
    best_dfs_sorted = sorted(best_dfs, key=lambda x: x[0], reverse=True)
    result = [df for _, df in best_dfs_sorted[:min(sample_size, len(best_dfs_sorted))]]
    if log:
        return result, log_results
    return result

def third_pass(dfs: list[pd.DataFrame], sample_size=40) -> list[pd.DataFrame]:
    return list(sorted(dfs, key=lambda df: compute_kpi(df=df)[1], reverse=True))[:min(len(dfs), sample_size)]

def fourth_pass(dfs: list[pd.DataFrame], df: pd.DataFrame, sample_size=20) -> list[tuple[float, pd.DataFrame]]:
    best_dfs: list[tuple[float, pd.DataFrame]] = []
    for peer_df in dfs:
        sim = similarity_between_traces(df, peer_df)
        best_dfs.append((sim, peer_df))
    best_dfs_sorted = sorted(best_dfs, key=lambda x: x[0], reverse=True)
    return best_dfs_sorted[:min(sample_size, len(best_dfs_sorted))]

def sample_peers(df) -> list[tuple[float, pd.DataFrame]]:
    return fourth_pass(dfs=third_pass(dfs=second_pass(dfs=first_pass(), df=df)), df=df)