from util.common import *
from algo.similarity import similarity_between_trace_headers, similarity_between_traces

def first_pass(sample_size=10) -> list[pd.DataFrame]:
    common = Common.instance
    case_ids = common.train_df[common.event_log_specs.case_id]
    return [common.train_df[common.train_df[common.event_log_specs.case_id] == case_id] for case_id in random.sample(list(case_ids), min(sample_size, len(case_ids)))]

def second_pass(dfs: list[pd.DataFrame], df: pd.DataFrame, sample_size=20, log=False) -> list[pd.DataFrame]:
    common = Common.instance
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
            #print(f"{i}: {current_sim}")
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
        #print()
    best_dfs_sorted = sorted(best_dfs, key=lambda x: x[0], reverse=True)
    result = [df for _, df in best_dfs_sorted[:min(sample_size, len(best_dfs_sorted))]]
    if log:
        return result, log_results
    return result

def third_pass(dfs: list[pd.DataFrame], df: pd.DataFrame, sample_size=5) -> list[pd.DataFrame]:
    common = Common.instance
    best_dfs: list[tuple[float, pd.DataFrame]] = []
    for peer_df in tqdm.tqdm(dfs, desc='Third Pass: Processing Peer DataFrames'):
        sim = similarity_between_traces(df, peer_df)
        best_dfs.append((sim, peer_df))
    best_dfs_sorted = sorted(best_dfs, key=lambda x: x[0], reverse=True)
    return [df for _, df in best_dfs_sorted[:min(sample_size, len(best_dfs_sorted))]]

def sample_peers(df) -> list[pd.DataFrame]:
    return third_pass(dfs=second_pass(dfs=first_pass(), df=df), df=df)