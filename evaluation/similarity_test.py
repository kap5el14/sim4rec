from util.common import *
from algo.similarity import similarity_between_traces, similarity_between_trace_headers
from algo.sampling import first_pass, second_pass, third_pass

def plot_similarities(iters=1, sample_size=1):
    def plot(listA, listB, listC):
        plt.figure(figsize=(10, 6))
        plt.plot(listA, label='Similarities between trace headers', color='blue', marker='o')
        plt.plot(listB, label='Similarities between traces', color='green')
        for i in listC:
            plt.scatter(i, listA[i], color='red', zorder=5)
            plt.text(i, listA[i], f'({i}, {listA[i]})', fontsize=10, ha='right')
        for i in range(len(listC) - 1):
            start_idx = listC[i]
            end_idx = listC[i+1]
            plt.annotate(
                '', 
                xy=(end_idx, listA[end_idx]), 
                xytext=(start_idx, listA[start_idx]),
                arrowprops=dict(arrowstyle="->", color='red', lw=1.5)
            )
        plt.xlabel('Peer Trace Prefix Length')
        plt.ylabel('Similarity')
        plt.title('Second Pass: Visualization')
        plt.legend()
        plt.grid(True)
        plt.show()
    while iters:
        common = Common.get_instance()
        sampled_test_case_id = random.choice(list(common.test_df[common.event_log_specs.case_id]))
        df = common.test_df[common.test_df[common.event_log_specs.case_id] == sampled_test_case_id]
        idx = random.choice(list(range(1, len(df) + 1)))
        df = df.head(idx)
        dfs = first_pass(sample_size=sample_size)
        trace_header_sims = []
        trace_sims = []
        for peer_df in dfs:
            th_sims = []
            t_sims = []
            for i in tqdm.tqdm(range(1, len(peer_df) + 1), desc='Plot similarities'):
                trimmed_df = peer_df.head(i)
                th_sims.append(similarity_between_trace_headers(df, trimmed_df))
                t_sims.append(similarity_between_traces(df, trimmed_df))
            trace_header_sims.append(th_sims)
            trace_sims.append(t_sims)
        dfs, log_results = second_pass(dfs=dfs, df=df, sample_size=sample_size, log=True)
        print(len(df))
        for i in range(sample_size):
            plot(trace_header_sims[i], trace_sims[i], log_results[i])
        iters -= 1
