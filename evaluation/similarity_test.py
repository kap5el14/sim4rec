from util.common import *
from algo.similarity import similarity_between_traces, similarity_between_trace_headers
from algo.sampling import first_pass, second_pass, third_pass

def plot_similarities(sample_size=1):
    def plot_similarities(t_sims, th_sims, ed_sims, emd_sims, gm_sims, i):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(t_sims) + 1), t_sims, label='Trace Similarities', color='green', linewidth=5)
        plt.plot(range(1, len(t_sims) + 1), th_sims, label='Trace Header Similarities', color='red')
        plt.plot(range(1, len(t_sims) + 1), ed_sims, label='ED Similarities', color='blue')
        plt.plot(range(1, len(t_sims) + 1), emd_sims, label='EMD Similarities', color='brown')
        plt.plot(range(1, len(t_sims) + 1), gm_sims, label='GM Similarities', color='orange')
        plt.axvline(x=i, linestyle='--', linewidth=2, label=f'Cutting point identified by the algorithm')
        plt.xlabel('Cutting Point')
        plt.ylabel('Similarity')
        plt.title('Similarity Plot')
        plt.grid(True)
        plt.legend()
        plt.savefig("data/trace_sims.jpg")
    def plot_component_similarities(component_sims: pd.DataFrame, i):
        plt.figure(figsize=(10, 6))
        for column in component_sims.columns:
            random_color = np.random.rand(3,)
            plt.plot(range(1, len(component_sims) + 1), component_sims[column], label=column, color=random_color)
        plt.axvline(x=i, linestyle='--', linewidth=2, label=f'Cutting point identified by the algorithm')
        plt.title("Component Similarities")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("data/component_sims.jpg")
    common = Common.get_instance()
    sampled_test_case_id = random.choice(list(common.test_df[common.event_log_specs.case_id]))
    df = common.test_df[common.test_df[common.event_log_specs.case_id] == sampled_test_case_id]
    idx = random.choice(list(range(1, len(df) + 1)))
    df = df.head(idx)
    [peer_df] = first_pass(sample_size=1)
    th_sims = []
    t_sims = []
    ed_sims = []
    emd_sims = []
    gm_sims = []
    component_sims = []
    for i in tqdm.tqdm(range(1, len(peer_df) + 1), desc='Plot similarities'):
        trimmed_df = peer_df.head(i)
        t_sim, sims, c_sims = similarity_between_traces(df, trimmed_df, log=True)
        th_sims.append(sims['th'])
        ed_sims.append(sims['ed'])
        emd_sims.append(sims['emd'])
        gm_sims.append(sims['gm'])
        t_sims.append(t_sim)
        component_sims.append(c_sims)
    i = len(second_pass(dfs=[peer_df], df=df)[0])
    plot_similarities(t_sims, th_sims, ed_sims, emd_sims, gm_sims, i)
    plot_component_similarities(pd.DataFrame(component_sims), i)

def pearson_correlation():
    def plot_correlations(maxima):
        df = pd.DataFrame(maxima)
        correlation_matrix = df.corr(method='pearson')
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title('Pearson Correlation Between Pairs of Lists')
        plt.show()
    common = Common.get_instance()
    test_case_ids = list(common.test_df[common.event_log_specs.case_id].unique())
    maxima = {
        't_sims': [],
        'th_sims': [],
        'ed_sims': [],
        'emd_sims': [],
        'gm_sims': [],
        'identified': []
    }
    for test_case_id in tqdm.tqdm(test_case_ids, "Testing set"):
        df = common.test_df[common.test_df[common.event_log_specs.case_id] == test_case_id]
        idx = random.choice(list(range(1, len(df) + 1)))
        df = df.head(idx)
        [peer_df] = first_pass(sample_size=1)
        res = {
            't_sims': [],
            'th_sims': [],
            'ed_sims': [],
            'emd_sims': [],
            'gm_sims': []
        }
        for i in range(1, len(peer_df) + 1):
            trimmed_df = peer_df.head(i)
            t_sim, sims, c_sims = similarity_between_traces(df, trimmed_df, log=True)
            res['t_sims'].append(t_sim)
            res['th_sims'].append(sims['th'])
            res['ed_sims'].append(sims['ed'])
            res['emd_sims'].append(sims['emd'])
            res['gm_sims'].append(sims['gm'])
        for key in res.keys():
            results = res[key]
            maxima[key].append(results.index(max(results)) + 1)
        maxima['identified'].append(len(second_pass(dfs=[peer_df], df=df)[0]))
    plot_correlations(maxima)

    
        
