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
