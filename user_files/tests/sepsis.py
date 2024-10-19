from algo.performance import KPIUtils
from common import *
from algo.pipeline import Pipeline
from algo.recommendation import Recommendation
from util.synchronize import synchronize

plot_dir_path = os.path.join('evaluation_results', 'sepsis')
actual_activities = {}
recommended_activities = {}
jaccard_sims = []
performances = []
u_test_data = []
elapsed_times = []

def visualize_runtime():
    mean_time = np.mean(elapsed_times)
    plt.figure(figsize=(8, 6))
    sns.boxplot(elapsed_times, color='skyblue', showfliers=False)
    plt.axhline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.4f} s')
    plt.legend()
    plt.title('Sepsis Patients: Recommendation Time')
    plt.ylabel('Time (seconds)')
    plt.savefig(os.path.join(plot_dir_path, 'runtime.png'))

def visualize_rec_effect(followers, nonfollowers):
    followers = np.array(followers)
    nonfollowers = np.array(nonfollowers)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Sepsis Patients: Effect of Following Recommendations on Performance', fontsize=16)
    sns.histplot(followers, kde=True, color='blue', label='followers', ax=axes[0, 0], stat='density', bins=20)
    sns.histplot(nonfollowers, kde=True, color='orange', label='non-followers', ax=axes[0, 0], stat='density', bins=20)
    axes[0, 0].set_title('Histogram with KDE')
    axes[0, 0].legend()

    sns.kdeplot(followers, color='blue', label='followers', ax=axes[0, 1], shade=True, clip=(0, 1))
    sns.kdeplot(nonfollowers, color='orange', label='non-followers', ax=axes[0, 1], shade=True, clip=(0, 1))
    axes[0, 1].set_title('Kernel Density Estimate (KDE)')
    axes[0, 1].legend()

    sns.boxplot(data=[followers, nonfollowers], ax=axes[1, 0], palette=["blue", "orange"])
    axes[1, 0].set_xticklabels(['followers', 'non-followers'])
    axes[1, 0].set_title('Boxplot')

    sns.violinplot(data=[followers, nonfollowers], ax=axes[1, 1], palette=["blue", "orange"], cut=0)
    axes[1, 1].set_xticklabels(['followers', 'non-followers'])
    axes[1, 1].set_title('Violin Plot')

    fig.text(0.5, 0.06, f'{len(followers)} followers, {len(nonfollowers)} non-followers', ha='center', fontsize=12)
    u_test_result = stats.mannwhitneyu(followers, nonfollowers)
    fig.text(0.5, 0.02, f'U-statistic: {u_test_result.statistic}, p-value: {u_test_result.pvalue}', ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    plt.savefig(os.path.join(plot_dir_path, 'recommendation_effect.png'))


def visualize_prediction(jaccard_sims):
    jaccard_sims = np.array(jaccard_sims)
    plt.figure(figsize=(10, 6))
    plt.title('Sepsis Patients: Predictive Accuracy of Recommendations')
    sns.kdeplot(jaccard_sims, color='blue', shade=True, clip=(0, 1))
    plt.savefig(os.path.join(plot_dir_path, 'predictive_strength.png'))

def visualize_activities():
    pairs = {}
    for k in set(actual_activities.keys()).union(set(recommended_activities.keys())):
        pairs[k] = [0, 0]
    for k, v in actual_activities.items():
        pairs[k][0] = v
    for k, v in recommended_activities.items():
        pairs[k][1] = v
        labels = list(pairs.keys())
    total_actual = sum([v[0] for v in pairs.values()])
    actual_values = [v[0] / total_actual for v in pairs.values()]
    total_recommended = sum([v[0] for v in pairs.values()])
    recommended_values = [v[1] / total_recommended for v in pairs.values()]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, actual_values, width, label='actual', color='blue')
    rects2 = ax.bar(x + width/2, recommended_values, width, label='recommended', color='orange')
    ax.set_xlabel('activities')
    ax.set_ylabel('density')
    ax.set_title('Sepsis Patients: Actual vs Recommended Activities')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir_path, 'activities.png'))


def jaccard(list1, list2):
    if not list1:
        return 0
    if not list2:
        return None
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def evaluate(commons: list[Common]):
    used_case_ids = set()
    for common in tqdm.tqdm(commons, 'Evaluating training-testing set pairs'):
        synchronize(common)        
        print(f'Training period: {common.training_period}')
        activity_col = common.conf.event_log_specs.activity
        case_ids = common.test_df[common.conf.event_log_specs.case_id].unique()
        for case_id in tqdm.tqdm(case_ids, 'Evaluating traces'):
            if case_id in used_case_ids:
                continue
            used_case_ids.add(case_id)
            full_original_df = common.conf.df[common.conf.df[common.conf.event_log_specs.case_id] == case_id]
            full_normalized_df = common.test_df[common.test_df[common.conf.event_log_specs.case_id] == case_id]
            if len(full_original_df) == 1:
                continue
            cutting_point = random.sample(range(1, len(full_original_df)), 1)[0]
            past_original_df = full_original_df.iloc[:cutting_point]
            if str(past_original_df[activity_col].iloc[-1]).startswith('Release'):
                continue
            future_original_df = full_original_df[cutting_point:]
            true_activities = list(future_original_df[(future_original_df[common.conf.event_log_specs.timestamp] - past_original_df[common.conf.event_log_specs.timestamp].iloc[-1]) <= pd.Timedelta(days=2)][common.conf.event_log_specs.activity])
            true_activities = true_activities[:min(len(true_activities), 3)]
            if not true_activities:
                continue
            performance = KPIUtils.instance.compute_kpi(full_normalized_df)[1]
            start_time = time.time()
            recommendations = Pipeline(df=past_original_df).get_all_recommendations(interactive=True)
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
            if not recommendations:
                continue
            proposed_activities = [r.event[common.conf.event_log_specs.activity] for r in recommendations]
            proposed_activities = proposed_activities[:min(len(proposed_activities), 2)]
            for k in true_activities:
                if k not in actual_activities:
                    actual_activities[k] = 1
                else:
                    actual_activities[k] += 1
            for k in proposed_activities:
                if k not in recommended_activities:
                    recommended_activities[k] = 1
                else:
                    recommended_activities[k] += 1
            print(f'recommended: {proposed_activities}')
            print(f'actual: {true_activities}')
            jaccard_sim = jaccard(proposed_activities, true_activities)
            if jaccard_sim is None:
                continue
            print(f'Jaccard similarity: {jaccard_sim}, performance: {performance}')
            jaccard_sims.append(jaccard_sim)
            performances.append(performance)
            counter = 0
            for act in proposed_activities:
                if act in true_activities:
                    counter += 1
            if counter >= min(2, len(true_activities)):
                u_test_data.append((True, performance))
            else:
                u_test_data.append((False, performance))
            followers = [performance for followed, performance in u_test_data if followed]
            nonfollowers = [performance for followed, performance in u_test_data if not followed]
            print(f'{len(followers)} followers, {len(nonfollowers)} non-followers')
            try:
                u_test_result = stats.mannwhitneyu(followers, nonfollowers)
                print(u_test_result)
                print(f'correlation: {u_test_result.statistic / (len(followers) * len(nonfollowers))}')
            except Exception as e:
                pass

    visualize_runtime()
    visualize_rec_effect(followers, nonfollowers)
    visualize_prediction(jaccard_sims)
    visualize_activities()
