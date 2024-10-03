from algo.performance import compute_kpi
from util.common import *
from algo.pipeline import recommendation_pipeline
from algo.recommendation import Recommendation

plot_dir_path = os.path.join('evaluation_results', 'sepsis')
no_recommendation = []
performances1 = []
performances2 = []
recommended_activities = {}

def plot_rec_statistics():
    fractions = [
        sum(no_recommendation) / len(no_recommendation)
    ]
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x=['no recommendations'], y=fractions, palette="rocket_r")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.title('Recommendation Statistics')
    for i, v in enumerate(fractions):
        bar_plot.text(i, v, f'{round(v * 100)}%', ha='center', va='bottom')
    plt.savefig(os.path.join(plot_dir_path, 'rec_statistics.svg'))

def plot_t_test1():
    t_stat, p_value = stats.ttest_ind(scores1, performances1)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(scores1, label='strength of recommendation', color='green', shade=True)
    sns.kdeplot(performances1, label='performance', color='red', shade=True)
    plt.title("Released")
    plt.text(1.05, 0.5, f"T-statistic: {t_stat:.4f}\nP-value: {p_value:.4f}", ha='left', va='center', fontsize=10, rotation=90, transform=plt.gca().transAxes)
    plt.xlabel('performance')
    plt.ylabel('density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir_path, 't_test1.svg'))

def plot_t_test2():
    t_stat, p_value = stats.ttest_ind(performances1, performances2)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(performances1, label='followed', color='green', shade=True)
    sns.kdeplot(performances2, label='did not follow', color='red', shade=True)
    plt.title("Not Released")
    plt.text(1.05, 0.5, f"T-statistic: {t_stat:.4f}\nP-value: {p_value:.4f}", ha='left', va='center', fontsize=10, rotation=90, transform=plt.gca().transAxes)
    plt.xlabel('performance')
    plt.ylabel('density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir_path, 't_test2.svg'))

def evaluate(commons: list[Common]):
    for common in tqdm.tqdm(commons, 'Evaluating training-testing set pairs'):
        Common.set_instance(common)
        print(f'Training period: {common.training_period}')
        activity_col = common.conf.event_log_specs.activity
        case_ids = common.test_df[common.conf.event_log_specs.case_id].unique()
        for case_id in tqdm.tqdm(case_ids, 'Evaluating traces'):
            full_original_df = common.conf.df[common.conf.df[common.conf.event_log_specs.case_id] == case_id]
            full_normalized_df = common.test_df[common.test_df[common.conf.event_log_specs.case_id] == case_id]
            past_original_df = full_original_df[full_original_df[common.conf.event_log_specs.timestamp] <= common.training_period[1]]
            if str(past_original_df[activity_col].iloc[-1]).startswith('Release'):
                continue
            past_normalized_df = full_normalized_df[full_normalized_df.index.isin(past_original_df.index)]
            future_original_df = full_original_df[full_original_df[common.conf.event_log_specs.timestamp] > common.training_period[1]]
            performance = compute_kpi(full_normalized_df)[1]
            recommendation = recommendation_pipeline(df=past_normalized_df, interactive=False)
            if not recommendation:
                no_recommendation.append(True)
                continue
            no_recommendation.append(False)
            recommended_activity = recommendation.event[common.conf.event_log_specs.activity]
            actual_activities = future_original_df[(future_original_df[common.conf.event_log_specs.timestamp] - past_original_df[common.conf.event_log_specs.timestamp].iloc[-1]) <= pd.Timedelta(days=14)][common.conf.event_log_specs.activity].values
            print(f'recommended: {recommended_activity}')
            print(f'actual: {actual_activities}')
            if recommended_activity in actual_activities:
                performances1.append(performance)
            else:
                performances2.append(performance)
            print(stats.ttest_ind(performances1, performances2))
            input("Press enter...")

    plot_rec_statistics()
    plot_t_test1()
    plot_t_test2()
