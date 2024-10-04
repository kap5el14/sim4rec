from algo.performance import KPIUtils
from common import *
from algo.pipeline import Pipeline
from algo.recommendation import Recommendation
from util.synchronize import synchronize

plot_dir_path = os.path.join('evaluation_results', 'sepsis')
no_recommendation = []
recommended_activities = {}
maps = []
performances = []
t_test_data = []

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

def plot_correlation():
    pearson_corr, _ = stats.pearsonr(maps, performances)
    spearman_corr, _ = stats.spearmanr(maps, performances)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=maps, y=performances, label=f"r1 = {pearson_corr:.2f}, r2 = {spearman_corr}, {len(maps)} pairs", palette=['blue'])
    sns.regplot(x=maps, y=performances, scatter=False)
    plt.title('Correlation', fontsize=14)
    plt.xlabel('mean average precision', fontsize=12)
    plt.ylabel('performance', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(plot_dir_path, 'correlation.svg'))

def average_precision(list1, list2):
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
            past_normalized_df = full_normalized_df[full_normalized_df.index.isin(past_original_df.index)]
            future_original_df = full_original_df[cutting_point:]
            actual_activities = list(future_original_df[(future_original_df[common.conf.event_log_specs.timestamp] - past_original_df[common.conf.event_log_specs.timestamp].iloc[-1]) <= pd.Timedelta(days=2)][common.conf.event_log_specs.activity])
            actual_activities = actual_activities[:min(len(actual_activities), 3)]
            if not actual_activities:
                continue
            performance = KPIUtils.instance.compute_kpi(full_normalized_df)[1]
            recommendations = Pipeline(df=past_normalized_df).get_all_recommendations(interactive=True)
            if not recommendations:
                no_recommendation.append(True)
                continue
            recommendation = recommendations[0]
            no_recommendation.append(False)
            recommended_activities = [r.event[common.conf.event_log_specs.activity] for r in recommendations]
            recommended_activities = recommended_activities[:min(len(recommended_activities), 2)]

            print(f'recommended: {recommended_activities}')
            print(f'actual: {actual_activities}')
            map_val = average_precision(recommended_activities, actual_activities)
            if map_val is None:
                continue
            print(f'map: {map_val}, performance: {performance}')
            maps.append(map_val)
            performances.append(performance)
            try:
                print(f'{stats.spearmanr(maps, performances)}')
                print(f'{stats.pearsonr(maps, performances)}')
            except Exception as e:
                pass
            for act in recommended_activities:
                if act in actual_activities:
                    t_test_data.append((True, performance))
                    break
            if not set(recommended_activities).intersection(set(actual_activities)):
                t_test_data.append((False, performance))
            followers = [performance for followed, performance in t_test_data if followed]
            nonfollowers = [performance for followed, performance in t_test_data if not followed]
            print(f'{len(followers)} followers, {len(nonfollowers)} non-followers')
            try:
                print(stats.mannwhitneyu(followers, nonfollowers))
            except Exception as e:
                pass

            
    
    plot_correlation()
