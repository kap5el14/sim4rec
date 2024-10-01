from algo.performance import compute_kpi
from util.common import *
from algo.pipeline import recommendation_pipeline
from algo.recommendation import Recommendation

plot_dir_path = os.path.join('evaluation_results', 'edu')
# No recommendation made
no_recommendation = []
# Already passed course was recommended
already_passed = []
# Nicht bestanden or gestrichen was recommended
not_passed = []
not_passed_counts = {}
# relative_semester doesn't align with the recommendation semester
divergent_semester = []
# actual semester minus relative semester
semester_deltas = []
# True if recommendation was followed, False otherwise
mask = []
# recommendation.kpi / performance if performance and followed
ratios_followers = []
# recommendation.kpi / performance if performance and didn't follow
ratios_nonfollowers = []
# A: recommendation followed, B: performance
t_test_data = []
recommendation_counts = {}
kpi_dict = {}

def plot_rec_statistics():
    fractions = [
        sum(no_recommendation) / len(no_recommendation),
        sum(already_passed) / len(already_passed),
        sum(not_passed) / len(not_passed),
        sum(divergent_semester) / len(divergent_semester)
    ]
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x=['no recommendations', 'already passed', 'state != passed', 'semester differs'], y=fractions, palette="rocket_r")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.title('Recommendation Statistics')
    for i, v in enumerate(fractions):
        bar_plot.text(i, v, f'{round(v * 100)}%', ha='center', va='bottom')
    plt.savefig(os.path.join(plot_dir_path, 'rec_statistics.svg'))

def plot_semester_deltas():
    counts = Counter(semester_deltas)
    total = sum(counts.values())
    normalized_counts = {k: v / total for k, v in counts.items()}
    sorted_keys = sorted(normalized_counts.keys())
    sorted_values = [normalized_counts[k] for k in sorted_keys]
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x=sorted_keys, y=sorted_values, palette='Blues')
    plt.ylim(0, 1)
    for i, v in enumerate(sorted_values):
        bar_plot.text(i, v, f'{round(v * 100)}%', ha='center', va='bottom')
    plt.xlabel('delta')
    plt.ylabel('density')
    plt.title('Semester Deltas')
    plt.savefig(os.path.join(plot_dir_path, 'semester_deltas.svg'))

def plot_mask():
    following = sum(mask) / len(mask)
    plt.figure(figsize=(10, 6))
    bar = plt.bar(['recommendation'], [following], color=['orange'])[0]
    plt.ylim(0, 1)
    plt.ylabel('following')
    plt.title('Recommendation Following')
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{round(following * 100)}%', ha='center', va='bottom')
    plt.savefig(os.path.join(plot_dir_path, 'masks.svg'))

def plot_kpi_ratios():    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[ratios_nonfollowers, ratios_followers], showfliers=False, palette=['red', 'green'], fill=False)
    plt.xticks(ticks=[0, 1], labels=['non-followers', 'followers'])
    plt.title('KPI-to-Performance Ratio')
    plt.ylabel('ratio')
    medians = [np.median(ratios_nonfollowers), np.median(ratios_followers)]
    for i, median in enumerate(medians):
        plt.text(i, median, f'{median:.2f}', ha='center', va='bottom', color='black', fontsize=10)
    plt.savefig(os.path.join(plot_dir_path, 'kpi_ratios.svg'))

def plot_t_test():
    followers = [performance for followed, performance in t_test_data if followed]
    nonfollowers = [performance for followed, performance in t_test_data if not followed]
    t_stat, p_value = stats.ttest_ind(followers, nonfollowers)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(followers, label='followers', color='green', shade=True)
    sns.kdeplot(nonfollowers, label='non-followers', color='red', shade=True)
    plt.title("Effect of Following a Recommendation on Performance")
    plt.text(1.05, 0.5, f"T-statistic: {t_stat:.4f}\nP-value: {p_value:.4f}", ha='left', va='center', fontsize=10, rotation=90, transform=plt.gca().transAxes)
    plt.xlabel('performance')
    plt.ylabel('density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir_path, 't_test.svg'))

def plot_coverage():
    plt.figure(figsize=(10, 6))
    sns.histplot(recommendation_counts.values(), stat='density', bins=30, kde=True, palette='viridis')
    mean_value = np.mean(list(recommendation_counts.values()))
    plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {round(mean_value * 100)}%')
    plt.ylim(0, 1)
    plt.xlabel('recommendations')
    plt.ylabel('courses')
    plt.title('Coverage')
    plt.legend()
    plt.savefig(os.path.join(plot_dir_path, 'coverage.svg'))

def plot_not_passed():
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(not_passed_counts.values()), y=list(not_passed_counts.keys()), orient='h', palette='crest')
    plt.xlabel('count')
    plt.ylabel('course')
    plt.title('Courses Recommended with state != \'Bestanden\'')
    plt.savefig(os.path.join(plot_dir_path, 'not_passed.svg'))

def plot_component_kpis():    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=list(kpi_dict.values()), showfliers=False, palette='cubehelix', fill=False)
    plt.xticks(ticks=list(range(len(kpi_dict))), labels=list(kpi_dict.keys()))
    plt.title('Component KPIs')
    plt.ylabel('KPI value')
    medians = [np.median(v) for v in kpi_dict.values()]
    for i, median in enumerate(medians):
        plt.text(i, median, f'{median:.2f}', ha='center', va='bottom', color='black', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir_path, 'component_kpis.svg'))

def evaluate(commons: list[Common]):
    for common in tqdm.tqdm([commons[0]], 'Evaluating training-testing set pairs'):
        Common.set_instance(common)
        print(f'Training period: {common.training_period}')
        activity_col = common.conf.event_log_specs.activity
        case_ids = common.test_df[common.conf.event_log_specs.case_id].unique()[:5]
        for course in list(common.conf.df[activity_col]):
            if course not in recommendation_counts:
                recommendation_counts[course] = 0
        for case_id in tqdm.tqdm(case_ids, 'Testing trace'):
            full_original_df = common.conf.df[common.conf.df[common.conf.event_log_specs.case_id] == case_id]
            full_normalized_df = common.test_df[common.test_df[common.conf.event_log_specs.case_id] == case_id]
            past_original_df = full_original_df[full_original_df[common.conf.event_log_specs.timestamp] <= common.training_period[1]]
            past_normalized_df = full_normalized_df[full_normalized_df.index.isin(past_original_df.index)]
            future_original_df = full_original_df[full_original_df[common.conf.event_log_specs.timestamp] > common.training_period[1]]
            recommendation = recommendation_pipeline(df=past_normalized_df, interactive=False)
            course = recommendation.event[activity_col]
            if not recommendation:
                no_recommendation.append(True)
                continue
            no_recommendation.append(False)
            already_passed.append('Bestanden' in list(past_normalized_df[past_normalized_df[activity_col].isin([recommendation.event[activity_col]])]['state']))
            not_passed.append(recommendation.event['state'] != 'Bestanden')
            if not_passed[-1]:
                if course in not_passed_counts:
                    not_passed_counts[course] += 1
                else:
                    not_passed_counts[course] = 1
            target_semester = int(past_normalized_df['relative_semester'].iloc[-1]) + 1
            actual_semester = recommendation.event['relative_semester']
            divergent_semester.append(bool(target_semester != actual_semester))
            semester_deltas.append(actual_semester - target_semester)
            courses_taken_next_semester = list(future_original_df[future_original_df['relative_semester'] == target_semester][activity_col])
            followed = course in courses_taken_next_semester
            mask.append(followed)
            performance = compute_kpi(full_normalized_df)[1]
            if performance:
                if followed:
                    ratios_followers.append(recommendation.kpi / performance)
                else:
                    ratios_nonfollowers.append(recommendation.kpi / performance)
            t_test_data.append((followed, performance))
            recommendation_counts[course] += 1
            for k, v in recommendation.kpi_dict.items():
                if k in kpi_dict:
                    kpi_dict[k].append(v)
                else:
                    kpi_dict[k] = [v]
    plot_rec_statistics()
    plot_semester_deltas()
    plot_mask()
    plot_kpi_ratios()
    plot_t_test()
    plot_coverage()
    plot_not_passed()
    plot_component_kpis()
