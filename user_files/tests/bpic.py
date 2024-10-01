from algo.performance import compute_kpi
from util.common import *
from algo.pipeline import recommendation_pipeline
from algo.recommendation import Recommendation

plot_dir_path = os.path.join('evaluation_results', 'bpic')
no_recommendation = []
similarities = []
performances = []
ratios_first_withdrawal_amount = []
ratios_number_of_terms = []
ratios_monthly_cost = []
ratios_offered_amount = []
ratios_kpi = []

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

def plot_similarities():    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[similarities], showfliers=False, palette='Blues', fill=False)
    plt.title('Similarity of Recommended and Accepted Offer')
    plt.ylim(0, 1)
    plt.ylabel('similarity')
    medians = [np.median(similarities)]
    for i, median in enumerate(medians):
        plt.text(i, median, f'{median:.2f}', ha='center', va='bottom', color='black', fontsize=10)
    plt.savefig(os.path.join(plot_dir_path, 'similarities.svg'))

def plot_correlation():
    pearson_corr, _ = stats.pearsonr(similarities, performances)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=similarities, y=performances, label=f"r = {pearson_corr:.2f}")
    sns.regplot(x=similarities, y=performances, scatter=False)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.title('Correlation between Similarity and Performance', fontsize=14)
    plt.xlabel('similarity', fontsize=12)
    plt.ylabel('performance', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(plot_dir_path, 'correlation.svg'))

def plot_ratios():    
    ratios = [ratios_first_withdrawal_amount, ratios_monthly_cost, ratios_number_of_terms, ratios_offered_amount]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=ratios, showfliers=False, palette='cubehelix', fill=False)
    plt.xticks(ticks=[0,1,2,3], labels=['first withdrawal amount', 'monthly cost', 'number of terms', 'offered amount'])
    plt.title('Recommended vs Accepted Offer: Attributes')
    plt.ylabel('ratio')
    medians = [np.median(v) for v in ratios]
    for i, median in enumerate(medians):
        plt.text(i, median, f'{median:.2f}', ha='center', va='bottom', color='black', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir_path, 'ratios.svg'))

def plot_kpi_ratios():    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[ratios_kpi], showfliers=False, palette='Blues', fill=False)
    plt.title('KPI-to-Performance Ratio')
    plt.ylabel('ratio')
    medians = [np.median(ratios_kpi)]
    for i, median in enumerate(medians):
        plt.text(i, median, f'{median:.2f}', ha='center', va='bottom', color='black', fontsize=10)
    plt.savefig(os.path.join(plot_dir_path, 'kpi_ratios.svg'))

def evaluate(commons: list[Common]):
    for common in tqdm.tqdm([commons[0]], 'Evaluating training-testing set pairs'):
        Common.set_instance(common)
        print(f'Training period: {common.training_period}')
        activity_col = common.conf.event_log_specs.activity
        case_ids = common.test_df[common.conf.event_log_specs.case_id].unique()
        for case_id in tqdm.tqdm(case_ids, 'Testing trace'):
            full_original_df = common.conf.df[common.conf.df[common.conf.event_log_specs.case_id] == case_id]
            full_normalized_df = common.test_df[common.test_df[common.conf.event_log_specs.case_id] == case_id]
            past_original_df = full_original_df[full_original_df[common.conf.event_log_specs.timestamp] <= common.training_period[1]]
            past_normalized_df = full_normalized_df[full_normalized_df.index.isin(past_original_df.index)]
            future_original_df = full_original_df[full_original_df[common.conf.event_log_specs.timestamp] > common.training_period[1]]
            performance = compute_kpi(full_normalized_df)[1]
            if not performance:
                continue
            o_accepted_df = future_original_df[future_original_df[activity_col].isin(['O_Accepted'])]
            if o_accepted_df.empty:
                continue
            accepted_normalized_offer = full_normalized_df[
                (full_normalized_df['EventID'].isin([o_accepted_df['OfferID'].iloc[-1]])) &
                (full_normalized_df[activity_col].isin(['O_Create Offer']))
            ].iloc[-1]
            accepted_original_offer = full_original_df[
                (full_original_df['EventID'].isin([o_accepted_df['OfferID'].iloc[-1]])) &
                (full_original_df[activity_col].isin(['O_Create Offer']))
            ].iloc[-1]
            recommendation = recommendation_pipeline(df=past_normalized_df, interactive=False)
            if not recommendation:
                no_recommendation.append(True)
                continue
            no_recommendation.append(False)
            if accepted_original_offer['FirstWithdrawalAmount']:
                ratios_first_withdrawal_amount.append(recommendation.event['FirstWithdrawalAmount'] / accepted_original_offer['FirstWithdrawalAmount'])
            if accepted_original_offer['NumberOfTerms']:
                ratios_number_of_terms.append(recommendation.event['NumberOfTerms'] / accepted_original_offer['NumberOfTerms'])
            if accepted_original_offer['MonthlyCost']:
                ratios_monthly_cost.append(recommendation.event['MonthlyCost'] / accepted_original_offer['MonthlyCost'])
            if accepted_original_offer['OfferedAmount']:
                ratios_offered_amount.append(float(recommendation.event['OfferedAmount'] / accepted_original_offer['OfferedAmount']))
            sim = 1 - abs(accepted_normalized_offer['OfferedAmount'] - recommendation.normalized_event['OfferedAmount'])
            similarities.append(sim)
            performances.append(performance)
            ratios_kpi.append(recommendation.kpi / performance)
    plot_rec_statistics()
    plot_similarities()
    plot_correlation()
    plot_ratios()
    plot_kpi_ratios()
