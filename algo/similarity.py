from util.common import *
from scipy.stats import wasserstein_distance


def similarity_between_events(row1, row2):
    common = Common.get_instance()
    def ap():
        if row1[common.event_log_specs.activity] != row2[common.event_log_specs.activity]:
            return 0
        return 1 - abs(row1[ACTIVITY_OCCURRENCE] - row2[ACTIVITY_OCCURRENCE]) / 2
    def ceap(attr):
        if row1[attr] == row2[attr]:
            return 1
        return 0
    def neap(attr):
        diffs = []
        for a in [attr, f'{attr}{CUMSUM}', f'{attr}{CUMAVG}', f'{attr}{MW_SUM}', f'{attr}{MW_AVG}']:
            diffs.append(abs(row1[a] - row2[a]))
        return 1 - sum(diffs) / 5
    def tep():
        diffs = []
        for a in [common.event_log_specs.timestamp, TIME_FROM_TRACE_START, TIME_FROM_PREVIOUS_EVENT]:
            diffs.append(abs(row1[a] - row2[a]))
        return 1 - sum(diffs) / 3
    sims = {
        'ap': ap(),
        'tep': tep(),
    }
    weights = {
        'ap': common.similarity_weights.activity / 2,
        'tep': common.similarity_weights.timestamp / 2,
    }
    for cat_attr, weight in common.similarity_weights.categorical_event_attributes.items():
        sims[f'ceap({cat_attr})'] = ceap(cat_attr)
        weights[f'ceap({cat_attr})'] = weight
    for num_attr, weight in common.similarity_weights.numerical_event_attributes.items():
        sims[f'neap({num_attr})'] = neap(num_attr)
        weights[f'neap({num_attr})'] = weight / 2
    return sum([sims[k] * weights[k] for k in sims.keys()]) / common.similarity_weights.event

def similarity_between_trace_headers(df1, df2, log=False):
    common = Common.get_instance()
    component_sims = {}
    def asp():
        c_sims = []
        for a in [UNIQUE_ACTIVITIES, ACTIVITIES_MEAN, ACTIVITIES_STD]:
            r = 1 - abs(df1[a].iloc[-1] - df2[a].iloc[-1])
            component_sims[a] = r
            c_sims.append(r)
        return sum(c_sims) / 3
    def ctap(attr):
        r = 1 if df1[attr].iloc[0] == df2[attr].iloc[0] else 0
        component_sims[attr] = r
        return r
    def ntap(attr):
        r = 1 - abs(df1[attr].iloc[0] - df2[attr].iloc[0])
        component_sims[attr] = r
        return r
    def anap(attr):
        c_sims = []
        for a in [f'{attr}{CUMSUM}', f'{attr}{CUMAVG}']:
            r = 1 - abs(df1[a].iloc[-1] - df2[a].iloc[-1])
            component_sims[a] = r
            c_sims.append(r)
        return sum(c_sims) / 2
    def ttp():
        c_sims = []
        r = 1 - abs(df1[TRACE_START].iloc[0] - df2[TRACE_START].iloc[0])
        component_sims[TRACE_START] = r
        c_sims.append(r)
        r = 1 - abs(df1[TIME_FROM_TRACE_START].iloc[-1] - df2[TIME_FROM_TRACE_START].iloc[-1])
        component_sims[TRACE_DURATION] = r
        c_sims.append(r)
        r = 1 - abs(df1[common.event_log_specs.timestamp].iloc[-1] - df2[common.event_log_specs.timestamp].iloc[-1])
        component_sims[TRACE_END] = r
        c_sims.append(r)
        return sum(c_sims) / 3
    def tlp():
        r = 1 - abs(df1[INDEX].iloc[-1] - df2[INDEX].iloc[-1])
        component_sims[TRACE_LENGTH] = r
        return r
    sims = {
        'asp': asp(),
        'ttp': ttp(),
        'tlp': tlp()
    }
    weights = {
        'asp': common.similarity_weights.activity / 2,
        'ttp': common.similarity_weights.timestamp / 2,
        'tlp': common.similarity_weights.trace_length
    }
    for cat_attr, weight in common.similarity_weights.categorical_trace_attributes.items():
        sims[f"ctap({cat_attr})"] = ctap(cat_attr)
        weights[f"ctap({cat_attr})"] = weight
    for num_attr, weight in common.similarity_weights.numerical_event_attributes.items():
        sims[f"anap({num_attr})"] = anap(num_attr)
        weights[f"anap({num_attr})"] = weight / 2
    for num_attr, weight in common.similarity_weights.numerical_trace_attributes.items():
        sims[f"ntap({num_attr})"] = ntap(num_attr)
        weights[f"ntap({num_attr})"] = weight
    result = sum([sims[k] * weights[k] for k in sims.keys()]) / common.similarity_weights.trace
    if log:
        return result, component_sims
    return result

def similarity_between_traces(df1, df2, log=False):
    common = Common.get_instance()
    def ed(distance_matrix, m, n):
        dp = np.zeros((m + 1, n + 1))
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + 1
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = min(
                    dp[i - 1][j] + 1, 
                    dp[i][j - 1] + 1,  
                    dp[i - 1][j - 1] + distance_matrix[i - 1, j - 1]  
                )
        return 1 - dp[m][n] / max(m,n)
    def gm(distance_matrix, m, n):
        flattened = [(distance_matrix[i, j], (i, j)) for i in range(distance_matrix.shape[0]) for j in range(distance_matrix.shape[1])]
        sorted_flattened = sorted(flattened, key=lambda x: x[0])
        set1 = set()
        set2 = set()
        total_cost = 0
        for cost, (i,j) in sorted_flattened:
            if i not in set1 and j not in set2:
                total_cost += cost
                set1.add(i)
                set2.add(j)
            if len(set1) == m or len(set2) == n:
                return 1 - (total_cost + max(m,n) - min(m,n)) / max(m,n)
    def emd(m, n):
        distance_matrix1 = np.zeros((m, m))
        for i in range(m):
            row1 = df1.iloc[i]
            for j in range(m):
                row2 = df1.iloc[j]
                distance_matrix1[i, j] = 1 - similarity_between_events(row1, row2)
        distance_matrix2 = np.zeros((n, n))
        for i in range(n):
            row1 = df2.iloc[i]
            for j in range(n):
                row2 = df2.iloc[j]
                distance_matrix2[i, j] = 1 - similarity_between_events(row1, row2)
        distances_A = distance_matrix1[np.triu_indices_from(distance_matrix1, k=1)]
        distances_B = distance_matrix2[np.triu_indices_from(distance_matrix2, k=1)]
        if distances_A.size == 0 and distances_B.size == 0:
            return 1
        if distances_A.size == 0 or distances_B.size == 0: 
            return 0
        all_values = list(distances_A) + list(distances_B)
        perc_values = np.percentile(all_values, [0,10,20,30,40,50,60,70,80,90,100])
        hist_A, _ = np.histogram(distances_A, bins=perc_values)
        hist_B, _ = np.histogram(distances_B, bins=perc_values)
        max_sum = max(np.sum(hist_A), np.sum(hist_B))
        hist_A = hist_A / max_sum
        hist_B = hist_B / max_sum
        return 1 - wasserstein_distance(hist_A, hist_B)
    num_rows_df1 = len(df1)
    num_rows_df2 = len(df2)
    distance_matrix = np.zeros((num_rows_df1, num_rows_df2))
    for i in range(num_rows_df1):
        row1 = df1.iloc[i]
        for j in range(num_rows_df2):
            row2 = df2.iloc[j]
            distance_matrix[i, j] = 1 - similarity_between_events(row1, row2)
    if log:
        th_sim, th_component_similarities = similarity_between_trace_headers(df1, df2, log=True)
    else:
        th_sim = similarity_between_trace_headers(df1, df2)
    sims = {
        'th': th_sim,
        'ed': ed(distance_matrix, num_rows_df1, num_rows_df2),
        'gm': gm(distance_matrix, num_rows_df1, num_rows_df2),
        'emd': emd(num_rows_df1, num_rows_df2)
    }
    weights = {
        'th': common.similarity_weights.trace,
        'ed': common.similarity_weights.event / 3,
        'gm': common.similarity_weights.event / 3,
        'emd': common.similarity_weights.event / 3
    }
    result = sum([sims[k] * weights[k] for k in sims.keys()])
    if log:
        return result, sims, th_component_similarities
    return result
