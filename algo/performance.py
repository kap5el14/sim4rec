from util.common import *

performance_dict = {}
perc_values = None

def normalize(tup: tuple[dict[str, float], float]) -> tuple[dict[str, float], float]:
    (kpi_dict, performance) = tup
    global perc_values
    if perc_values is None:
        return kpi_dict, performance
    if performance <= perc_values[0]:
        return kpi_dict, 0.0
    elif performance >= perc_values[-1]:
        return kpi_dict, 1.0
    for i in range(1, len(perc_values)):
        if performance <= perc_values[i]:
            lower_bound = perc_values[i-1]
            upper_bound = perc_values[i]
            if upper_bound == lower_bound:
                return kpi_dict, (i-1)/10 + 0.05
            else:
                return kpi_dict, (i-1)/10 + (performance - lower_bound) / (upper_bound - lower_bound) * 0.1
    return performance

def create_kpi_normalizer(commons: list[Common]):
    global perc_values, performance_dict
    for common in commons:
        if common.conf.just_prediction:
            return
        Common.set_instance(common)
        for _, df in common.train_df.groupby(common.conf.event_log_specs.case_id):
            compute_kpi(df=df)
    perc_values = np.percentile([tup[1] for tup in performance_dict.values()], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    performance_dict = {k: normalize(v) for k, v in performance_dict.items()}


def compute_kpi(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    global performance_dict
    common = Common.instance
    if common.conf.just_prediction:
        return {}, 0.5
    case_id = df[common.conf.event_log_specs.case_id].iloc[0]
    if case_id in performance_dict:
        return performance_dict[case_id]
    kpi_dict = {}
    kpi_weights = {}
    normalized_last_row = common.future_df[common.future_df[common.conf.event_log_specs.case_id] == df[common.conf.event_log_specs.case_id].iloc[0]].iloc[0]
    if common.conf.custom_performance_function:
        original_df = common.conf.df.loc[df.index]
        performance = common.conf.custom_performance_function(original_df, df, normalized_last_row)
        performance = max(0, min(1, performance))
        kpi_dict['custom'] = performance
        performance_dict[case_id] = (kpi_dict, performance)
        return normalize((kpi_dict, performance))
    if common.conf.performance_weights.trace_length:
        kpi_dict[TRACE_LENGTH] = 1 - normalized_last_row[TRACE_LENGTH]
        kpi_weights[TRACE_LENGTH] = common.conf.performance_weights.trace_length
    if common.conf.performance_weights.trace_duration:
        kpi_dict[TRACE_DURATION] = 1 - normalized_last_row[TRACE_DURATION]
        kpi_weights[TRACE_DURATION] = common.conf.performance_weights.trace_duration
    for k, v in common.conf.performance_weights.numerical_trace_attributes.items():
        if 'min' in v:
            kpi_dict[k] = 1 - normalized_last_row[k]
        elif 'max' in v:
            kpi_dict[k] = normalized_last_row[k]
        else:
            raise ValueError
        kpi_weights[k] = v[1]
    for k, v in common.conf.performance_weights.categorical_trace_attributes.items():
        value = v[0]
        name = f"{k}=={value}"
        kpi_dict[name] = 1 if normalized_last_row[k] == value else 0
        kpi_weights[name] = v[1]
    for k, v in common.conf.performance_weights.numerical_event_attributes.items():
        for v2 in v:
            if 'sum' in v2:
                t = CUMSUM
            elif 'avg' in v2:
                t = CUMAVG
            else:
                raise ValueError
            name = f"{k}{t}"
            if 'min' in v2:
                kpi_dict[name] = 1 - normalized_last_row[name]
            elif 'max' in v2:
                kpi_dict[name] = normalized_last_row[name]
            else:
                raise ValueError
            kpi_weights[name] = v2[2]
    performance = max(0, min(1, sum([kpi_dict[k] * kpi_weights[k] for k in kpi_dict.keys()])))
    performance_dict[case_id] = (kpi_dict, performance)
    return normalize((kpi_dict, performance))