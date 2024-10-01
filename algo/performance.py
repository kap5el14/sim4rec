from util.common import CUMAVG, CUMSUM, TRACE_DURATION, TRACE_LENGTH, Common, pd

performance_dict = {}

def compute_kpi(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    common = Common.instance
    case_id = df[common.conf.event_log_specs.case_id].iloc[0]
    if case_id in performance_dict:
        return performance_dict[case_id]
    kpi_dict = {}
    kpi_weights = {}
    normalized_last_row = common.future_df[common.future_df[common.conf.event_log_specs.case_id] == df[common.conf.event_log_specs.case_id].iloc[0]].iloc[0]
    if common.conf.custom_performance_function:
        original_df = common.conf.df.loc[df.index]
        custom_performance = common.conf.custom_performance_function(original_df, df, normalized_last_row)
        custom_performance = max(0, min(1, custom_performance))
        kpi_dict['custom'] = custom_performance
        performance_dict[case_id] = (kpi_dict, custom_performance)
        return kpi_dict, custom_performance
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
    return kpi_dict, performance