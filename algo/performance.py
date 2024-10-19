from common import *

class KPIUtils:
    instance: 'KPIUtils' = None
    def __init__(self, common: Common):
        self.stats_computed = False
        self.performance_dict = {}
        if common.conf.just_prediction:
            return
        for _, df in common.train_df.groupby(common.conf.event_log_specs.case_id):
            self.compute_kpi(df=df)
        self.perc_values = np.percentile([v[1] for v in self.performance_dict.values()], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        self.stats_computed = True

    def normalize(self, tup: tuple[dict[str, float], float]) -> tuple[dict[str, float], float]:
        (kpi_dict, performance) = tup
        if not self.stats_computed:
            return kpi_dict, max(0, min(1, performance))
        def _normalize():
            if performance <= self.perc_values[0]:
                return 0.0
            elif performance >= self.perc_values[-1]:
                return 1.0
            for i in range(1, len(self.perc_values)):
                if performance <= self.perc_values[i]:
                    lower_bound = self.perc_values[i-1]
                    upper_bound = self.perc_values[i]
                    if upper_bound == lower_bound:
                        return (i-1)/10 + 0.05
                    else:
                        return (i-1)/10 + (performance - lower_bound) / (upper_bound - lower_bound) * 0.1
            return 0.5
        return kpi_dict, max(0, min(1, _normalize()))

    def compute_kpi(self, df: pd.DataFrame) -> tuple[dict[str, float], float]:
        common = Common.instance
        if common.conf.just_prediction:
            return {}, 0.5
        case_id = df[common.conf.event_log_specs.case_id].iloc[0]
        if case_id in self.performance_dict:
            return self.performance_dict[case_id]
        kpi_dict = {}
        kpi_weights = {}
        normalized_last_row = common.future_df[common.future_df[common.conf.event_log_specs.case_id] == df[common.conf.event_log_specs.case_id].iloc[0]].iloc[0]
        if common.conf.custom_performance_function:
            original_df = common.get_original(df)[0]
            performance = common.conf.custom_performance_function(original_df, df, normalized_last_row)
            performance = max(0, min(1, performance))
            self.performance_dict[case_id] = (kpi_dict, performance)
            return self.normalize((kpi_dict, performance))
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
        for k, v in common.conf.performance_weights.activity_occurrences.items():
            occurrences = df[df['event'] == k][ACTIVITY_OCCURRENCE].iloc[-1] if k in df['event'].values else 0
            if v[0] == 'min':
                result = 1 - occurrences
            elif v[0] == 'max':
                result = occurrences
            else:
                raise ValueError
            kpi_dict[k] = result
            kpi_weights[k] = v[1]
        performance = max(0, min(1, sum([kpi_dict[k] * kpi_weights[k] for k in kpi_dict.keys()])))
        self.performance_dict[case_id] = (kpi_dict, performance)
        return self.normalize((kpi_dict, performance))