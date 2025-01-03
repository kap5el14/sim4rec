from common import *


def generate_evaluation_datasets(conf: Configuration) -> list[Common]:
    grouped = conf.df.groupby(conf.event_log_specs.case_id)[conf.event_log_specs.timestamp]
    trace_start = grouped.min()
    trace_end = grouped.max()
    trace_boundaries = pd.DataFrame({
        TRACE_START: trace_start,
        TRACE_END: trace_end
    })
    commons = []
    for i, training_period in tqdm.tqdm(enumerate(conf.evaluation_datasets_format.training_periods), "Generating training-testing set pairs"):
        start = training_period[0]
        end = training_period[1]
        train_case_ids = trace_boundaries[(trace_boundaries[TRACE_START] >= start) & (trace_boundaries[TRACE_END] <= end)].index.unique()
        train_case_ids = random.sample(list(train_case_ids), min(conf.evaluation_datasets_format.training_size, len(train_case_ids)))
        train_df = conf.df[conf.df[conf.event_log_specs.case_id].isin(train_case_ids)]
        print(f"\n{len(train_case_ids)} traces in training set {i}.")
        if conf.evaluation_datasets_format.starts_after:
            test_case_ids = trace_boundaries[trace_boundaries[TRACE_START] > end].index.unique()
            test_case_ids = sorted(test_case_ids, key=lambda x: trace_start[x])
            test_case_ids = test_case_ids[:min(conf.evaluation_datasets_format.testing_size, len(test_case_ids))]
            test_df = conf.df[conf.df[conf.event_log_specs.case_id].isin(test_case_ids)]
        else:
            test_case_ids = trace_boundaries[(trace_boundaries[TRACE_START] <= end) & (trace_boundaries[TRACE_END] > end)].index.unique()
            test_case_ids = random.sample(list(test_case_ids), min(conf.evaluation_datasets_format.testing_size, len(test_case_ids)))
            test_df = conf.df[conf.df[conf.event_log_specs.case_id].isin(test_case_ids)]
        print(f"{len(test_case_ids)} traces in testing set {i}.")
        if not train_case_ids or not test_case_ids:
            continue
        common = Common(conf=conf, train_df=train_df, test_df=test_df, training_period=training_period)
        commons.append(common)
    return commons