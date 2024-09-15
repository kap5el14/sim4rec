from util.common import *
import json
import sys
import itertools
from tqdm import tqdm
from pandas.errors import SettingWithCopyWarning
from evaluation import *

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

if len(sys.argv) > 1:
    conf_file_path = sys.argv[1]
    with open(conf_file_path, 'r') as conf_file:
        config = json.load(conf_file)
        log_path = config['log_path']
        common = Common.get_instance()
        common.event_log_specs = EventLogSpecs(
            case_id=config['case_id'],
            activity=config['activity'],
            timestamp=config['timestamp']
        )
        common.similarity_weights = SimilarityWeights(
            activity=config['similarity_weights']['activity'],
            timestamp=config['similarity_weights']['timestamp'],
            numerical_event_attributes=config['similarity_weights']['numerical_event_attributes'],
            categorical_event_attributes=config['similarity_weights']['categorical_event_attributes'],
            numerical_trace_attributes=config['similarity_weights']['numerical_trace_attributes'],
            categorical_trace_attributes=config['similarity_weights']['categorical_trace_attributes'],
            trace_length=config['similarity_weights']['trace_length']
        )

    def create_add_attributes(common: Common):
        def add_attributes(df: pd.DataFrame) -> pd.DataFrame:
            grouped = df.groupby(common.event_log_specs.case_id)
            THRESHOLD = 1e-10
            df[INDEX] = df.groupby(common.event_log_specs.case_id).cumcount()
            for attr in common.similarity_weights.numerical_event_attributes.keys():
                df[f'{attr}{CUMSUM}'] = grouped[attr].cumsum().apply(lambda y: y if abs(y) > THRESHOLD else 0)
                df[f'{attr}{CUMAVG}'] = grouped[attr].transform(lambda x: x.expanding().mean()).apply(lambda y: y if abs(y) > THRESHOLD else 0)
                df[f'{attr}{MW_SUM}'] = grouped[attr].transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).sum()).apply(lambda y: y if abs(y) > THRESHOLD else 0)
                df[f'{attr}{MW_AVG}'] = grouped[attr].transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).mean()).apply(lambda y: y if abs(y) > THRESHOLD else 0)
            df[common.event_log_specs.timestamp] = (df[common.event_log_specs.timestamp] - pd.Timestamp("1970-01-01")).dt.total_seconds()
            df[TIME_FROM_TRACE_START] = grouped[common.event_log_specs.timestamp].transform(lambda x: x - x.min())
            df[TIME_FROM_PREVIOUS_EVENT] = grouped[common.event_log_specs.timestamp].diff().fillna(0)
            df[ACTIVITY_OCCURRENCE] = df.groupby([common.event_log_specs.case_id, common.event_log_specs.activity]).cumcount() + 1
            df[TRACE_START] = grouped[common.event_log_specs.timestamp].transform('min')
            def expanding_unique_count(series):
                unique_counts = []
                seen = set()
                for value in series:
                    seen.add(value)
                    unique_counts.append(len(seen))
                return unique_counts
            df[UNIQUE_ACTIVITIES] = grouped[common.event_log_specs.activity].transform(expanding_unique_count)
            df[ACTIVITIES_MEAN] = grouped[ACTIVITY_OCCURRENCE].transform(lambda x: x.expanding().mean())
            df[ACTIVITIES_STD] = grouped[ACTIVITY_OCCURRENCE].transform(lambda x: x.expanding().std().fillna(0))
            return df
        return add_attributes

    def create_normalizer():
        def create_normalizer_with_percentiles(attr, perc_values):
            def normalize(row):
                if row[attr] <= perc_values[0]:
                    row[attr] = 0.0
                    return row
                elif row[attr] >= perc_values[-1]:
                    row[attr] = 1.0
                    return row
                for i in range(1, len(perc_values)):
                    if row[attr] <= perc_values[i]:
                        lower_bound = perc_values[i-1]
                        upper_bound = perc_values[i]
                        if upper_bound == lower_bound:
                            row[attr] = (i-1)/10 + 0.05
                            return row
                        else:
                            row[attr] = (i-1)/10 + (row[attr] - lower_bound) / (upper_bound - lower_bound) * 0.1
                            return row
            return normalize
        def create_activity_occurrences_normalizer():
            unique_activities = common.train_df[common.event_log_specs.activity].unique()
            normalizers = {}
            for activity in unique_activities:
                activity_mask = common.train_df[common.event_log_specs.activity] == activity
                activity_data = common.train_df.loc[activity_mask, ACTIVITY_OCCURRENCE]
                perc_values = np.percentile(activity_data, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) 
                normalizers[activity] = create_normalizer_with_percentiles(ACTIVITY_OCCURRENCE, perc_values)
            def normalize(row):
                if row[common.event_log_specs.activity] in normalizers:
                    return normalizers[row[common.event_log_specs.activity]](row)
                row[ACTIVITY_OCCURRENCE] = np.nan
                return row
            return normalize
        def create_timestamp_normalizer(attr):
            min_val = common.train_df[attr].min()
            max_val = common.train_df[attr].max()
            def normalize(row):
                if max_val - min_val == 0:
                    row[attr] = 0.5
                    return row
                if row[attr] <= min_val:
                    row[attr] = 0
                    return row
                if row[attr] >= max_val:
                    row[attr] = 1
                    return row
                row[attr] = (row[attr] - min_val) / (max_val - min_val)
                return row
            return normalize
        attribute_normalizers = {}
        for attr in [TIME_FROM_TRACE_START, TIME_FROM_PREVIOUS_EVENT, INDEX, UNIQUE_ACTIVITIES, ACTIVITIES_MEAN, ACTIVITIES_STD] + list(itertools.chain.from_iterable([[attr, f'{attr}{CUMSUM}', f'{attr}{CUMAVG}', f'{attr}{MW_SUM}', f'{attr}{MW_AVG}'] for attr in common.similarity_weights.numerical_event_attributes.keys()])):
            perc_values = np.percentile(common.train_df[attr], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            attribute_normalizers[attr] = create_normalizer_with_percentiles(attr, perc_values)
        for attr in list(common.similarity_weights.numerical_trace_attributes.keys()):
            perc_values = np.percentile(common.train_df.groupby(common.event_log_specs.case_id).first()[attr], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            attribute_normalizers[attr] = create_normalizer_with_percentiles(attr, perc_values)
        attribute_normalizers[ACTIVITY_OCCURRENCE] = create_activity_occurrences_normalizer()
        for attr in [TRACE_START, common.event_log_specs.timestamp]:
            attribute_normalizers[attr] = create_timestamp_normalizer(attr)
        def normalize(df: pd.DataFrame) -> pd.DataFrame:
            def normalize_row(row: pd.DataFrame):
                new_row = row.copy()
                for attr, normalizer in attribute_normalizers.items():
                    new_row = normalizer(new_row)
                return new_row
            return df.apply(normalize_row, axis=1)
        return normalize

    df = pd.read_csv(log_path)
    df[common.event_log_specs.timestamp] = pd.to_datetime(df[common.event_log_specs.timestamp])
    df.sort_values(by=[common.event_log_specs.case_id, common.event_log_specs.timestamp], inplace=True)
    case_ids = df[common.event_log_specs.case_id].unique()
    train_case_ids, test_case_ids = train_test_split(case_ids, test_size=0.2, random_state=42)
    common.train_df = df[df[common.event_log_specs.case_id].isin(train_case_ids)]
    common.test_df = df[df[common.event_log_specs.case_id].isin(test_case_ids)]
    add_attributes = create_add_attributes(common)
    common.train_df = add_attributes(common.train_df)
    normalize = create_normalizer()
    common.train_df = normalize(common.train_df)
    def create_preprocessor(add_attributes, normalize):
        def preprocess(df: pd.DataFrame):
            return normalize(add_attributes(df))
        return preprocess
    common.preprocess = create_preprocessor(add_attributes, normalize)
    common.test_df = common.preprocess(common.test_df)
    with open('data/preprocessed/common.pkl', 'wb') as f:
        dill.dump(common, f)
else:
    with open('data/preprocessed/common.pkl', 'rb') as f:
        Common.set_instance(dill.load(f))

#plot_similarities()
pearson_correlation()