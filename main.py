from util.common import *
import json
import sys
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
        event_log_specs = EventLogSpecs(
            case_id=config['case_id'],
            activity=config['activity'],
            timestamp=config['timestamp']
        )
        similarity_weights = SimilarityWeights(
            activity=config['similarity_weights']['activity'],
            timestamp=config['similarity_weights']['timestamp'],
            numerical_event_attributes=config['similarity_weights']['numerical_event_attributes'],
            categorical_event_attributes=config['similarity_weights']['categorical_event_attributes'],
            numerical_trace_attributes=config['similarity_weights']['numerical_trace_attributes'],
            categorical_trace_attributes=config['similarity_weights']['categorical_trace_attributes'],
            trace_length=config['similarity_weights']['trace_length']
        )
        common = Common(event_log_specs=event_log_specs, similarity_weights=similarity_weights)

        df = pd.read_csv(log_path)
        df[self.event_log_specs.timestamp] = pd.to_datetime(df[self.event_log_specs.timestamp])
        df.sort_values(by=[self.event_log_specs.case_id, self.event_log_specs.timestamp], inplace=True)
        case_ids = df[self.event_log_specs.case_id].unique()
        train_case_ids, test_case_ids = train_test_split(case_ids, test_size=0.2, random_state=42)
        self.train_df = df[df[self.event_log_specs.case_id].isin(train_case_ids)]
        self.test_df = df[df[self.event_log_specs.case_id].isin(test_case_ids)]
    with open('data/preprocessed/common.pkl', 'wb') as f:
        dill.dump(common, f)
else:
    with open('data/preprocessed/common.pkl', 'rb') as f:
        Common.set_instance(dill.load(f))

#plot_similarities()
pearson_correlation()