from util.common import *
from util.obj import read_conf
import sys
from pandas.errors import SettingWithCopyWarning
from evaluation import *

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

k = 5
folds = []
if len(sys.argv) > 1:
    log_path, event_log_specs, similarity_weights, performance_weights = read_conf(sys.argv[1])
    df = pd.read_csv(log_path)
    df[event_log_specs.timestamp] = pd.to_datetime(df[event_log_specs.timestamp])
    df.sort_values(by=[event_log_specs.case_id, event_log_specs.timestamp], inplace=True)
    case_ids = df[event_log_specs.case_id].unique()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_index = 0
    for train_index, test_index in kf.split(case_ids):
        train_case_ids = case_ids[train_index]
        test_case_ids = case_ids[test_index]
        train_df = df[df[event_log_specs.case_id].isin(train_case_ids)]
        test_df = df[df[event_log_specs.case_id].isin(test_case_ids)]
        folds.append(Common(name=f"fold_{fold_index}", event_log_specs=event_log_specs, similarity_weights=similarity_weights, performance_weights=performance_weights, original_df=df, train_df=train_df, test_df=test_df))
        fold_index += 1
else:
    for i in range(k):
        name = f"fold_{i}"
        folds.append(Common.load(name))

#plot_similarities()
#pearson_correlation(folds)
recommend(folds)