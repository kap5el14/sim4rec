from util.common import *
import sys
from pandas.errors import SettingWithCopyWarning
from evaluation import *
from algo.pipeline import recommendation_pipeline

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

NEW = '-n' in sys.argv
EVALUATION = '-e' in sys.argv
NAME = next((arg for arg in sys.argv[1:] if arg not in ['-e', '-n']), None)
def get_pkl_files(evaluation=EVALUATION):
    return glob.glob(os.path.join(Configuration.get_directory(NAME, evaluation), '*.pkl'))
if NEW:
    for evaluation in [True, False]:
        for pkl_file in get_pkl_files(evaluation):
            os.remove(pkl_file)
folds = []
if not NEW:
    for pkl_file in get_pkl_files():
        folds.append(Common.deserialize(pkl_file))
else:
    conf = Configuration(NAME)
    if EVALUATION:
        k = 5
        case_ids = conf.df[conf.event_log_specs.case_id].unique()
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(case_ids):
            train_case_ids = case_ids[train_index]
            test_case_ids = case_ids[test_index]
            train_df = conf.df[conf.df[conf.event_log_specs.case_id].isin(train_case_ids)]
            test_df = conf.df[conf.df[conf.event_log_specs.case_id].isin(test_case_ids)]
            fold = Common(conf=conf, train_df=train_df, test_df=test_df)
            folds.append(fold)
    else:
        folds.append(Common(conf=conf, train_df=conf.df))
    for i, fold in enumerate(folds):
        fold.serialize(os.path.join(Configuration.get_directory(NAME, EVALUATION), f'{i}.pkl'))
if EVALUATION:
    evaluate(folds)
else:
    Common.set_instance(folds[0])
    while True:
        user_input = input("Provide the name of the CSV file containing your trace or type 'q' to quit:\n")
        if user_input.lower() == 'q':
            print("User exited the program.")
            break
        else:
            try:
                df = pd.read_csv(f'user_files/traces/{NAME}/{user_input}.csv')
            except Exception as e:
                print(e)
            else:
                print("You provided the following dataframe:")
                print(df)
                scenarios = recommendation_pipeline(df=df)
                for s in scenarios:
                    print(s)
                print("\nJSON has been copied to clipboard.")
                
        