from util.common import *
import sys
from pandas.errors import SettingWithCopyWarning
from evaluation.datasets import generate_evaluation_datasets
from algo.pipeline import recommendation_pipeline

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

NEW = '-n' in sys.argv
EVALUATION = '-e' in sys.argv
NAME = next((arg for arg in sys.argv[1:] if arg not in ['-e', '-n']), None)
def get_pkl_files(evaluation=EVALUATION):
    return list(sorted(glob.glob(os.path.join(Configuration.get_directory(NAME, evaluation), '*.pkl'))))
if NEW:
    for pkl_file in get_pkl_files(EVALUATION):
        os.remove(pkl_file)
commons = []
if not NEW:
    for pkl_file in get_pkl_files():
        commons.append(Common.deserialize(pkl_file))
else:
    conf = Configuration(NAME)
    if EVALUATION:
        commons = generate_evaluation_datasets(conf)
    else:
        commons.append(Common(conf=conf, train_df=conf.df))
    for i, fold in enumerate(commons):
        fold.serialize(os.path.join(Configuration.get_directory(NAME, EVALUATION), f'{i}.pkl'))
if EVALUATION:
    path = os.path.join('user_files', 'tests', f'{NAME}.py')
    if not os.path.isfile(path):
        raise ModuleNotFoundError(f"{path} not found. The user has to specify a custom test module.")
    spec = importlib.util.spec_from_file_location("custom_test_module", path)
    custom_test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_test_module)
    if not hasattr(custom_test_module, 'evaluate'):
        raise ValueError(f"Function 'evaluate' not found in {path}")
    test_function = getattr(custom_test_module, 'evaluate')
    plot_dir_path = os.path.join('evaluation_results', 'edu')
    os.makedirs(plot_dir_path, exist_ok=True)
    old_plots = glob.glob(os.path.join(plot_dir_path, '*.svg'))
    for old_plot in old_plots:
        os.remove(old_plot)
    test_function(commons)
else:
    Common.set_instance(commons[0])
    while True:
        user_input = input("Provide the name of the CSV file containing your trace or type 'q' to quit:\n")
        if user_input.lower() == 'q':
            print("User exited the program.")
            break
        else:
            try:
                df = pd.read_csv(os.path.join('user_files', 'traces', NAME, f'{user_input}.csv'))
            except Exception as e:
                print(e)
            else:
                print("You provided the following dataframe:")
                print(df)
                recommendation = recommendation_pipeline(df=df)
                if recommendation:
                    print(recommendation)
                    print("\nJSON has been copied to clipboard.")
                else:
                    print("No recommendation could be made.")
                
        