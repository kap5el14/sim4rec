from common import *
import sys
from pandas.errors import SettingWithCopyWarning
from evaluation.datasets import generate_evaluation_datasets
from algo.pipeline import Pipeline
from util.synchronize import synchronize

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
np.seterr(invalid='ignore')

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
    plot_dir_path = os.path.join('evaluation_results', NAME)
    os.makedirs(plot_dir_path, exist_ok=True)
    old_plots = glob.glob(os.path.join(plot_dir_path, '*.svg'))
    for old_plot in old_plots:
        os.remove(old_plot)
    test_function(commons)
else:
    synchronize(commons[0])
    while True:
        user_input = input("Provide the name of the CSV file containing your trace and, optionally, the preferred number of recommendations (integer or '-all'; default=1). Alternatively, type 'q' to quit:\n")
        if user_input.lower() == 'q':
            print("User exited the program.")
            break
        else:
            try:
                inputs = user_input.split()
                df = pd.read_csv(os.path.join('user_files', 'traces', NAME, f'{inputs[0]}.csv'))
            except Exception as e:
                print(e)
            else:
                print("You provided the following dataframe:")
                print(df)
                if len(inputs) == 1:
                    recommendation = Pipeline(df=df).get_best_recommendation(interactive=True)
                    if recommendation:
                        print("\nJSON has been copied to clipboard.")
                    else:
                        print("No recommendation could be made.")
                elif inputs[1] == '-all':
                    recommendations = Pipeline(df=df).get_all_recommendations(interactive=True)
                else:
                    try:
                        number_of_recs = int(inputs[1])
                    except Exception as e:
                        print(e)
                        continue
                    recommendations = Pipeline(df=df).get_n_recommendations(n=number_of_recs, interactive=True)
                    if recommendations:
                        print("\nJSON has been copied to clipboard.")
                    else:
                        print("No recommendation could be made.")

                    
                    
        