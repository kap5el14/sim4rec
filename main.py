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
    evaluate(commons)
else:
    Common.set_instance(commons[0])
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
                recommendation_package = recommendation_pipeline(df=df)
                print(recommendation_package)
                print("\nJSON has been copied to clipboard.")
                
        