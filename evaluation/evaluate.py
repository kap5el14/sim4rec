from util.common import *
from pandas.errors import SettingWithCopyWarning
from evaluation.recommendation_test import recommend
from evaluation.similarity_test import pearson_correlation

def evaluate(folds):
    #pearson_correlation(folds)
    recommend(folds)
