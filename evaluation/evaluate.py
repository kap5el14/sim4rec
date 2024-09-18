from util.common import *
from pandas.errors import SettingWithCopyWarning
from evaluation.recommendation_test import recommend

def evaluate(folds):
    recommend(folds)
