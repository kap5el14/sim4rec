from common import Common
from algo.performance import KPIUtils
from algo.recommendation import RecommendationUtils

def synchronize(instance: Common):
    Common.instance = instance
    KPIUtils.instance = KPIUtils(instance)
    RecommendationUtils.instance = RecommendationUtils(instance)
    