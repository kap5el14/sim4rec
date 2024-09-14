from util.constants import *
from util.obj import EventLogSpecs, SimilarityWeights
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
import random
import dill
import heapq
from collections import deque
from sklearn.model_selection import train_test_split
import pprint
import tqdm
import warnings
import cProfile


@dataclass
class Common:
    event_log_specs: EventLogSpecs = field(init=False, default=None)
    similarity_weights: SimilarityWeights = field(init=False, default=None)
    train_df: pd.DataFrame = field(init=False, default=None)
    test_df: pd.DataFrame = field(init=False, default=None)
    preprocess: Callable[[pd.DataFrame, bool], pd.DataFrame] = field(init=False, default=None)
    _instance: 'Common' = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_instance(cls, instance: 'Common'):
        cls._instance = instance
