from util.constants import *
from util.obj import EventLogSpecs, SimilarityWeights, PerformanceWeights, OutputFormat
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
import random
import dill
import heapq
from collections import deque
from sklearn.model_selection import KFold
import pprint
import tqdm
import warnings
import cProfile
import itertools
from copy import deepcopy



@dataclass
class Common:
    name: str = field(init=True, default=None)
    event_log_specs: EventLogSpecs = field(init=True, default=None)
    similarity_weights: SimilarityWeights = field(init=True, default=None)
    performance_weights: PerformanceWeights = field(init=True, default=None)
    output_format: OutputFormat = field(init=True, default=None)
    original_df: pd.DataFrame = field(init=True, default=None)
    train_df: pd.DataFrame = field(init=True, default=None)
    test_df: pd.DataFrame = field(init=True, default=None)
    output_columns: list[str] = field(init=False, default=None)
    preprocess: Callable[[pd.DataFrame, bool], pd.DataFrame] = field(init=False, default=None)
    instance: 'Common' = None

    @classmethod
    def set_instance(cls, instance: 'Common'):
        cls.instance = instance

    def __post_init__(self):
        def create_add_attributes():
            numerical_event_attributes = set(self.similarity_weights.numerical_event_attributes.keys()).union(self.performance_weights.numerical_event_attributes.keys())
            def add_attributes(df: pd.DataFrame) -> pd.DataFrame:
                grouped = df.groupby(self.event_log_specs.case_id)
                THRESHOLD = 1e-10
                df[INDEX] = df.groupby(self.event_log_specs.case_id).cumcount()
                for attr in numerical_event_attributes:
                    df[f'{attr}{CUMSUM}'] = grouped[attr].cumsum().apply(lambda y: y if abs(y) > THRESHOLD else 0)
                    df[f'{attr}{CUMAVG}'] = grouped[attr].transform(lambda x: x.expanding().mean()).apply(lambda y: y if abs(y) > THRESHOLD else 0)
                    df[f'{attr}{MW_SUM}'] = grouped[attr].transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).sum()).apply(lambda y: y if abs(y) > THRESHOLD else 0)
                    df[f'{attr}{MW_AVG}'] = grouped[attr].transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).mean()).apply(lambda y: y if abs(y) > THRESHOLD else 0)
                df[self.event_log_specs.timestamp] = (df[self.event_log_specs.timestamp] - pd.Timestamp("1970-01-01")).dt.total_seconds()
                df[TIME_FROM_TRACE_START] = grouped[self.event_log_specs.timestamp].transform(lambda x: x - x.min())
                df[TIME_FROM_PREVIOUS_EVENT] = grouped[self.event_log_specs.timestamp].diff().fillna(0)
                df[ACTIVITY_OCCURRENCE] = df.groupby([self.event_log_specs.case_id, self.event_log_specs.activity]).cumcount() + 1
                df[TRACE_START] = grouped[self.event_log_specs.timestamp].transform('min')
                def expanding_unique_count(series):
                    unique_counts = []
                    seen = set()
                    for value in series:
                        seen.add(value)
                        unique_counts.append(len(seen))
                    return unique_counts
                df[UNIQUE_ACTIVITIES] = grouped[self.event_log_specs.activity].transform(expanding_unique_count)
                df[ACTIVITIES_MEAN] = grouped[ACTIVITY_OCCURRENCE].transform(lambda x: x.expanding().mean())
                df[ACTIVITIES_STD] = grouped[ACTIVITY_OCCURRENCE].transform(lambda x: x.expanding().std().fillna(0))
                return df
            return add_attributes
        def create_normalizer():
            def create_normalizer_with_percentiles(attr, perc_values):
                def normalize(row):
                    if row[attr] <= perc_values[0]:
                        row[attr] = 0.0
                        return row
                    elif row[attr] >= perc_values[-1]:
                        row[attr] = 1.0
                        return row
                    for i in range(1, len(perc_values)):
                        if row[attr] <= perc_values[i]:
                            lower_bound = perc_values[i-1]
                            upper_bound = perc_values[i]
                            if upper_bound == lower_bound:
                                row[attr] = (i-1)/10 + 0.05
                                return row
                            else:
                                row[attr] = (i-1)/10 + (row[attr] - lower_bound) / (upper_bound - lower_bound) * 0.1
                                return row
                return normalize
            def create_activity_occurrences_normalizer():
                unique_activities = self.train_df[self.event_log_specs.activity].unique()
                normalizers = {}
                for activity in unique_activities:
                    activity_mask = self.train_df[self.event_log_specs.activity] == activity
                    activity_data = self.train_df.loc[activity_mask, ACTIVITY_OCCURRENCE]
                    perc_values = np.percentile(activity_data, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) 
                    normalizers[activity] = create_normalizer_with_percentiles(ACTIVITY_OCCURRENCE, perc_values)
                def normalize(row):
                    if row[self.event_log_specs.activity] in normalizers:
                        return normalizers[row[self.event_log_specs.activity]](row)
                    row[ACTIVITY_OCCURRENCE] = np.nan
                    return row
                return normalize
            def create_timestamp_normalizer(attr):
                min_val = self.train_df[attr].min()
                max_val = self.train_df[attr].max()
                def normalize(row):
                    if max_val - min_val == 0:
                        row[attr] = 0.5
                        return row
                    if row[attr] <= min_val:
                        row[attr] = 0
                        return row
                    if row[attr] >= max_val:
                        row[attr] = 1
                        return row
                    row[attr] = (row[attr] - min_val) / (max_val - min_val)
                    return row
                return normalize
            attribute_normalizers = {}
            numerical_event_attributes = set(self.similarity_weights.numerical_event_attributes.keys()).union(self.performance_weights.numerical_event_attributes.keys())
            for attr in [TIME_FROM_TRACE_START, TIME_FROM_PREVIOUS_EVENT, INDEX, UNIQUE_ACTIVITIES, ACTIVITIES_MEAN, ACTIVITIES_STD] + list(itertools.chain.from_iterable([[attr, f'{attr}{CUMSUM}', f'{attr}{CUMAVG}', f'{attr}{MW_SUM}', f'{attr}{MW_AVG}'] for attr in numerical_event_attributes])):
                perc_values = np.percentile(self.train_df[attr], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                attribute_normalizers[attr] = create_normalizer_with_percentiles(attr, perc_values)
            numerical_trace_attributes = set(self.similarity_weights.numerical_trace_attributes.keys()).union(self.performance_weights.numerical_trace_attributes.keys())
            for attr in numerical_trace_attributes:
                perc_values = np.percentile(self.train_df.groupby(self.event_log_specs.case_id).first()[attr], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                attribute_normalizers[attr] = create_normalizer_with_percentiles(attr, perc_values)
            attribute_normalizers[ACTIVITY_OCCURRENCE] = create_activity_occurrences_normalizer()
            for attr in [TRACE_START, self.event_log_specs.timestamp]:
                attribute_normalizers[attr] = create_timestamp_normalizer(attr)
            def normalize(df: pd.DataFrame) -> pd.DataFrame:
                def normalize_row(row: pd.DataFrame):
                    new_row = row.copy()
                    for attr, normalizer in attribute_normalizers.items():
                        new_row = normalizer(new_row)
                    return new_row
                return df.apply(normalize_row, axis=1)
            return normalize
        add_attributes = create_add_attributes()
        self.train_df = add_attributes(self.train_df)
        normalize = create_normalizer()
        self.train_df = normalize(self.train_df)
        def create_preprocessor(add_attributes, normalize):
            def preprocess(df: pd.DataFrame):
                if df is None:
                    return
                return normalize(add_attributes(df))
            return preprocess
        self.preprocess = create_preprocessor(add_attributes, normalize)
        self.test_df = self.preprocess(self.test_df)
        with open(f'data/preprocessed/{self.name}.pkl', 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, name: str):
        with open(f'data/preprocessed/{name}.pkl', 'rb') as f:
            return dill.load(f)

