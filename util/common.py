import math
from util.constants import *
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
import hdbscan
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
import json
import glob
import importlib.util
import os


@dataclass
class EventLogSpecs:
    case_id: str
    activity: str
    timestamp: str

@dataclass
class SimilarityWeights:
    activity: float
    timestamp: float
    numerical_event_attributes: dict[str, float]
    categorical_event_attributes: dict[str, float]
    numerical_trace_attributes: dict[str, float]
    categorical_trace_attributes: dict[str, float]
    trace_length: float
    trace: float = field(init=False, default=None)
    event: float = field(init=False, default=None)

    def __post_init__(self):
        self.trace = self.activity / 2 + self.timestamp / 2 + sum(list(self.numerical_trace_attributes.values()) + list(self.categorical_trace_attributes.values())) + sum(list(self.numerical_event_attributes.values())) / 2 + self.trace_length
        self.event = self.activity / 2 + self.timestamp / 2 + sum(list(self.numerical_event_attributes.values())) / 2 + sum(list(self.categorical_event_attributes.values()))
        if not (0.95 <= self.trace + self.event <= 1.05):
            raise ValueError("Similarity weights don't sum up to 1!")

@dataclass
class PerformanceWeights:
    trace_length: float
    trace_duration: float
    numerical_trace_attributes: dict[str, float]
    categorical_trace_attributes: dict[str, float]
    numerical_event_attributes: dict[str, list]

    def get_all_numerical_attributes(self):
        res = list(self.numerical_trace_attributes.keys()) + [TRACE_LENGTH, TRACE_DURATION]
        for attr, _ in self.numerical_event_attributes.items():
            res.append(attr)
            res.append(f'{attr}{CUMSUM}')
            res.append(f'{attr}{CUMAVG}')
            res.append(f'{attr}{MW_SUM}')
            res.append(f'{attr}{MW_AVG}')
        return res
    
@dataclass
class OutputFormat:
    numerical_attributes: list[str]
    categorical_attributes: list[str]
    timestamp_attributes: list[str]
    activities: list[str]

    def __post_init__(self):
        self.numerical_attributes = list(set(self.numerical_attributes))
        self.categorical_attributes = list(set(self.categorical_attributes))
        self.timestamp_attributes = list(set(self.timestamp_attributes))

@dataclass
class EvaluationDatasetsFormat:
    training_size: int
    testing_size: int
    training_periods: list

    def __post_init__(self):
        new_training_periods = []
        for period in self.training_periods:
            start = pd.to_datetime(period['start'])
            end = pd.to_datetime(period['end'])
            new_training_periods.append((start, end))
        self.training_periods = new_training_periods


@dataclass
class Configuration:
    df: pd.DataFrame = field(init=False, default=None)
    similarity_weights: SimilarityWeights = field(init=False, default=None)
    performance_weights: PerformanceWeights = field(init=False, default=None)
    custom_performance_function: Callable[[pd.DataFrame, pd.DataFrame, pd.Series], float] = field(init=False, default=None)
    output_format: OutputFormat = field(init=False, default=None)
    evaluation_datasets_format: EvaluationDatasetsFormat = field(init=False, default=None)
    def __init__(self, name):
        self.name = name
        with open(os.path.join('user_files', 'conf', f"{name}.json"), 'r') as conf_file:
            config = json.load(conf_file)
            def try_read(strings: list[str], default=None):
                try:
                    result = config
                    for s in strings:
                        result = result[s]
                except Exception as e:
                    result = default
                return result
            self.event_log_specs = EventLogSpecs(
                case_id=config['case_id'],
                activity=config['activity'],
                timestamp=config['timestamp']
            )
            log_path = os.path.join('user_files', 'log', f'{name}.csv')
            self.df = pd.read_csv(log_path)
            self.df[self.event_log_specs.timestamp] = pd.to_datetime(self.df[self.event_log_specs.timestamp], format='mixed')
            self.df.sort_values(by=[self.event_log_specs.case_id, self.event_log_specs.timestamp], inplace=True)
            self.similarity_weights = SimilarityWeights(
                activity=try_read(['similarity_weights', 'activity'], default=0),
                timestamp=try_read(['similarity_weights', 'timestamp'], default=0),
                numerical_event_attributes=try_read(['similarity_weights', 'numerical_event_attributes'], default={}),
                categorical_event_attributes=try_read(['similarity_weights', 'categorical_event_attributes'], default={}),
                numerical_trace_attributes=try_read(['similarity_weights', 'numerical_trace_attributes'], default={}),
                categorical_trace_attributes=try_read(['similarity_weights', 'categorical_trace_attributes'], default={}),
                trace_length=try_read(['similarity_weights', 'trace_length'], default=0)
            )
            self.performance_weights = PerformanceWeights(
                trace_length=try_read(['performance_weights', 'trace_length']),
                trace_duration=try_read(['performance_weights', 'trace_duration']),
                numerical_trace_attributes=try_read(['performance_weights', 'numerical_trace_attributes'], default={}),
                categorical_trace_attributes=try_read(['performance_weights', 'categorical_trace_attributes'], default={}),
                numerical_event_attributes=try_read(['performance_weights', 'numerical_event_attributes'], default={})
                )
            if 'performance_weights' not in config:
                path = os.path.join('user_files', 'performance', f'{name}.py')
                if not os.path.isfile(path):
                    raise ModuleNotFoundError(f"{path} not found. The user has to specify either the performance weights or a custom performance function.")
                spec = importlib.util.spec_from_file_location("custom_performance_module", path)
                custom_performance_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_performance_module)
                if not hasattr(custom_performance_module, 'performance'):
                    raise ValueError(f"Function 'performance' not found in {path}")
                self.custom_performance_function = getattr(custom_performance_module, 'performance')
            self.output_format = OutputFormat(
                numerical_attributes=try_read(['output_attributes', 'numerical'], default=[]),
                categorical_attributes=try_read(['output_attributes', 'categorical'], default=[]) + [self.event_log_specs.activity],
                timestamp_attributes=try_read(['output_attributes', 'timestamp'], default=[]),
                activities=try_read(['output_attributes', 'activities'], default=list(self.df[self.event_log_specs.activity].unique()))
            )
            self.evaluation_datasets_format = EvaluationDatasetsFormat(
                training_size=try_read(['evaluation', 'training_size'], default=math.inf),
                testing_size=try_read(['evaluation', 'testing_size'], default=math.inf),
                training_periods=try_read(['evaluation', 'training_periods'], default=[])
            )

    @classmethod
    def get_directory(cls, name: str, evaluation: bool):
        suffix = 'eval' if evaluation else 'normal'
        return os.path.join('data', name, suffix)

@dataclass
class Common:
    conf: Configuration = field(init=True, default=None)
    train_df: pd.DataFrame = field(init=True, default=None)
    test_df: pd.DataFrame = field(init=True, default=None)
    future_train_df: pd.DataFrame = field(init=False, default=None)
    future_test_df: pd.DataFrame = field(init=False, default=None)
    add_attributes: Callable[[pd.DataFrame], pd.DataFrame] = field(init=False, default=None)
    normalize: Callable[[pd.DataFrame], pd.DataFrame] = field(init=False, default=None)
    future_normalize: Callable[[pd.DataFrame], pd.DataFrame] = field(init=False, default=None)
    instance: 'Common' = None

    @classmethod
    def set_instance(cls, instance: 'Common'):
        cls.instance = instance

    def __post_init__(self):
        def create_add_attributes():
            numerical_event_attributes = set(self.conf.similarity_weights.numerical_event_attributes.keys()).union(self.conf.performance_weights.numerical_event_attributes.keys())
            def add_attributes(df: pd.DataFrame) -> pd.DataFrame:
                if df is None:
                    return
                grouped = df.groupby(self.conf.event_log_specs.case_id)
                THRESHOLD = 1e-10
                df[INDEX] = df.groupby(self.conf.event_log_specs.case_id).cumcount()
                for attr in numerical_event_attributes:
                    df[f'{attr}{CUMSUM}'] = grouped[attr].transform(lambda x: x.expanding().sum()).apply(lambda y: y if y is not None and abs(y) > THRESHOLD else 0)
                    df[f'{attr}{CUMAVG}'] = grouped[attr].transform(lambda x: x.expanding().mean()).apply(lambda y: y if y is not None and abs(y) > THRESHOLD else 0)
                    df[f'{attr}{MW_SUM}'] = grouped[attr].transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).apply(lambda y: y[~pd.isnull(y)].sum() if not pd.isnull(y).all() else 0)).apply(lambda y: y if abs(y) > THRESHOLD else 0)
                    df[f'{attr}{MW_AVG}'] = grouped[attr].transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).apply(lambda y: y[~pd.isnull(y)].mean() if not pd.isnull(y).all() else 0)).apply(lambda y: y if abs(y) > THRESHOLD else 0)
                if pd.api.types.is_datetime64tz_dtype(df[self.conf.event_log_specs.timestamp]):
                    unix_epoch = pd.Timestamp("1970-01-01", tz='UTC')
                else:
                    unix_epoch = pd.Timestamp("1970-01-01")
                df[self.conf.event_log_specs.timestamp] = (df[self.conf.event_log_specs.timestamp] - unix_epoch).dt.total_seconds()
                df[TIME_FROM_TRACE_START] = grouped[self.conf.event_log_specs.timestamp].transform(lambda x: x - x.min())
                df[TIME_FROM_PREVIOUS_EVENT] = grouped[self.conf.event_log_specs.timestamp].diff().fillna(0)
                df[ACTIVITY_OCCURRENCE] = df.groupby([self.conf.event_log_specs.case_id, self.conf.event_log_specs.activity]).cumcount() + 1
                df[TRACE_START] = grouped[self.conf.event_log_specs.timestamp].transform('min')
                def expanding_unique_count(series):
                    unique_counts = []
                    seen = set()
                    for value in series:
                        seen.add(value)
                        unique_counts.append(len(seen))
                    return unique_counts
                df[UNIQUE_ACTIVITIES] = grouped[self.conf.event_log_specs.activity].transform(expanding_unique_count)
                df[ACTIVITIES_MEAN] = grouped[ACTIVITY_OCCURRENCE].transform(lambda x: x.expanding().mean())
                df[ACTIVITIES_STD] = grouped[ACTIVITY_OCCURRENCE].transform(lambda x: x.expanding().std().fillna(0))
                return df
            return add_attributes
        def create_normalizer(base_df):
            def create_normalizer_with_percentiles(attr, perc_values):
                def normalize(row):
                    if pd.isna(row[attr]):
                        return row
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
                    return row
                return normalize
            def create_activity_occurrences_normalizer():
                unique_activities = base_df[self.conf.event_log_specs.activity].unique()
                normalizers = {}
                for activity in unique_activities:
                    activity_mask = base_df[self.conf.event_log_specs.activity] == activity
                    activity_data = base_df.loc[activity_mask, ACTIVITY_OCCURRENCE]
                    perc_values = np.percentile(activity_data, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) 
                    normalizers[activity] = create_normalizer_with_percentiles(ACTIVITY_OCCURRENCE, perc_values)
                def normalize(row):
                    if row[self.conf.event_log_specs.activity] in normalizers:
                        return normalizers[row[self.conf.event_log_specs.activity]](row)
                    row[ACTIVITY_OCCURRENCE] = np.nan
                    return row
                return normalize
            def create_timestamp_normalizer(attr):
                min_val = base_df[attr].min()
                max_val = base_df[attr].max()
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
            numerical_event_attributes = set(self.conf.similarity_weights.numerical_event_attributes.keys()).union(self.conf.performance_weights.numerical_event_attributes.keys())
            for attr in [TIME_FROM_TRACE_START, TIME_FROM_PREVIOUS_EVENT, INDEX, UNIQUE_ACTIVITIES, ACTIVITIES_MEAN, ACTIVITIES_STD] + list(itertools.chain.from_iterable([[attr, f'{attr}{CUMSUM}', f'{attr}{CUMAVG}', f'{attr}{MW_SUM}', f'{attr}{MW_AVG}'] for attr in numerical_event_attributes])):
                perc_values = np.nanpercentile(base_df[attr], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                attribute_normalizers[attr] = create_normalizer_with_percentiles(attr, perc_values)
            numerical_trace_attributes = set(self.conf.similarity_weights.numerical_trace_attributes.keys()).union(self.conf.performance_weights.numerical_trace_attributes.keys())
            for attr in numerical_trace_attributes:
                perc_values = np.nanpercentile(base_df.groupby(self.conf.event_log_specs.case_id).first()[attr], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                attribute_normalizers[attr] = create_normalizer_with_percentiles(attr, perc_values)
            attribute_normalizers[ACTIVITY_OCCURRENCE] = create_activity_occurrences_normalizer()
            for attr in [TRACE_START, self.conf.event_log_specs.timestamp]:
                attribute_normalizers[attr] = create_timestamp_normalizer(attr)
            def normalize(df: pd.DataFrame) -> pd.DataFrame:
                if df is None:
                    return
                def normalize_row(row: pd.DataFrame):
                    new_row = row.copy()
                    for attr, normalizer in attribute_normalizers.items():
                        new_row = normalizer(new_row)
                    return new_row
                return df.apply(normalize_row, axis=1)
            return normalize
        self.add_attributes = create_add_attributes()
        self.train_df = self.add_attributes(self.train_df)
        self.future_train_df = self.train_df.groupby(self.conf.event_log_specs.case_id).last().reset_index()
        self.normalize = create_normalizer(self.train_df)
        self.future_normalize = create_normalizer(self.future_train_df)
        self.train_df = self.normalize(self.train_df)
        self.future_train_df = self.future_normalize(self.future_train_df)
        if self.test_df is not None:
            self.test_df = self.add_attributes(self.test_df)
            self.future_test_df = self.test_df.groupby(self.conf.event_log_specs.case_id).last().reset_index()
            self.test_df = self.normalize(self.test_df)
            self.future_test_df = self.future_normalize(self.future_test_df)

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df_with_attributes = self.add_attributes(df=df)
        normalized_df = self.normalize(df=df)
        normalized_last_row = self.future_normalize(df=df_with_attributes.tail(1)).iloc[0]
        return normalized_df, normalized_last_row

    def serialize(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            dill.dump(self, f)
    
    @classmethod
    def deserialize(cls, path) -> 'Common':
        with open(path, 'rb') as f:
            return dill.load(f)

