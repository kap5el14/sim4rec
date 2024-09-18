from dataclasses import dataclass, field
import json

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

@dataclass
class PerformanceWeights:
    trace_length: float
    trace_duration: float
    numerical_trace_attributes: dict[str, float]
    categorical_trace_attributes: dict[str, float]
    numerical_event_attributes: dict[str, list]

@dataclass
class OutputFormat:
    numerical_attributes: list[str]
    categorical_attributes: list[str]
    timestamp_attributes: list[str]

    def __post_init__(self):
        self.numerical_attributes = list(set(self.numerical_attributes))
        self.categorical_attributes = list(set(self.categorical_attributes))
        self.timestamp_attributes = list(set(self.timestamp_attributes))

def read_conf(conf_file_path) -> tuple[str, EventLogSpecs, SimilarityWeights, PerformanceWeights]:
    with open(conf_file_path, 'r') as conf_file:
        config = json.load(conf_file)
        log_path = config['log_path']
        def try_read(strings: list[str], default=None):
            try:
                result = config
                for s in strings:
                    result = result[s]
            except Exception as e:
                result = default
            return result
        event_log_specs = EventLogSpecs(
            case_id=config['case_id'],
            activity=config['activity'],
            timestamp=config['timestamp']
        )
        similarity_weights = SimilarityWeights(
            activity=try_read(['similarity_weights', 'activity'], default=0),
            timestamp=try_read(['similarity_weights', 'timestamp'], default=0),
            numerical_event_attributes=try_read(['similarity_weights', 'numerical_event_attributes'], default={}),
            categorical_event_attributes=try_read(['similarity_weights', 'categorical_event_attributes'], default={}),
            numerical_trace_attributes=try_read(['similarity_weights', 'numerical_trace_attributes'], default={}),
            categorical_trace_attributes=try_read(['similarity_weights', 'categorical_trace_attributes'], default={}),
            trace_length=try_read(['similarity_weights', 'trace_length'], default=0)
        )
        performance_specs = PerformanceWeights(
            trace_length=try_read(['performance_weights', 'trace_length']),
            trace_duration=try_read(['performance_weights', 'trace_duration']),
            numerical_trace_attributes=try_read(['performance_weights', 'numerical_trace_attributes'], default={}),
            categorical_trace_attributes=try_read(['performance_weights', 'categorical_trace_attributes'], default={}),
            numerical_event_attributes=try_read(['performance_weights', 'numerical_event_attributes'], default={})
        )
        output_format = OutputFormat(
            numerical_attributes=try_read(['output_attributes', 'numerical'], default=[]),
            categorical_attributes=try_read(['output_attributes', 'categorical'], default=[]) + [event_log_specs.activity],
            timestamp_attributes=try_read(['output_attributes', 'timestamp'], default=[]) + [event_log_specs.timestamp]
        )
    return log_path, event_log_specs, similarity_weights, performance_specs, output_format
