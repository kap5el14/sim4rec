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

def read_conf(conf_file_path) -> tuple[str, EventLogSpecs, SimilarityWeights, PerformanceWeights]:
    with open(conf_file_path, 'r') as conf_file:
        config = json.load(conf_file)
        log_path = config['log_path']
        def try_read(strings: list[str], default=None):
            result = None
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
            activity=try_read(['similarity_weights', 'activity']),
            timestamp=try_read(['similarity_weights', 'timestamp']),
            numerical_event_attributes=try_read(['similarity_weights', 'numerical_event_attributes']),
            categorical_event_attributes=try_read(['similarity_weights', 'categorical_event_attributes']),
            numerical_trace_attributes=try_read(['similarity_weights', 'numerical_trace_attributes']),
            categorical_trace_attributes=try_read(['similarity_weights', 'categorical_trace_attributes']),
            trace_length=try_read(['similarity_weights', 'trace_length'])
        )
        performance_specs = PerformanceWeights(
            trace_length=try_read(['performance_weights', 'trace_length']),
            trace_duration=try_read(['performance_weights', 'trace_duration']),
            numerical_trace_attributes=try_read(['performance_weights', 'numerical_trace_attributes'], default={}),
            categorical_trace_attributes=try_read(['performance_weights', 'categorical_trace_attributes'], default={}),
            numerical_event_attributes=try_read(['performance_weights', 'numerical_event_attributes'], default={})
        )
    return log_path, event_log_specs, similarity_weights, performance_specs
