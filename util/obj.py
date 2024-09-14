from dataclasses import dataclass, field

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