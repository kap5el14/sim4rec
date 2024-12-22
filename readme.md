#### What do I do with all this code?
The code in this repository can be run as a standalone program in order to make recommendations that are:
- individual (for a particular user),
- peer-based (based on similar users),
- next-step (regarding the next best course of action for a particular process execution),
- explainable (providing values along each quality dimension of a recommendation, supporting peers, and predictions on attribute values accompanying a particular activity/step recommendation).

#### What are the prerequisites to run it and what can I expect?
1. Make sure this solution suits your needs, i.e., you overlook a certain process and aim to enhance performance of individual process instances/executions/users.
2. Make sure you know what it means for a particular user to be high- or low-performing within your domain. You will have to specify it for this program; in particular, you will most likely need to provide a Python function that takes an incomplete trace of a running case (user, for which you want to make recommendations) as input, and returns a [0,1]-value on output.
3. Consider your priorities. Are you fine with your recommendations being based on outlier peers, as long as those outliers achieved high performance? Or do you want to obtain "safe" recommendations, i.e., ones that are broadly supported by the peers (and, thus, unlikely to cause major problems)? Maybe you need to give a certain boost to rare activities that are unlikely to be supported by many peers, in order to avoid over-specialization? Is your process highly-unstructured and it doesn't really matter much when a particular activity takes place, or do you need to ensure that the top recommendations really can be executed right now and don't necessitate being preceded by other activities? Finally, is it of importance to you that the top recommendations are based on events occurring in a similar context, i.e., that the recommendations are coherent? 

    Think about the questions above and assign appropriate weights to each quality dimension (also called optimization dimensions), with which the "goodness" of a recommendation is measured:
    - performance
    - support
    - novelty
    - timeliness
    - agreement on attributes

4. You may or may not be interested in getting additional predictions on attribute values that are likely to accompany an activity that was recommended if you decide to follow that recommendation. You will have to specify them in the configuration, otherwise only the timing will be predicted.

5. The recommendations are based on a peer-group, i.e., a set of historical partial traces (or trace prefixes) that are similar to the incomplete trace for which you want to obtain recommendations. However, this solution was designed to be as generic as possible and what it means for a pair of trace prefixes to be similar may well vary from one domain to another. Moreover, note that is often not enough just to compare the activity sequences or sets, as other information relevant w.r.t. this notion of similarity may be contained in the attribute values. You, as the domain-expert, are the one bearing the heavy responsibility of telling the program which attributes (you think) are "important" when it comes to two users/partial traces being similar. For example, consider traces consisting of course names and grades obtained in those courses. You will get a better idea of how similar two students are by also comparing their respective grades on top of what courses they took and in what order. Internally, the program then extracts a number of grade-related features that it incorporates into the similarity computation. This, however, does not interest you - what interests you that you have a vague idea that grades are somehow important.

6. You need a historical event log in .csv format (you can use ProM or Python to convert a file in .xes format into the .csv format).

7. The user for which you want to obtain recommendations (also called the running case) also needs to be provided in .csv format.

8. If everything goes as planned, you will get a ranking of activity recommendations together with their corresponding explanations. Yay.

#### Ok, so how do I configure this thing?
Follow these steps:
1. Provide a historical event log `<your_use_case_name>.csv` under `user_files/logs`.
2. Create a `<your_use_case_name>.json` configuration file under `user_files/confs` with the following fields:
```json
            {
                "case_id": <name of case_id column>,
                "activity": <name of activity column>,
                "timestamp": <name of timestamp column>,

                # at least one weight is necessary
                # weights should sum up to 1
                "similarity_weights": {
                    "activity": <weight>,
                    "trace_length": <weight>,
                    "timestamp": <weight>,
                    "numerical_event_attributes": {
                        <attr1>: <weight>,
                        <attr2>: <weight>,
                        <...>
                    },
                    "categorical_event_attributes": {
                        <attr1>: <weight>,
                        <attr2>: <weight>,
                        <...>
                    },
                    "numerical_trace_attributes": {
                        <attr1>: <weight>,
                        <attr2>: <weight>,
                        <...>
                    },
                    "categorical_trace_attributes": {
                        <attr1>: <weight>,
                        <attr2>: <weight>,
                        <...>
                    },
                },

                # at least one weight is necessary; otherwise, a custom performance function needs to be provided
                # weights should sum up to 1 (in case no performance function was provided)
                "performance_weights": {
                    "trace_length": <float>,
                    "trace_duration": <float>,
                    "numerical_trace_attributes": {
                        <attr1>: [<"min" or "max">, <float>],
                        <attr2>: [<"min" or "max">, <float>],
                        <...>
                    },
                    "categorical_trace_attributes": {
                        <attr1>: [<"min" or "max">, <float>],
                        <attr2>: [<"min" or "max">, <float>],
                        <...>
                    },
                    "numerical_event_attributes": {
                        <attr1>: [<"min" or "max">, <"avg" or "sum">, <float>],
                        <attr1>: [<"min" or "max">, <"avg" or "sum">, <float>],
                        <...>
                    },
                    "activity_occurrences": {
                        <label1>: [<"min" or "max">, <float>],
                        <label2>: [<"min" or "max">, <float>],
                        <...>
                    }
                },

                # at least one weight is necessary
                # weights should sum up to 1
                # if you don't want the default weights to be used, make sure to assign custom weights to all parameters here
                "optimize": {
                    "performance": <float>,
                    "support": <float>,
                    "timeliness": <float>,
                    "novelty": <float>,
                    "coherence": <float>
                },

                # optional, default=10
                "peer_group_size": <int>,

                # optional, default=inf
                "horizon": <int>,

                # optional
                "output_format": {

                    # default=none, specifies which attribute values will be predicted alongside the activity recommendations
                    "attributes": {
                        "numerical": [
                            <attr1>,
                            <attr2>,
                            <...>
                        ],
                        "categorical": [
                            <attr1>,
                            <attr2>,
                            <...>
                        ],
                        "timestamp": [
                            <attr1>,
                            <attr2>,
                            <...>
                        ]
                    },

                    # default=all, specifies which activities will be considered as viable recommendations
                    "activities": [
                        <label1>,
                        <label2>,
                        <...>
                    ]
                },
            }
```

If you don't want your performance function to be a simple linear combination of features, you need to provide a `<your_use_case_name>.py` module with a `def performance(original_df: pd.DataFrame, normalized_df: pd.DataFrame, normalized_last_row: pd.Series) -> float:` function under `user_files/performance`. An example can be seen here:
```python
import pandas as pd
from util.constants import *

def performance(original_df: pd.DataFrame, normalized_df: pd.DataFrame, normalized_last_row: pd.Series) -> float:
    occurrences_returns = normalized_df[normalized_df['event'] == 'Return ER'][ACTIVITY_OCCURRENCE].iloc[-1] if 'Return ER' in normalized_df['event'].values else 0
    if occurrences_returns:
        return 0
    performance = 1 - (normalized_last_row[TRACE_LENGTH] + normalized_last_row[TRACE_DURATION]) / 2
    if not (0 <= performance <= 1):
        raise ValueError(f"Performance={performance} not within the [0,1] range.")
    return performance
```
#### So can I finally run it?
Sure, here's how you do it:
1. You need to provide an (incomplete) trace of the running case for which you want to generate recommendations at `user_files/<your_use_case_name>/<some_identifier_of_the_running_case>.csv`. Include the column names in the beginning of the .csv file as well.
2. Run `python3 main.py <your_use_case_name> -n` if you are running with this configuration for the first time.
3. Otherwise, run `python3 main.py <your_use_case_name>`.
4. You will be prompted for something like `<some_identifier_of_the_running_case> <number>`, where `<number>` is the number of recommendations you want to generate.
5. The resulting recommendations will be printed out in the console. You can also access them by opening the `recommendation.html` file in your browser. The results in json format will also be copied to your clipboard.

#### How can I evaluate the recommendations?
1. You need to write your own test at `user_files/tests/<some_identifier_of_the_running_case>.py`. The module has to contain a `def evaluate(commons: list[Common]):` function.
2. Add necessary information to the configuration:
```json
"evaluation": {
        "training_size": <int>,
        "testing_size": <int>,
        "starts_after": <boolean>,
        "training_periods": [
            {
                "start": <timestamp>,
                "end": <timestamp>
            },
            {
                "start": <timestamp>,
                "end": <timestamp>
            },
            ...
        ]
}
```

If `starts_after` is on, then the program will only sample those traces for the testing set that started after the training period was closed. Otherwise, traces will be sampled that started before the end of the training period but haven't finished before then.
3. Run `python3 main.py <your_use_case_name> -e -n` if you are running the evaluation for the first time. 
4. Otherwise, run `python3 main.py <your_use_case_name> -e`.

#### Verbose Mode
If you want to run this program in verbose mode, you can (manually) set the `interactive` or `very_interactive` flag to `True` in `def __str__(self, n=None, interactive=True, very_interactive=True):` in the `pipeline.py` file.

#### Copyright
Copyright 2024 Kacper Kuca

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

