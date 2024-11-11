import os, sys
dir1 = os.path.abspath("")
if not dir1 in sys.path: 
    sys.path.append(dir1)
from math import sqrt
from collections import Counter
from datetime import datetime
import traceback

import numpy as np

from fuzzy_artmap import FuzzyArtMap

import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

number_of_training_patterns = 1_000
number_of_test_patterns = 1_000
size_of_square = 1;                         # Size of square
radius = size_of_square/sqrt(2*np.pi);              # Radius of circle so it's half area of square
x_center = 0.5
y_center = 0.5                    # Centre of circle
xs = x_center*np.ones((1,number_of_training_patterns))
ys = y_center*np.ones((1,number_of_training_patterns))
train_rng = np.random.Generator(np.random.PCG64(12345))
rng = np.random.Generator(np.random.PCG64(12345))
a = np.concatenate((xs,ys)) + 0.5-train_rng.random((2, number_of_training_patterns))
bmat = ((a[0,:]-x_center)**2 + (a[1,:]-y_center)**2) > radius**2
bmat = np.array((bmat, 1-bmat))

xs = x_center*np.ones((1,number_of_test_patterns))
ys = y_center*np.ones((1,number_of_test_patterns))
test_set = np.concatenate((xs,ys)) + 0.5-rng.random((2, number_of_test_patterns))
test_truth = ((test_set[0,:]-x_center)**2 + (test_set[1,:]-y_center)**2) > radius**2
test_truth = np.array((test_truth, 1-test_truth))


def get_traceback_string(e: Exception):
    if e is None:
        return "Passed exception is none!"
    if sys.version_info.minor >= 10:
        return ''.join(traceback.format_exception(e))
    else:
        return ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))


def main():
    x = FuzzyArtMap(baseline_vigilance = 0.0)
    start_time = datetime.now()
    print(start_time)
    for i in range(number_of_training_patterns):
        test_input = np.transpose(a[:, i, None])
        ground_truth = np.transpose(bmat[:, i, None])
        complement_encoded_input = FuzzyArtMap.complement_encode(test_input)
        x.fit(complement_encoded_input, ground_truth)

    out_test_point = np.array(([0.115, 0.948],))
    encoded_test_point = FuzzyArtMap.complement_encode(out_test_point)
    prediction = x.predict(encoded_test_point)
    print(prediction)

    in_test_point = np.array(([0.262, 0.782],))
    encoded_test_point = FuzzyArtMap.complement_encode(in_test_point)
    prediction = x.predict(encoded_test_point)
    print(prediction)

    test_predictions = Counter()
    for i in range(number_of_test_patterns):
        test_input = np.transpose(test_set[:, i, None])
        ground_truth = np.transpose(test_truth[:, i, None])
        complement_encoded_input = FuzzyArtMap.complement_encode(test_input)        
        prediction = x.predict(complement_encoded_input)[0]
        correct = np.all(prediction == ground_truth).item()
        test_predictions.update([correct])
    stop_time = datetime.now()
    print(f"elapsed: {stop_time-start_time}- {stop_time}")
    print(test_predictions)
    print(x.get_weight_a().shape)
    print(x.get_weight_ab().shape)
    print(np.count_nonzero(x.get_weight_a()[:, 0] < 1, 0))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        trace_back_string = get_traceback_string(e)
        LOGGER.error(f"Error <{e}>\ntraceback: {trace_back_string}")
   