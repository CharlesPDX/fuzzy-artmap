from collections import Counter
from datetime import datetime
from math import sqrt

import numpy as np
import numpy.typing as npt
import pytest
import torch

from fuzzy_artmap.fuzzy_artmap import FuzzyArtMap
from fuzzy_artmap.procedural_fuzzy_artmap import FuzzyArtMap as procedural_fam

number_of_training_patterns = 1_000
number_of_test_patterns = 1_000
size_of_square = 1;                         # Size of square
out_class = np.array([1., 0.])
in_class = np.array([0., 1.])

@pytest.fixture
def pytorch_circle_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    radius = size_of_square/sqrt(2*torch.pi);              # Radius of circle so it's half area of square
    x_center = 0.5
    y_center = 0.5                    # Centre of circle
    xs = x_center*torch.ones((1,number_of_training_patterns))
    ys = y_center*torch.ones((1,number_of_training_patterns))
    train_rng = torch.random.manual_seed(12345)
    training_patterns = torch.concatenate((xs,ys)) + 0.5-torch.rand((2, number_of_training_patterns))
    bmat = ((training_patterns[0,:]-x_center)**2 + (training_patterns[1,:]-y_center)**2) > radius**2
    bmat = torch.vstack((bmat.long(), 1-bmat.long()))

    rng = torch.random.manual_seed(12345)
    xs = x_center*torch.ones((1,number_of_test_patterns))
    ys = y_center*torch.ones((1,number_of_test_patterns))
    test_set = torch.concatenate((xs,ys)) + 0.5-torch.rand((2, number_of_test_patterns))
    test_truth = ((test_set[0,:]-x_center)**2 + (test_set[1,:]-y_center)**2) > radius**2
    test_truth = torch.vstack((test_truth.long(), 1-test_truth.long()))

    return (training_patterns, bmat, test_set, test_truth)

@pytest.fixture
def numpy_circle_data() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    r = size_of_square/sqrt(2*np.pi);              # Radius of circle so it's half area of square
    x_center = 0.5
    y_center = 0.5                    # Centre of circle
    xs = x_center*np.ones((1,number_of_training_patterns))
    ys = y_center*np.ones((1,number_of_training_patterns))
    train_rng = np.random.Generator(np.random.PCG64(12345))
    rng = np.random.Generator(np.random.PCG64(12345))
    training_patterns = np.concatenate((xs,ys)) + 0.5-train_rng.random((2, number_of_training_patterns))
    bmat = ((training_patterns[0,:]-x_center)**2 + (training_patterns[1,:]-y_center)**2) > r**2
    bmat = np.array((bmat, 1-bmat))

    xs = x_center*np.ones((1,number_of_test_patterns))
    ys = y_center*np.ones((1,number_of_test_patterns))
    test_set = np.concatenate((xs,ys)) + 0.5-rng.random((2, number_of_test_patterns))
    test_truth = ((test_set[0,:]-x_center)**2 + (test_set[1,:]-y_center)**2) > r**2
    test_truth = np.array((test_truth, 1-test_truth))
    return (training_patterns, bmat, test_set, test_truth)

def test_torch_circle_square(pytorch_circle_data) -> None:
    """Test the performance in terms of true and false positive rate using pytorch tensors"""
    x = FuzzyArtMap(baseline_vigilance = 0.0, committed_node_learning_rate=1.0)
    start_time = datetime.now()
    print(start_time)
    for i in range(number_of_training_patterns):
        test_input = torch.transpose(pytorch_circle_data[0][:, i, None], 0, 1)
        ground_truth = torch.transpose(pytorch_circle_data[1][:, i, None], 0, 1)
        complement_encoded_input = FuzzyArtMap.complement_encode(test_input)
        x.fit(complement_encoded_input, ground_truth)

    out_test_point = torch.tensor(([0.115, 0.948],))
    encoded_test_point = FuzzyArtMap.complement_encode(out_test_point)
    prediction = x.predict(encoded_test_point)
    assert np.all(prediction == out_class).item()
    

    in_test_point = torch.tensor(([0.262, 0.782],))
    encoded_test_point = FuzzyArtMap.complement_encode(in_test_point)
    prediction = x.predict(encoded_test_point)
    assert np.all(prediction == in_class).item()
    
    test_predictions = Counter()
    for i in range(number_of_test_patterns):
        test_input = torch.transpose(pytorch_circle_data[2][:, i, None], 0, 1)
        ground_truth = torch.transpose(pytorch_circle_data[3][:, i, None], 0, 1)
        complement_encoded_input = FuzzyArtMap.complement_encode(test_input)        
        prediction = torch.from_numpy(x.predict(complement_encoded_input))[0]
        correct = torch.all(prediction == ground_truth).item()
        test_predictions.update([correct])
    stop_time = datetime.now()
    assert test_predictions[True] == 963, "Expected 963 (96.3%) true positive rate"
    assert test_predictions[False] == 37, "Expected 37 (3.7%) false positive rate"
    assert torch.count_nonzero(x.get_weight_a()[:, 0] < 1, 0) == 20, "Expected 20 `in` classes"
    elapsed_time = (stop_time-start_time).total_seconds()
    assert elapsed_time < 0.2, "Expected testing to take less than 200ms"

def test_numpy_to_torch_circle_square(numpy_circle_data) -> None:
    """Test the performance in terms of true and false positive rate using numpy arrays converted to pytorch tensors."""
    x = FuzzyArtMap(baseline_vigilance = 0.0, committed_node_learning_rate=1.0)
    start_time = datetime.now()
    print(start_time)
    for i in range(number_of_training_patterns):
        test_input = torch.transpose(torch.from_numpy(numpy_circle_data[0])[:, i, None], 0, 1)
        ground_truth = torch.transpose(torch.from_numpy(numpy_circle_data[1])[:, i, None], 0, 1)
        complement_encoded_input = FuzzyArtMap.complement_encode(test_input)
        x.fit(complement_encoded_input, ground_truth)

    out_test_point = torch.tensor(([0.115, 0.948],))
    encoded_test_point = FuzzyArtMap.complement_encode(out_test_point)
    prediction = x.predict(encoded_test_point)
    assert np.all(prediction == out_class).item()

    in_test_point = torch.tensor(([0.262, 0.782],))
    encoded_test_point = FuzzyArtMap.complement_encode(in_test_point)
    prediction = x.predict(encoded_test_point)
    assert np.all(prediction == in_class).item()

    test_predictions = Counter()
    for i in range(number_of_test_patterns):
        test_input = torch.transpose(torch.from_numpy(numpy_circle_data[2])[:, i, None], 0, 1)
        ground_truth = torch.transpose(torch.from_numpy(numpy_circle_data[3])[:, i, None], 0, 1)
        complement_encoded_input = FuzzyArtMap.complement_encode(test_input)        
        prediction = torch.from_numpy(x.predict(complement_encoded_input))[0]
        correct = torch.all(prediction == ground_truth).item()
        test_predictions.update([correct])
    stop_time = datetime.now()
    
    assert test_predictions[True] == 954, "Expected 954 (95.4%) true positive rate"
    assert test_predictions[False] == 46, "Expected 46 (4.6%) false positive rate"
    assert torch.count_nonzero(x.get_weight_a()[:, 0] < 1, 0) == 16, "Expected 16 `in` classes"
    elapsed_time = (stop_time-start_time).total_seconds()
    assert elapsed_time < 0.2, "Expected testing to take less than 200ms"

def test_procedural_fuzzy_artmap_circle_square(numpy_circle_data) -> None:
    """Test the performance in terms of true and false positive rate using numpy arrays and a procedural implementation of Fuzzy ARTMAP."""
    x = procedural_fam(4, 10, rho_a_bar = 0.0)
    start_time = datetime.now()
    print(start_time)
    for i in range(number_of_training_patterns):
        test_input = np.transpose(numpy_circle_data[0][:, i, None])
        ground_truth = np.transpose(numpy_circle_data[1][:, i, None])
        complement_encoded_input = procedural_fam.complement_encode(test_input)
        x.train(complement_encoded_input, ground_truth)

    out_test_point = np.array(([0.115, 0.948],))
    encoded_test_point = procedural_fam.complement_encode(out_test_point)
    prediction = x.predict(encoded_test_point)
    assert np.all(prediction == out_class).item()

    in_test_point = np.array(([0.262, 0.782],))
    encoded_test_point = procedural_fam.complement_encode(in_test_point)
    prediction = x.predict(encoded_test_point)
    assert np.all(prediction == in_class).item()

    test_predictions = Counter()
    for i in range(number_of_test_patterns):
        test_input = np.transpose(numpy_circle_data[2][:, i, None])
        ground_truth = np.transpose(numpy_circle_data[3][:, i, None])
        complement_encoded_input = procedural_fam.complement_encode(test_input)        
        prediction = x.predict(complement_encoded_input)
        correct = np.all(prediction == ground_truth).item()
        test_predictions.update([correct])
    stop_time = datetime.now()

    assert test_predictions[True] == 954, "Expected 954 (95.4%) true positive rate"
    assert test_predictions[False] == 46, "Expected 46 (4.6%) false positive rate"
    assert np.count_nonzero(x.weight_a[:, 0] < 1, 0), "Expected 16 `in` classes"
    elapsed_time = (stop_time-start_time).total_seconds()
    assert elapsed_time < 0.2, "Expected testing to take less than 200ms"