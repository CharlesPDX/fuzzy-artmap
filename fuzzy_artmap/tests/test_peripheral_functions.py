import pytest
import torch

from fuzzy_artmap.fuzzy_artmap import FuzzyArtMap


def test_range_validation() -> None:
    fuzzy_artmap = FuzzyArtMap(auto_complement_encode=True, auto_scale=True)
    fuzzy_artmap.fit([[1]], [[1]])
    for param in fuzzy_artmap._range_validation_params:
        # Test param too low < 0.0
        setattr(fuzzy_artmap, param, -1.1)
        with pytest.raises(ValueError):
            fuzzy_artmap._range_validation()
        
        # Test param too high > 1.0
        setattr(fuzzy_artmap, param, 2.1)
        with pytest.raises(ValueError):
            fuzzy_artmap._range_validation()
        setattr(fuzzy_artmap, param, 0.1)


def test_vector_validation() -> None:
    over_sized_vector = torch.tensor([[0.5, 1.2, 0.8],[1.5, 0.7, 0.9]])
    vector_greater_than_one = torch.tensor([[0.5, 1.2, 0.8]])
    vector_less_than_zero  = torch.tensor([[0.5, 0.2, -0.1]])
    nan_vector = torch.tensor([[0.5, 1.0, float("nan")]])
    plain_vector = torch.tensor([[0.5, 0.2, 0.1]])
    
    with pytest.raises(AssertionError):
        FuzzyArtMap._vector_validation(over_sized_vector, "oversized")
    
    with pytest.raises(AssertionError):
        FuzzyArtMap._vector_validation(vector_greater_than_one, "greater_than_one")
    
    with pytest.raises(AssertionError):
        FuzzyArtMap._vector_validation(vector_less_than_zero, "less_than_zero")
    
    with pytest.raises(AssertionError):
        FuzzyArtMap._vector_validation(nan_vector, "nan")
    
    try:
        FuzzyArtMap._vector_validation(plain_vector, "plain")
    except Exception as e:
        pytest.fail(f"Plain vector raised exception-{e}")


def test_complement_encoding() -> None:
    test_vector = torch.tensor([[0.5, 0.2, 0.1]])
    expected_vector = torch.tensor([[0.5, 0.2, 0.1, 0.5, 0.8, 0.9]])
    actual_vector = FuzzyArtMap.complement_encode(test_vector)
    assert torch.equal(expected_vector, actual_vector)


def test_complement_encoding_debug_raises() -> None:
    over_sized_vector = torch.tensor([[0.5, 1.2, 0.8],[1.5, 0.7, 0.9]])
    with pytest.raises(AssertionError):
        FuzzyArtMap.complement_encode(over_sized_vector, True)
