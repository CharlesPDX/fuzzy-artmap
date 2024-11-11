from pathlib import Path
from unittest.mock import patch

import numpy as np

from fuzzy_artmap.fuzzy_artmap import FuzzyArtMap


def test_save_model():
    fam = FuzzyArtMap()
    fam._input_vector_size = 4
    fam._number_of_labels = 2
    fam._set_defaults()
    descriptor = "a-:b.c"
    with patch("numpy.savez_compressed") as mock_torch_save:
        saved_location = fam.save_model(descriptor)
        mock_torch_save.assert_called_once()
        assert len(mock_torch_save.call_args[1]) == 4

        assert mock_torch_save.call_args[1]["weight_a"] is fam._weight_a
        assert mock_torch_save.call_args[1]["weight_ab"] is fam._weight_ab
        assert mock_torch_save.call_args[1]["committed_nodes"] is fam._committed_nodes
        assert mock_torch_save.call_args[1]["parameters"] == fam.get_params()

        cleaned_descriptor = "a__b_c"
        assert cleaned_descriptor in saved_location
        assert "fam_" in saved_location
        assert saved_location[-4:] == ".npz"
        for character in ["-", ":", "."]:
            assert character not in saved_location[:-4]


def test_load_model():
    try:
        test_parameters = {
            "number_of_category_nodes" : 20,
            "learning_rate": 0.95,
            "map_field_learning_rate": 0.95,
            "map_field_vigilance": 0.99,
            "baseline_vigilance": 0.95,
            "committed_node_learning_rate": 0.6,
            "max_nodes": 20,
            "debugging": True,
            "auto_complement_encode": True,
            "auto_scale": True,
        }
        fam = FuzzyArtMap(**test_parameters)
        fam._input_vector_size = 4
        fam._number_of_labels = 2
        fam._set_defaults()

        committed_node = 1
        fam._committed_nodes.add(committed_node)
        test_a = np.random.random((2,2))
        test_ab = np.random.random((2,2))
        fam._weight_a = test_a
        fam._weight_ab = test_ab

        # easier to save off the model rather than fake a buffer
        saved_location = fam.save_model(path_prefix=None)        
        fam = FuzzyArtMap()
        fam._input_vector_size = 4
        fam._number_of_labels = 2
        fam._set_defaults()
        # validate that these have been reset by the previous assignment
        assert test_a.shape != fam._weight_a.shape
        assert test_ab.shape != fam._weight_ab.shape

        reloaded_fam = fam.load_model(saved_location)
        
        for parameter_name, parameter_value in reloaded_fam.get_params().items():
            # these parameters are overriden by the actual size of the weight_a tensor loaded from the model
            if parameter_name in ["input_vector_size", "number_of_category_nodes"]:
                continue
            assert test_parameters[parameter_name] == parameter_value
            assert getattr(reloaded_fam, parameter_name) == test_parameters[parameter_name]
        
        assert len(reloaded_fam._committed_nodes) == 1
        assert committed_node in reloaded_fam._committed_nodes

        assert np.all(np.array_equal(test_a, reloaded_fam._weight_a))
        assert np.all(np.array_equal(test_ab, reloaded_fam._weight_ab))

    finally:
        # cleanup saved model
        saved_path = Path(saved_location)
        saved_path.unlink(missing_ok=True)