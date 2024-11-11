from datetime import datetime
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Self, Tuple

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import numpy.typing as npt


class FuzzyArtMap(BaseEstimator):
    _validated = False
    _device = "cpu"
    _input_vector_size: int
    number_of_category_nodes: int
    _number_of_labels: int
    _choice_parameter = 0.001 # alpha > 0. Set small for the conservative limit (Fuzzy AM paper, Sect.3)
    def __init__(self,
                 baseline_vigilance: float = 0.0,
                 learning_rate: float = 1.0,
                 map_field_learning_rate: float = 1.0,
                 map_field_vigilance: float = 0.95,
                 max_nodes: int = None,
                 committed_node_learning_rate: float = 0.75,
                 debugging: bool = False,
                 auto_scale: bool = False,
                 auto_complement_encode: bool = False,
                 number_of_category_nodes = 10,
                 ):
        """
        Initialize Fuzzy ARTMAP instance
        Keyword arguments:
        f1_size - the number of elements in the compliment encoded vector (two-times the size of the input vector)
        f2_size - the initial number of coding category nodes - this will automatically grow up to max_nodes, unlimited if None
        number_of_labels - the compliment encoded number of labels for multi-label classification (e.g. for relevant, privilege this would be set to 4, for just relevant - 2)
        rho_a_bar - the baseline vigilance parameter to set the degree of match required to trigger resonance
        beta - Learning rate. Set to 1 for fast learning
        beta_ab - Map field learning rate, Enables Slow-learning mode from Carpenter, Grossberg, & Reynolds
        rho_ab - Map field vigilance in [0,1]
        max_nodes - the maximum number of f2 nodes to grow to (Implementing the Max-Nodes mode of Carpenter, Grossberg, & Reynolds, 1995 - "A Fuzzy ARTMAP Nonparametric Probability Estimator for Nonstationary Pattern Recognition Problems")
        committed_beta - the learning rate for nodes that have already been commited (see Fast-Commit Slow-Recode Option in Carpenter et al., 1992)
        debugging - Enables or disables bounds checking, and profiling; enabling may result in very poor execution time performance
        """
        self.learning_rate = learning_rate  
        self.map_field_learning_rate = map_field_learning_rate 
        self.map_field_vigilance = map_field_vigilance
        self.baseline_vigilance = baseline_vigilance  # rho a bar for ARTa, in range [0,1]
        self.committed_node_learning_rate = committed_node_learning_rate
        self.max_nodes = max_nodes
        self.debugging = debugging
        self.auto_scale = auto_scale
        self.auto_complement_encode = auto_complement_encode
        self.number_of_category_nodes = number_of_category_nodes


    def _set_defaults(self) -> None:
        self._range_validation_params = ["baseline_vigilance", "committed_node_learning_rate", "map_field_vigilance", "learning_rate"]
        self._vigilance_refinement_step = 0.001 # Fab mismatch raises ARTa vigilance to this much above what is needed to reset ARTa
        
        self._dtype = np.floating
        self._committed_nodes = set()
        self._updated_nodes = set()
        self._node_increase_step = 50 # number of F2 nodes to add when required
        self._number_of_increases = 0
        self._weight_a = np.ones((self.number_of_category_nodes, self._input_vector_size), device=self._device, dtype=self._dtype)
        self._input_vector_sum = self._input_vector_size / 2
        self._weight_ab = np.ones((self.number_of_category_nodes, self._number_of_labels), device=self._device, dtype=self._dtype)
        self._A_and_w = np.empty(self._weight_a.shape, device=self._device, dtype=self._dtype)
        self._validated = False

        logger.debug(f"f1_size: {self._input_vector_size}, f2_size:{self.number_of_category_nodes}, committed beta = {self.committed_node_learning_rate}")

    def _range_validation(self) -> None:
        for param_name in self._range_validation_params:
            value = getattr(self, param_name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{param_name} must be between 0.0 and 1.0, received {value}")

    @staticmethod
    def _vector_validation(vector: np.ndarray, vector_name: str) -> None:
        assert len(vector) == 1 or vector.shape[0] == 1, f"{vector_name} is not a 1d vector, {vector.shape}"

        any_value_over_one = (vector > 1.0).any()
        assert not any_value_over_one.item(), f"{vector_name} vector contains one or more values greater than 1.0"
        
        any_value_below_zero = (vector < 0.0).any()
        assert not any_value_below_zero.item(), f"{vector_name} vector contains one or more values less than 0.0"

        has_nan = np.any(np.isnan(vector))
        assert not has_nan, f"{vector_name} contains one or more NaN values"

        has_floating_point_values = np.issubdtype(vector.dtype, np.floating)
        assert has_floating_point_values, f"{vector_name} contains one or more non-floating point values"

    def set_params(self, parameters) -> Self:
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        return {
            # "input_vector_size": self.input_vector_size,
            "number_of_category_nodes" : self.number_of_category_nodes,
            # "number_of_labels": self.number_of_labels,
            "learning_rate": self.learning_rate,
            "map_field_learning_rate": self.map_field_learning_rate,
            "map_field_vigilance": self.map_field_vigilance,
            "baseline_vigilance": self.baseline_vigilance,
            "committed_node_learning_rate": self.committed_node_learning_rate,
            "max_nodes": self.max_nodes,
            "debugging": self.debugging,
            "auto_complement_encode": self.auto_complement_encode,
            "auto_scale": self.auto_scale,
        }

    def _resonance_search_vector(self, input_vector: np.ndarray, already_reset_nodes: List[int], rho_a: float) -> Tuple[int, float]:
        if self.debugging:
            FuzzyArtMap._vector_validation(input_vector, "Input")

        resonant_a = False
        number_of_f2_nodes, match_function, category_choice_function = self._calculate_category_choice(input_vector)
        
        category_choice_function[already_reset_nodes] = np.zeros((len(already_reset_nodes), ), dtype=self._dtype, device=self._device)
        has_evaluated_all_nodes = False
        while not resonant_a:
            category_index = np.argmax(category_choice_function)

            # check for resonance (the full expression of eq 7)
            if match_function[category_index].item() >= rho_a or math.isclose(match_function[category_index].item(), rho_a):
                resonant_a = True
                break
            else:
                resonant_a = False
                already_reset_nodes.append(category_index)
                category_choice_function[category_index.item()] = 0

            # Creating new nodes if we've reset all of them
            # or go into fixed node mode
            if len(already_reset_nodes) >= number_of_f2_nodes:
                if has_evaluated_all_nodes:
                    raise RuntimeError(f"Resonance A search failed twice, ensure values are in range [0.0, 1.0]")

                if self.max_nodes is None or self.max_nodes > (number_of_f2_nodes + self._node_increase_step):
                    self._weight_a = np.vstack((self._weight_a, np.ones((self._node_increase_step, self._weight_a.shape[1]), device=self._device, dtype=self._dtype)))
                    self._weight_ab = np.vstack((self._weight_ab, np.ones((self._node_increase_step, self._weight_ab.shape[1]), device=self._device, dtype=self._dtype)))
                    self._A_and_w = np.vstack((self._A_and_w, np.empty((self._node_increase_step, self._weight_a.shape[1]), device=self._device, dtype=self._dtype)))
                    self._number_of_increases += 1
                else:
                    self.map_field_vigilance = 0
                    self.map_field_learning_rate = 0.75
                    self.baseline_vigilance = 0
                    rho_a = self.baseline_vigilance
                    logger.warning(f"Maximum number of nodes reached, {len(already_reset_nodes)} - adjusting rho_ab to {self.map_field_vigilance} and beta_ab to {self.map_field_learning_rate}")
                    already_reset_nodes.clear()
                
                has_evaluated_all_nodes = True
                number_of_f2_nodes, match_function, category_choice_function = self._calculate_category_choice(input_vector)
                category_choice_function[already_reset_nodes] = np.zeros((len(already_reset_nodes), ), dtype=self._dtype, device=self._device)
                
        # return the selected category index j and the degree of membership
        return category_index.item(), match_function[category_index].item()

    def _calculate_category_choice(self, input_vector: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        This method calculates the category choice function for all Tj, which is
        The L1-norm of I fuzzy min wj divided by alpha plus the L1-norm of wj ->
        |I ^ wj| / α + |wj|
        See Carpenter et al. (1992) p.700 eq 2-4
        """

        number_of_f2_nodes = self._weight_a.shape[0]  # Count how many F2a nodes we have

        # This is the top/first term of the category choice function - I (the input vector) min weight j
        # except here it's done for all categories (j's)
        np.minimum(np.repeat(input_vector, number_of_f2_nodes, 0), self._weight_a, out=self._A_and_w) # Fuzzy AND = min
        
        # Calculate the L1 norm (eq 4), save this term since it's used for next to calculate T (the category choice value)
        # And for resonance/fuzzy membership
        category_choice_numerator = np.sum(self._A_and_w, 1) # Row vector of signals to F2 nodes
        
        # This is the left-hand term of the resonance check equation (eq 7) for all categories
        match_function = category_choice_numerator / self._input_vector_sum
        
        category_choice_function = category_choice_numerator / (self._choice_parameter + np.sum(self._weight_a, 1)) # Choice function vector for F2
        return number_of_f2_nodes, match_function, category_choice_function

    def _train(self, input_vector: np.ndarray, class_vector: np.ndarray) -> None:
        rho_a = self.baseline_vigilance # We start off with ARTa vigilance at baseline
        resonant_ab = False # Not resonating in the Fab match layer
        already_reset_nodes = [] # We haven't rest any ARTa nodes for this input pattern yet, maintain list between resonance searches of Fa
        
        if self.debugging:
            assert class_vector.shape[1] >= 2, "Class vectors must be compliment encoded - requires at least 2 columns"
            FuzzyArtMap._vector_validation(class_vector, "Class")

        class_vector_sum = np.sum(class_vector, 1)
        while not resonant_ab:            
            selected_category, match_function = self._resonance_search_vector(input_vector, already_reset_nodes, rho_a)
            
            map_field_activation = np.minimum(class_vector, self._weight_ab[selected_category, None])
            
            category_match_function = np.sum(map_field_activation, 1)/class_vector_sum

            if category_match_function > self.map_field_vigilance or math.isclose(category_match_function, self.map_field_vigilance):
                resonant_ab = True
            else: 
                already_reset_nodes.append(selected_category)
                rho_a = match_function + self._vigilance_refinement_step
                if rho_a > 1.0:
                    rho_a = 1.0 - self._vigilance_refinement_step

        self._updated_nodes.add(str(selected_category))
        if selected_category in self._committed_nodes:
            learning_rate = self.committed_node_learning_rate
        else:
            learning_rate = self.learning_rate

        self._weight_a[selected_category, None] = (learning_rate * np.minimum(input_vector, self._weight_a[selected_category, None])) + ((1-learning_rate) * self._weight_a[selected_category, None])
        self._weight_ab[selected_category, None] = (self.map_field_learning_rate * map_field_activation) + ((1-self.map_field_learning_rate) * self._weight_ab[selected_category, None])
        self._committed_nodes.add(selected_category)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        if self.auto_complement_encode:
            X, y= self._validate_data(X, y, accept_sparse=False, accept_large_sparse=False)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Length of samples tensor (X)-{X.shape[0]} does not equal length of labels tensor (y)-{y.shape[0]}")

        if self.auto_scale:
            X = self.scale_samples(X)
            y = self.scale_labels(y)

        if self.auto_complement_encode:
            X = FuzzyArtMap.complement_encode(X)
            y = FuzzyArtMap.complement_encode(y)

        if not self._validated:
            self._input_vector_size = X.shape[1]
            self._number_of_labels = y.shape[1]
            
            self._set_defaults()
            self._range_validation()
            self._validated = True

        input_vector: np.ndarray
        for vector_index, input_vector in enumerate(X):
            self._train(input_vector.reshape(1,-1), y[vector_index].reshape(1,-1))
        
        logger.debug(f"training updated: {','.join(self._updated_nodes)}")
        self._updated_nodes.clear()
        return self

    def scale_samples(self, samples: npt.ArrayLike) -> npt.ArrayLike:
        scaler = MinMaxScaler()
        samples = scaler.fit_transform(samples)
        samples = np.clip(samples, 0.0, 1.0)
        return samples
    
    def scale_labels(self, labels: npt.ArrayLike) -> npt.ArrayLike:
        if len(labels.shape) == 1:
            labels = labels.reshape(-1,1)
        if np.any(np.vectorize(lambda x: isinstance(x, str))(labels)):
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels)
        scaler = MinMaxScaler()
        labels = scaler.fit_transform(labels)
        labels = np.clip(labels, 0.0, 1.0)
        return labels

    @staticmethod
    def complement_encode(original_vector: np.ndarray, debug: bool = False) -> np.ndarray:
        if debug:
            FuzzyArtMap._vector_validation(original_vector, "Original")
        complement = 1-original_vector
        complement_encoded_value = np.hstack((original_vector, complement))
        return complement_encoded_value

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        rho_a = 0 # set ARTa vigilance to first match
        if self.auto_complement_encode:
            X = self._validate_data(X, accept_sparse=False, accept_large_sparse=False)
        
        if self.auto_scale:
            X = self.scale_samples(X)
        if self.auto_complement_encode:
            X = FuzzyArtMap.complement_encode(X)
        results = []
        for input_vector in X:
            J, _ = self._resonance_search_vector(input_vector.reshape(1,-1), [], rho_a)
            # (Called x_ab in Fuzzy ARTMAP paper)
            results.append(self._weight_ab[J, None]) # Fab activation vector
        return np.array(results)

    def predict_with_membership(self, input_vector: np.ndarray) -> Tuple[np.ndarray, float]:
        rho_a = 0 # set ARTa vigilance to first match
        J, membership_degree = self._resonance_search_vector(input_vector, [], rho_a)
        
        # (Called x_ab in Fuzzy ARTMAP paper)
        return self._weight_ab[J, None], membership_degree # Fab activation vector & fuzzy membership value

    def save_model(self, descriptor: str = None, path_prefix: str = "models") -> str:
        model_timestamp = datetime.now().isoformat().replace("-", "_").replace(":", "_").replace(".", "_")
        cleaned_descriptor = ""
        if descriptor:
            cleaned_descriptor = "_" + descriptor.replace("-", "_").replace(":", "_").replace(".", "_")
        model_path = f"fam_{model_timestamp}{cleaned_descriptor}.npz"
        if path_prefix:
             model_path = f"{path_prefix}/{model_path}"
        np.savez_compressed(model_path, weight_a=self._weight_a, weight_ab=self._weight_ab, committed_nodes=self._committed_nodes, parameters=self.get_params())
        return model_path
    
    def load_model(self, model_path: str) -> Self:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"`{model_path}` was not found or is a directory")
        logger.info(f"Loading model from {model_path}")
        data = np.load(model_path, allow_pickle=True)
        weight_a = data["weight_a"]
        weight_ab = data["weight_ab"]
        committed_nodes = data["committed_nodes"][None][0]
        parameters= data["parameters"][None][0]

        loaded_f1_size = weight_a.shape[1]
        loaded_f2_size = weight_a.shape[0]
        logger.debug(f"Parameter f2: {parameters['number_of_category_nodes']} - Actual f1: {loaded_f1_size}, f2: {loaded_f2_size}")
        parameters["input_vector_size"] = weight_a.shape[1]
        parameters["number_of_category_nodes"] = weight_a.shape[0]
        self.set_params(parameters)
        
        self._weight_a = weight_a
        self._weight_ab = weight_ab
        self._committed_nodes = committed_nodes
        
        logger.info("Model loaded")
        return self
    
    def get_number_of_nodes(self) -> int:
        return self._weight_ab.shape[0]

    def get_number_of_increases(self) -> int:
        return self._number_of_increases

    def get_increase_size(self) -> int:
        return self._node_increase_step
    
    def get_committed_nodes(self) -> str:
        return ",".join([str(n) for n in self._committed_nodes])
    
    def get_weight_a(self) -> np.ndarray:
        return self._weight_a
    
    def get_weight_ab(self) -> np.ndarray:
        return self._weight_ab
