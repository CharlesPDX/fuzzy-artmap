from datetime import datetime
import logging
import math
from pathlib import Path
from typing import Any, Self

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import numpy.typing as npt
import torch

class FuzzyArtMap(BaseEstimator):
    _validated: bool = False
    _device: str | None = None
    _input_vector_size: int
    _number_of_labels: int
    _range_validation_params = ["baseline_vigilance", "committed_node_learning_rate", "map_field_vigilance", "learning_rate"]
    _dtype = torch.float

    def __init__(self, 
                 baseline_vigilance: float = 0.0,
                 learning_rate: float = 1.0,
                 committed_node_learning_rate: float = 0.75,
                 map_field_vigilance: float = 0.95,
                 map_field_learning_rate: float = 1.0,
                 max_nodes: int | None = None,
                 number_of_category_nodes: int = 10,
                 choice_parameter: float = 0.001,
                 vigilance_refinement_step: float = 0.001,
                 node_increase_step = 50,
                 use_cuda_if_available: bool = False,
                 online_learning: bool = True,
                 debugging: bool = False,
                 auto_scale: bool = False,
                 auto_complement_encode: bool = False
                 ):
        """
        Parameters
        ----------
        Primary parameters - The primary parameters are `baseline_vigilance` and `learning_rate`, the other parameters are provided as an interface to experiment
        and fine-tune the algorithm as necessary.

        baseline_vigilance: float = 0.0
            Degree of match required to trigger resonance, called rho a bar in Carpenter et al. (1992)
    
        learning_rate: float = 1.0
            How fast the network should learn, 1.0 for fast learning, 0.0 for no learning, called beta in Carpenter et al. (1992)
        
        These are additional refinement parameters
        
        committed_node_learning_rate: float = 0.75
            The slow recode learning rate, once a category is selected (see fast learning slow recode in Carpenter et al. (1992))
    
        map_field_vigilance: float = 0.95
            What is the degree of match between Fuzzy ART A and Fuzzy ART B (rho_ab in Carpenter et al. (1992))
    
        map_field_learning_rate: float = 1.0
            How fast does the map field between Fuzzy ART A and Fuzzy ART B learn (beta_ab in Carpenter et al. (1992))
    
        max_nodes: int | None = None
            The maximum number of category nodes (F2) to grow to 
            (Implementing the Max-Nodes mode of Carpenter, Grossberg, & Reynolds, 1995 - "A Fuzzy ARTMAP Nonparametric Probability Estimator for Nonstationary Pattern Recognition Problems")
    
        number_of_category_nodes: int = 10
            The initial number of coding category nodes (F2) - this will automatically grow up to max_nodes, unlimited if None
    

        These should rarely if ever need to be modified from default

        choice_parameter: float = 0.001
            Sets the degree of differentiation to choose between categories, must be greater than zero. 
            Set small for the conservative limit, called alpha in Carpenter et al. (1992) see section 3
    
        vigilance_refinement_step: float = 0.001
            When there is a category mismatch, how much to raise the vigilance parameter (rho_a, when there's an F_ab mismatch/category reset)
    

        Diagnostic, performance, and testing parameters
        
        node_increase_step = 50
            The number of category (F2) nodes to add when required.
            This is primarily a performance tweak, by allowing fewer memory allocations.
    
        use_cuda_if_available: bool = False
            Use CUDA if available - caution, this will cause out of memory errors with very large models, depending on GPU memory, and may actually be a performance bottleneck
    
        online_learning: bool = True
            Operate in online learning mode. Default True. In online learning mode the model will be retained between training sessions, 
            if set to False for offline training mode all training data must be present in the `fit` call and the model will be reset between trainings.
            This is primarily for parity with the scikit-learn expectations of estimators.
    
    
        debugging: bool = False
            Enable additional data checking, enabling may result in very poor execution time performance
    
        auto_scale: bool = False
            Automatically scale data in fit and predict to [0.0,1.0] interval, auto encode string labels to numerical. 
            This is primarily to support the scikit learn test suite.
            Warning: this does modify the input data
    
    
        auto_complement_encode: bool = False
            Automatically complement encode the input data and label. 
            This is primarily to support the scikit learn test suite.
            Warning: this does modify the input data.
        """
        self.baseline_vigilance = baseline_vigilance
        self.learning_rate = learning_rate
        self.committed_node_learning_rate = committed_node_learning_rate
        self.map_field_vigilance = map_field_vigilance
        self.map_field_learning_rate = map_field_learning_rate
        self.max_nodes = max_nodes
        self.number_of_category_nodes = number_of_category_nodes
        self.choice_parameter = choice_parameter
        self.vigilance_refinement_step = vigilance_refinement_step
        self.node_increase_step = node_increase_step
        self.use_cuda_if_available = use_cuda_if_available
        self.online_learning = online_learning
        self.debugging = debugging
        self.auto_scale = auto_scale
        self.auto_complement_encode = auto_complement_encode

    def _set_defaults(self) -> None:
        """Internal method that sets up the device if CUDA is available, and sets up the F2 and Map fields"""
        if self.use_cuda_if_available and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
            if self.use_cuda_if_available:
                logger.warning("CUDA requested but not available, using CPU.")
        
        self._committed_nodes = set()
        self._updated_nodes = set()
        
        self._number_of_increases = 0
        self._weight_a = torch.ones((self.number_of_category_nodes, self._input_vector_size), device=self._device, dtype=self._dtype)
        self._input_vector_sum = self._input_vector_size / 2
        self._weight_ab = torch.ones((self.number_of_category_nodes, self._number_of_labels), device=self._device, dtype=self._dtype)
        self._A_and_w = torch.empty(self._weight_a.shape, device=self._device, dtype=self._dtype)
        self._validated = False

        logger.debug(f"f1_size: {self._input_vector_size}, f2_size:{self.number_of_category_nodes}, committed beta = {self.committed_node_learning_rate}")

    def _range_validation(self) -> None:
        """Internal method for ensuring certain parameter values are on the interval [0.0, 1.0]"""
        for param_name in self._range_validation_params:
            value = getattr(self, param_name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{param_name} must be between 0.0 and 1.0, received {value}")

    @staticmethod
    def _vector_validation(vector: torch.Tensor, vector_name: str) -> None:
        """Internal method to validate the vector is:
        - more than one dimension
        - all values are in the [0.0, 1.0] interval
        - all are not NaN
        - all are floating point
        
        Parameters
        ----------
        vector: torch.Tensor
            The vector to validate
        
        vector_name: str
            The name of the vector, to make the error messages better

        Raises
        -------
        AssertionException
            If any conditions are violated
        """
        assert len(vector) == 1 or vector.shape[0] == 1, f"{vector_name} is not a 1d vector, {vector.shape}"

        any_value_over_one = (vector > 1.0).any()
        assert not any_value_over_one.item(), f"{vector_name} vector contains one or more values greater than 1.0"
        
        any_value_below_zero = (vector < 0.0).any()
        assert not any_value_below_zero.item(), f"{vector_name} vector contains one or more values less than 0.0"

        has_nan = torch.any(torch.isnan(vector))
        assert not has_nan, f"{vector_name} contains one or more NaN values"

        has_floating_point_values = torch.is_floating_point(vector)
        assert has_floating_point_values, f"{vector_name} contains one or more non-floating point values"

    def set_params(self, **parameters) -> Self:
        """For scikit-learn, to set paramters"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool=True) -> dict[str, Any]:
        """For scikit-learn, to get paramters"""
        return {
            "baseline_vigilance": self.baseline_vigilance,
            "learning_rate": self.learning_rate,
            "committed_node_learning_rate": self.committed_node_learning_rate,
            "map_field_vigilance": self.map_field_vigilance,
            "map_field_learning_rate": self.map_field_learning_rate,
            "max_nodes": self.max_nodes,
            "number_of_category_nodes": self.number_of_category_nodes,
            "choice_parameter": self.choice_parameter,
            "vigilance_refinement_step": self.vigilance_refinement_step,
            "node_increase_step": self.node_increase_step,
            "use_cuda_if_available": self.use_cuda_if_available,
            "online_learning" : self.online_learning,
            "debugging": self.debugging,
            "auto_scale": self.auto_scale,
            "auto_complement_encode": self.auto_complement_encode,
        }

    def _resonance_search_vector(self, input_vector: torch.Tensor, already_reset_nodes: set[int], rho_a: float) -> tuple[int, float]:
        """Core resonance search function over F1-F2 fields

        Parameters
        ----------
        input_vector: torch.Tensor
            The current input to match against 
        
        already_reset_nodes: list[int]
            The indexes of of nodes that have already been selected and not matched
        
        rho_a: float
            The current vigilance paramter
        
        Returns
        -------
        tuple[int, float]
            The category selected, and the degree of fuzzy set membership
        
        Notes
        -----
        This is a vectorized implementation with most of the heavy lifting in `_calculate_category_choice`
        """
        if self.debugging:
            FuzzyArtMap._vector_validation(input_vector, "Input")

        resonant_a = False
        number_of_f2_nodes, match_function, category_choice_function = self._calculate_category_choice(input_vector)
        indices = torch.argsort(category_choice_function, stable=True, descending=True)      
        
        category_choice_function[list(already_reset_nodes)] = torch.zeros((len(already_reset_nodes), ), dtype=self._dtype, device=self._device)
        has_evaluated_all_nodes = False
        while not resonant_a:
            for category_index in indices:
                if category_index.item() in already_reset_nodes:
                    continue

                # check for resonance (the full expression of eq 7)
                if match_function[category_index].item() >= rho_a or math.isclose(match_function[category_index].item(), rho_a):
                    resonant_a = True
                    break
                else:
                    resonant_a = False
                    already_reset_nodes.add(category_index.item())
                    category_choice_function[indices[category_index].item()] = 0

            # Creating new nodes if we've reset all of them
            # or go into fixed node mode
            if len(already_reset_nodes) >= number_of_f2_nodes:
                if has_evaluated_all_nodes:
                    raise RuntimeError(f"Resonance A search failed twice, ensure values are in range [0.0, 1.0]")

                if self.max_nodes is None or self.max_nodes > (number_of_f2_nodes + self.node_increase_step):
                    self._weight_a = torch.vstack((self._weight_a, torch.ones((self.node_increase_step, self._weight_a.shape[1]), device=self._device, dtype=self._dtype)))
                    self._weight_ab = torch.vstack((self._weight_ab, torch.ones((self.node_increase_step, self._weight_ab.shape[1]), device=self._device, dtype=self._dtype)))
                    self._A_and_w = torch.vstack((self._A_and_w, torch.empty((self.node_increase_step, self._weight_a.shape[1]), device=self._device, dtype=self._dtype)))
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
                indices = torch.argsort(category_choice_function, stable=True, descending=True)
                category_choice_function[list(already_reset_nodes)] = torch.zeros((len(already_reset_nodes), ), dtype=self._dtype, device=self._device)
                
        # return the selected category index j and the degree of membership
        return category_index.item(), match_function[category_index].item()

    def _calculate_category_choice(self, input_vector: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Internal method that does the primary calculations for resonance search across F1-F2
        
        Parameters
        ----------
        input_vector: torch.Tensor
            The current input to match against
        
        Returns
        -------
        tuple[int, torch.Tensor, torch.Tensor]
            - The current number of Category nodes (F2)
            - The result of the match function
            - The result of the category choice function

        Notes
        -----
        This method calculates the category choice function for all Tj, which is
        The L1-norm of I fuzzy min wj divided by alpha plus the L1-norm of wj ->
        |I ^ wj| / α + |wj|
        See Carpenter et al. (1992) p.700 eq 2-4
        """

        number_of_f2_nodes = self._weight_a.shape[0]  # Count how many F2a nodes we have

        # This is the top/first term of the category choice function - I (the input vector) min weight j
        # except here it's done for all categories (j's)
        torch.minimum(input_vector.repeat(number_of_f2_nodes,1), self._weight_a, out=self._A_and_w) # Fuzzy AND = min
        
        # Calculate the L1 norm (eq 4), save this term since it's used for next to calculate T (the category choice value)
        # And for resonance/fuzzy membership
        category_choice_numerator = torch.sum(self._A_and_w, 1) # Row vector of signals to F2 nodes
        
        # This is the left-hand term of the resonance check equation (eq 7) for all categories
        match_function = category_choice_numerator / self._input_vector_sum
        
        category_choice_function = category_choice_numerator / (self.choice_parameter + torch.sum(self._weight_a, 1)) # Choice function vector for F2
        return number_of_f2_nodes, match_function, category_choice_function

    def _train(self, input_vector: torch.Tensor, class_vector: torch.Tensor) -> None:
        """
        Internal training method that updates the weights between F1 & F2 and F2 and the Map Field
        
        Parameters
        ----------
        input_vector: torch.Tensor
            The current input to train on
        
        class_vector: torch.Tensor
            The ground truth label for the current input
        
        Notes
        -----
        This implements the option for fast-commit slow recode learning, see Carpenter et al. (1992). This method is also where the F2-Map Field
        is implemented.
        """
        vigilance = self.baseline_vigilance # We start off with ARTa vigilance at baseline
        resonant_ab = False # Not resonating in the Fab match layer
        already_reset_nodes = set() # We haven't rest any ARTa nodes for this input pattern yet, maintain list between resonance searches of Fa
        
        class_vector = class_vector.to(self._device)
        input_vector = input_vector.to(self._device)
        
        if self.debugging:
            assert class_vector.shape[1] >= 2, "Class vectors must be compliment encoded - requires at least 2 columns"
            FuzzyArtMap._vector_validation(class_vector, "Class")

        class_vector_sum = torch.sum(class_vector, 1)
        while not resonant_ab:            
            selected_category, match_function = self._resonance_search_vector(input_vector, already_reset_nodes, vigilance)
            
            map_field_activation = torch.minimum(class_vector, self._weight_ab[selected_category, None])
            
            category_match_function = torch.sum(map_field_activation, 1)/class_vector_sum

            if category_match_function > self.map_field_vigilance or math.isclose(category_match_function, self.map_field_vigilance):
                resonant_ab = True
            else: 
                already_reset_nodes.add(selected_category)
                vigilance = match_function + self.vigilance_refinement_step
                # This is ultimately a safety against floating point issues pushing vigilance over 1.0
                if vigilance > 1.0:
                    vigilance = 1.0 - self.vigilance_refinement_step

        self._updated_nodes.add(str(selected_category))
        if selected_category in self._committed_nodes:
            learning_rate = self.committed_node_learning_rate
        else:
            learning_rate = self.learning_rate

        self._weight_a[selected_category, None] = (learning_rate * torch.minimum(input_vector, self._weight_a[selected_category, None])) + ((1-learning_rate) * self._weight_a[selected_category, None])
        self._weight_ab[selected_category, None] = (self.map_field_learning_rate * map_field_activation) + ((1-self.map_field_learning_rate) * self._weight_ab[selected_category, None])
        self._committed_nodes.add(selected_category)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """
        Trains the model with the examples in `X` to the labels in `y`

        Parameters
        ----------
        X: npt.ArrayLike
            If `auto_complement_encode` is not set, this should be a `torch.Tensor` - the values to train on

        y: npt.ArrayLike
            If `auto_complement_encode` is not set, this should be a `torch.Tensor` - the labels to associate with the training values
            If `auto_scale` is set, categorical/string labels will be auto converted to numeric values
        
        Returns
        -------
        An instance of `FuzzyArtMap` to continue in the scikit-learn pipeline
        
        Notes
        -----
        Having `auto_complement_encode` or `auto_scale` will modify `X` and `y`.
        Setting `online_learning` to `False` will cause the model to be reset with every call to `fit`.
        """
        if self.auto_complement_encode:
            X, y= self._validate_data(X, y, accept_sparse=False, accept_large_sparse=False)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Length of samples tensor (X)-{X.shape[0]} does not equal length of labels tensor (y)-{y.shape[0]}")

        if self.auto_scale:
            X = self._scale_samples(X)
            y = self._scale_labels(y)

        if self.auto_complement_encode:
            if not isinstance(X, torch.Tensor):
                X = torch.from_numpy(np.array(X))
            X = FuzzyArtMap.complement_encode(X)
            if not isinstance(y, torch.Tensor):
                y = torch.from_numpy(np.array(y))
            y = FuzzyArtMap.complement_encode(y)

        if not self._validated:
            self._input_vector_size = X.shape[1]
            self._number_of_labels = y.shape[1]
            
            self._set_defaults()
            self._range_validation()

            # If in online learning mode, set validation, if not set, it enables
            # a reset of the F1, F2, & map fields in `_set_defaults`
            if self.online_learning:
                self._validated = True

        for vector_index, input_vector in enumerate(X):
            self._train(input_vector.unsqueeze_(0), y[vector_index].unsqueeze_(0))
        
        logger.debug(f"training updated: {','.join(self._updated_nodes)}")
        self._updated_nodes.clear()
        return self

    def _scale_samples(self, samples: npt.ArrayLike) -> npt.ArrayLike:
        """Internal function that scales samples (X values) to [0.0, 1.0]"""
        scaler = MinMaxScaler()
        samples = scaler.fit_transform(samples)
        samples = np.clip(samples, 0.0, 1.0)
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(np.array(samples))
        return samples
    
    def _scale_labels(self, labels: npt.ArrayLike) -> npt.ArrayLike:
        """Internal function that converts the shape of the labels, changes them from string to numeric if necessary, 
        and scales them to [0.0, 1.0]"""
        if len(labels.shape) == 1:
            labels = labels.reshape(-1,1)
        if np.any(np.vectorize(lambda x: isinstance(x, str))(labels)):
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels)
        scaler = MinMaxScaler()
        labels = scaler.fit_transform(labels)
        labels = np.clip(labels, 0.0, 1.0)
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(np.array(labels))
        return labels

    @staticmethod
    def complement_encode(original_vector: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Complement encode a vector

        Parameters
        ----------
        original_vector: torch.Tensor
            The value to complement encode
        
        debug: bool = False
            Validate the `original_vector` to make sure it meets Fuzzy ARTMAP requirements
        
        Returns
        -------
        torch.Tensor
            The complement encoded tensor
        
        Notes
        -----
        This returns (1-`original_vector`) horizontally stacked with the `original_vector`.
        """
        if debug:
            FuzzyArtMap._vector_validation(original_vector, "Original")
        complement = 1-original_vector
        complement_encoded_value = torch.hstack((original_vector, complement))
        return complement_encoded_value

    def predict(self, X: npt.ArrayLike) -> npt.NDArray:
        """
        Predict the class associated with `X`
        
        Parameters
        ----------
        X: npt.ArrayLike
            If `auto_scale` is set, values in `X` will automatically be scaled to the [0.0, 1.0] interval.
            
            If `auto_complement_encode` is not set, this should be a `torch.Tensor`, if it is set, `npt.ArrayLike` is 
            acceptable, and will be automatically run through `complement_encode`
        
        Returns
        -------
        npt.NDArray
            The complement encoded classes for each of the `X`s presented
        """
        rho_a = 0 # set ARTa vigilance to first match
        if self.auto_complement_encode:
            X = self._validate_data(X, accept_sparse=False, accept_large_sparse=False)
        
        if self.auto_scale:
            X = self._scale_samples(X)
        if self.auto_complement_encode:
            X = FuzzyArtMap.complement_encode(X)
        results = []
        for input_vector in X:
            J, _ = self._resonance_search_vector(input_vector.unsqueeze_(0), [], rho_a)
            # (Called x_ab in Fuzzy ARTMAP paper)
            results.append(torch.asarray(self._weight_ab[J, None])) # Fab activation vector
        return np.array(results)

    def predict_with_membership(self, input_vector: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Predict the class associated with `input_vector`
        
        Parameters
        ----------
        input_vector: torch.Tensor
            The input vector to get a category prediction for
        
        Returns
        -------
        tuple[torch.Tensor, float]
            The complement encoded category values
            The degree to which the `input_vector` matches the selected category
        
        Notes
        -----
        Membership is the Fuzzy Set Membership 
        """
        rho_a = 0 # set ARTa vigilance to first match
        J, membership_degree = self._resonance_search_vector(input_vector, [], rho_a)
        
        # (Called x_ab in Fuzzy ARTMAP paper)
        return self._weight_ab[J, None], membership_degree # Fab activation vector & fuzzy membership value

    def save_model(self, descriptor: str = None, path_prefix: str = "models") -> str:
        """
        Saves the current model and parameters to disk
        
        Parameters
        ----------
        descriptor: str = None
            Optional descriptor to include in the filename
        
        path_prefix: str = "models"
            Optional prefix of where to save the model

        Returns
        -------
        str
            The path of the file that was saved, in the form of `path_prefix`/fam{timestamp}{descriptor}.pt
        """
        model_timestamp = datetime.now().isoformat().replace("-", "_").replace(":", "_").replace(".", "_")
        cleaned_descriptor = ""
        if descriptor:
            cleaned_descriptor = "_" + descriptor.replace("-", "_").replace(":", "_").replace(".", "_")
        model_path = f"fam_{model_timestamp}{cleaned_descriptor}.pt"
        if path_prefix:
             model_path = f"{path_prefix}/{model_path}"
        torch.save((self._weight_a, self._weight_ab, self._committed_nodes, self.get_params()), model_path)
        return model_path
    
    def load_model(self, model_path: str) -> Self:
        """
        Loads the model and parameters from the specified `model_path` into the current instance.
        
        Paramters
        ---------
        model_path: str
            The path on disk to the model file
        
        Returns
        -------
        The current FuzzyArtMap instance, with the updated values from the loaded model
        """
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"`{model_path}` was not found or is a directory")
        logger.info(f"Loading model from {model_path}")
        weight_a, weight_ab, committed_nodes, parameters = torch.load(model_path)

        loaded_f1_size = weight_a.shape[1]
        loaded_f2_size = weight_a.shape[0]
        logger.debug(f"Parameter f2: {parameters['number_of_category_nodes']} - Actual f1: {loaded_f1_size}, f2: {loaded_f2_size}")
        self._input_vector_size = weight_a.shape[1]
        parameters["number_of_category_nodes"] = weight_a.shape[0]
        self.set_params(**parameters)
        
        self._weight_a = weight_a
        self._weight_ab = weight_ab
        self._committed_nodes = committed_nodes
        
        logger.info("Model loaded")
        return self
    
    def get_number_of_nodes(self) -> int:
        """
        Helper method to get the number of nodes in the model.
        
        Note
        ----
        This is works off the map field weights to get the number of categories (F2 nodes) from the model
        """
        return self._weight_ab.shape[0]

    def get_number_of_increases(self) -> int:
        """Helper method to get how often the number of categories needed to be increased"""
        return self._number_of_increases

    def get_increase_size(self) -> int:
        """Helper method to get the number of nodes added at each increase"""
        return self.node_increase_step
    
    def get_committed_nodes(self) -> str:
        """
        Helper method to get all the nodes that have been used during model training
        
        Returns
        -------
        str
            A comma delimited string of the the commited nodes by index (zero-based)
        
        Notes
        -----
        This reflects the nodes (in F2) that have been assigned to a value in the map field. 
        These nodes should have weight values associated with them in `_weight_ab`.
        """
        return ",".join([str(n) for n in self._committed_nodes])
    
    def get_weight_a(self) -> torch.tensor:
        """Helper method to get the weight tensor between F1 and F2"""
        return self._weight_a
    
    def get_weight_ab(self) -> torch.tensor:
        """Helper method to get the weight tensor between F2 and the map field"""
        return self._weight_ab
