from datetime import datetime
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, TypeVar, Tuple

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch

TFuzzyArtMap = TypeVar("TFuzzyArtMap", bound="FuzzyArtMap")

class FuzzyArtMap:
    def __init__(self,
                 f1_size: int,
                 f2_size: int = 10,
                 number_of_labels: int = 2,
                 rho_a_bar: float = 0.0,
                 beta: float = 1.0,
                 beta_ab: float = 1.0,
                 rho_ab: float = 0.95,
                 max_nodes: int = None,
                 committed_beta: float = 0.75,
                 use_cuda_if_available: bool = False,
                 debugging: bool = False,
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
        self.f1_size = f1_size
        self.f2_size = f2_size
        self.number_of_labels = number_of_labels
        self.beta = beta  
        self.beta_ab = beta_ab 
        self.rho_ab = rho_ab
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        self.committed_beta = committed_beta
        self.max_nodes = max_nodes
        self.use_cuda_if_available = use_cuda_if_available
        self.debugging = debugging

        self.parameters = {
            "f1_size": self.f1_size,
            "f2_size" : self.f2_size,
            "number_of_labels": self.number_of_labels,
            "beta": self.beta,
            "beta_ab": self.beta_ab,
            "rho_ab": self.rho_ab,
            "rho_a_bar": self.rho_a_bar,
            "committed_beta": self.committed_beta,
            "max_nodes": self.max_nodes,
            "use_cuda_if_available": self.use_cuda_if_available,
            "debugging": self.debugging
        }

        self._set_defaults()

    def _set_defaults(self) -> None:
        if self.use_cuda_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            if self.use_cuda_if_available:
                logger.warning("CUDA requested but not available, using CPU.")
        self.range_validation_params = ["rho_a_bar", "committed_beta", "rho_ab", "beta"]
        self.epsilon = 0.001        # Fab mismatch raises ARTa vigilance to this much above what is needed to reset ARTa
        self.alpha = 0.001  # "Choice" parameter > 0. Set small for the conservative limit (Fuzzy AM paper, Sect.3)
        self.dtype = torch.float
        self.committed_nodes = set()
        self.updated_nodes = set()
        self.node_increase_step = 50 # number of F2 nodes to add when required
        self.number_of_increases = 0
        self.weight_a = torch.ones((self.f2_size, self.f1_size), device=self.device, dtype=self.dtype)
        self.input_vector_sum = self.f1_size / 2
        self.weight_ab = torch.ones((self.f2_size, self.number_of_labels), device=self.device, dtype=self.dtype)
        self.A_and_w = torch.empty(self.weight_a.shape, device=self.device, dtype=self.dtype)
        self.validated = False

        logger.debug(f"f1_size: {self.f1_size}, f2_size:{self.f2_size}, committed beta = {self.committed_beta}")

    def _range_validation(self) -> None:
        for param_name in self.range_validation_params:
            value = getattr(self, param_name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{param_name} must be between 0.0 and 1.0, received {value}")

    @staticmethod
    def _vector_validation(vector: torch.tensor, vector_name: str) -> None:
        any_value_over_one = (vector > 1.0).any()
        assert not any_value_over_one.item(), f"{vector_name} vector contains one or more values greater than 1.0"
        
        any_value_below_zero = (vector < 0.0).any()
        assert not any_value_below_zero.item(), f"{vector_name} vector contains one or more values less than 0.0"

    def set_params(self, **parameters: Dict[str, Any]) -> TFuzzyArtMap:
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            self.parameters[parameter] = value
        self._set_defaults()
        return self

    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        return self.parameters

    def _resonance_search_vector(self, input_vector: torch.tensor, already_reset_nodes: List[int], rho_a: float) -> Tuple[int, float]:
        if self.debugging:
            FuzzyArtMap._vector_validation(input_vector, "Input")

        resonant_a = False
        N, S, T = self._calculate_activation(input_vector)
        _, indices = torch.sort(T, stable=True, descending=True)
        all_membership_degrees = S / self.input_vector_sum
        T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=self.dtype, device=self.device)
        has_evaluated_all_nodes = False
        while not resonant_a:
            for J in indices:
                if J.item() in already_reset_nodes:
                    continue

                if all_membership_degrees[J].item() >= rho_a or math.isclose(all_membership_degrees[J].item(), rho_a):
                    resonant_a = True
                    break
                else:
                    resonant_a = False
                    already_reset_nodes.append(indices[J].item())
                    T[indices[J].item()] = 0

            # Creating a new node if we've reset all of them
            if len(already_reset_nodes) >= N:
                if has_evaluated_all_nodes:
                    raise RuntimeError(f"Resonance A search failed twice, ensure values are in range [0.0, 1.0]")

                if self.max_nodes is None or self.max_nodes > (N + self.node_increase_step):
                    self.weight_a = torch.vstack((self.weight_a, torch.ones((self.node_increase_step, self.weight_a.shape[1]), device=self.device, dtype=self.dtype)))
                    self.weight_ab = torch.vstack((self.weight_ab, torch.ones((self.node_increase_step, self.weight_ab.shape[1]), device=self.device, dtype=self.dtype)))
                    self.A_and_w = torch.vstack((self.A_and_w, torch.empty((self.node_increase_step, self.weight_a.shape[1]), device=self.device, dtype=self.dtype)))
                    self.number_of_increases += 1
                else:
                    self.rho_ab = 0
                    self.beta_ab = 0.75
                    self.rho_a_bar = 0
                    rho_a = self.rho_a_bar
                    logger.warning(f"Maximum number of nodes reached, {len(already_reset_nodes)} - adjusting rho_ab to {self.rho_ab} and beta_ab to {self.beta_ab}")
                    already_reset_nodes.clear()
                has_evaluated_all_nodes = True
                N, S, T = self._calculate_activation(input_vector)
                _, indices = torch.sort(T, stable=True, descending=True)
                all_membership_degrees = S / self.input_vector_sum
                T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=self.dtype, device=self.device)
                
        return J.item(), all_membership_degrees[J].item()

    def _calculate_activation(self, input_vector: torch.tensor) -> Tuple[int, torch.tensor, torch.tensor]:
        N = self.weight_a.shape[0]  # Count how many F2a nodes we have

        torch.minimum(input_vector.repeat(N,1), self.weight_a, out=self.A_and_w) # Fuzzy AND = min
        S = torch.sum(self.A_and_w, 1) # Row vector of signals to F2 nodes
        T = S / (self.alpha + torch.sum(self.weight_a, 1)) # Choice function vector for F2
        return N,S,T

    def _train(self, input_vector: torch.tensor, class_vector: torch.tensor) -> None:
        rho_a = self.rho_a_bar # We start off with ARTa vigilance at baseline
        resonant_ab = False # Not resonating in the Fab match layer
        already_reset_nodes = [] # We haven't rest any ARTa nodes for this input pattern yet, maintain list between resonance searches of Fa
        
        class_vector = class_vector.to(self.device)
        input_vector = input_vector.to(self.device)
        
        if self.debugging:
            assert class_vector.shape[1] >= 2, "Class vectors must be compliment encoded - requires at least 2 columns"
            FuzzyArtMap._vector_validation(class_vector, "Class")

        class_vector_sum = torch.sum(class_vector, 1)
        while not resonant_ab:            
            J, x = self._resonance_search_vector(input_vector, already_reset_nodes, rho_a)
            
            z = torch.minimum(class_vector, self.weight_ab[J, None])
            
            resonance = torch.sum(z, 1)/class_vector_sum
            if resonance > self.rho_ab or math.isclose(resonance, self.rho_ab):
                resonant_ab = True
            else: 
                already_reset_nodes.append(J)
                rho_a = x + self.epsilon                
                if rho_a > 1.0:
                    rho_a = 1.0 - self.epsilon

        self.updated_nodes.add(J)
        if J in self.committed_nodes:
            beta = self.committed_beta
        else:
            beta = self.beta

        self.weight_a[J, None] = (beta * torch.minimum(input_vector, self.weight_a[J, None])) + ((1-beta) * self.weight_a[J, None])
        self.weight_ab[J, None] = (self.beta_ab * z) + ((1-self.beta_ab) * self.weight_ab[J, None])
        self.committed_nodes.add(J)

    def fit(self, input_vectors, class_vectors) -> None:
        if not self.validated:
            self._range_validation()
            self.validated = True

        for vector_index, input_vector in enumerate(input_vectors):
            self._train(input_vector, class_vectors[vector_index])
        self.updated_nodes.clear()


    @staticmethod
    def complement_encode(original_vector: torch.tensor) -> torch.tensor:
        complement = 1-original_vector
        complement_encoded_value = torch.hstack((original_vector,complement))
        FuzzyArtMap._vector_validation(complement_encoded_value, "Complement encoded")
        return complement_encoded_value

    def predict(self, input_vector: torch.tensor) -> torch.tensor:
        rho_a = 0 # set ARTa vigilance to first match
        J, _ = self._resonance_search_vector(input_vector, [], rho_a)
        
        # (Called x_ab in Fuzzy ARTMAP paper)
        return self.weight_ab[J, None] # Fab activation vector

    def predict_with_membership(self, input_vector: torch.tensor) -> Tuple[torch.tensor, float]:
        rho_a = 0 # set ARTa vigilance to first match
        J, membership_degree = self._resonance_search_vector(input_vector, [], rho_a)
        
        # (Called x_ab in Fuzzy ARTMAP paper)
        return self.weight_ab[J, None], membership_degree # Fab activation vector & fuzzy membership value

    def save_model(self, descriptor: str = None, path_prefix: str = "models") -> str:
        model_timestamp = datetime.now().isoformat().replace("-", "_").replace(":", "_").replace(".", "_")
        cleaned_descriptor = ""
        if descriptor:
            cleaned_descriptor = "_" + descriptor.replace("-", "_").replace(":", "_").replace(".", "_")
        model_path = f"fam_{model_timestamp}{cleaned_descriptor}.pt"
        if path_prefix:
             model_path = f"{path_prefix}/{model_path}"
        torch.save((self.weight_a, self.weight_ab, self.committed_nodes, self.parameters), model_path)
        return model_path
    
    def load_model(self, model_path: str) -> TFuzzyArtMap:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"`{model_path}` was not found or is a directory")
        logger.info(f"Loading model from {model_path}")
        weight_a, weight_ab, committed_nodes, parameters = torch.load(model_path)

        loaded_f1_size = weight_a.shape[1]
        loaded_f2_size = weight_a.shape[0]
        logger.debug(f"Parameter f1: {parameters['f1_size']}, f2: {parameters['f2_size']} - Actual f1: {loaded_f1_size}, f2: {loaded_f2_size}")
        parameters["f1_size"] = weight_a.shape[1]
        parameters["f2_size"] = weight_a.shape[0]
        self.set_params(parameters)
        
        self.weight_a = weight_a
        self.weight_ab = weight_ab
        self.committed_nodes = committed_nodes
        
        logger.info("Model loaded")
        return self
    
    def get_number_of_nodes(self) -> int:
        return self.weight_ab.shape[0]

    def get_number_of_increases(self) -> int:
        return self.number_of_increases

    def get_increase_size(self) -> int:
        return self.node_increase_step
    
    def get_committed_nodes(self) -> str:
        return ",".join([str(n) for n in self.committed_nodes])
    
    def get_weight_a(self) -> torch.tensor:
        return self.weight_a
    
    def get_weight_ab(self) -> torch.tensor:
        return self.weight_ab
