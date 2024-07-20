import torch
from torch import tensor
from fuzzy_artmap.fuzzy_artmap import FuzzyArtMap

in_class = tensor([[1,0]])
training_in_value = tensor([[0.1,0.1,0.9,0.9]])
test_in_value = tensor([[0.11,0.11,0.99,0.99]])

out_class = tensor([[0,1]])
training_out_value = tensor([[0.9,0.9,0.1,0.1]])
test_out_value = tensor([[0.99,0.99,0.11,0.11]])


def test_fuzzy_artmap_selects_first_uncommited_node_with_wide_variance_input() -> None:

    fuzzy_artmap = FuzzyArtMap(4, initial_number_of_category_nodes=2, baseline_vigilance=0.95)

    starting_a_first_weight = fuzzy_artmap.weight_a[0, None].clone()
    starting_ab_first_weight = fuzzy_artmap.weight_ab[0, None].clone()
    starting_a_second_weight = fuzzy_artmap.weight_a[1, None].clone()
    starting_ab_second_weight = fuzzy_artmap.weight_ab[1, None].clone()
    weight_a_shape = fuzzy_artmap.weight_a.shape
    weight_ab_shape = fuzzy_artmap.weight_ab.shape

    fuzzy_artmap.fit([training_in_value], [in_class])

    learned_a_weight = fuzzy_artmap.weight_a[0, None].clone()
    learned_ab_weight = fuzzy_artmap.weight_ab[0, None].clone()
    a_second_weight = fuzzy_artmap.weight_a[1, None]
    ab_second_weight = fuzzy_artmap.weight_ab[1, None]

    assert not torch.equal(starting_a_first_weight, learned_a_weight)
    assert not torch.equal(starting_ab_first_weight, learned_ab_weight)
    assert torch.equal(starting_a_second_weight, a_second_weight)
    assert torch.equal(starting_ab_second_weight, ab_second_weight)
    assert weight_a_shape == fuzzy_artmap.weight_a.shape
    assert weight_ab_shape == fuzzy_artmap.weight_ab.shape

    fuzzy_artmap.fit([training_out_value], [out_class])
    first_category_a_weight = fuzzy_artmap.weight_a[0, None]
    first_category_ab_weight = fuzzy_artmap.weight_ab[0, None]
    
    assert not torch.equal(starting_a_second_weight, a_second_weight)
    assert not torch.equal(starting_ab_second_weight, ab_second_weight)
    assert torch.equal(learned_a_weight, first_category_a_weight)
    assert torch.equal(learned_ab_weight, first_category_ab_weight)
    assert weight_a_shape == fuzzy_artmap.weight_a.shape
    assert weight_ab_shape == fuzzy_artmap.weight_ab.shape


def test_fuzzy_artmap_predicts_training_value() -> None:
    fuzzy_artmap = FuzzyArtMap(4, initial_number_of_category_nodes=2, baseline_vigilance=0.95)
    
    fuzzy_artmap.fit([training_in_value, training_out_value], [in_class, out_class])
    
    predicted_class = fuzzy_artmap.predict(training_in_value)
    assert torch.equal(in_class, predicted_class)

    predicted_class = fuzzy_artmap.predict(training_out_value)
    assert torch.equal(out_class, predicted_class)


def test_fuzzy_artmap_predicts_training_value_with_membership() -> None:    
    fuzzy_artmap = FuzzyArtMap(4, initial_number_of_category_nodes=2, baseline_vigilance=0.95)
    
    fuzzy_artmap.fit([training_in_value, training_out_value], [in_class, out_class])

    predicted_class, membership_degree = fuzzy_artmap.predict_with_membership(training_in_value)
    assert torch.equal(in_class, predicted_class)
    assert membership_degree == 1.0

    predicted_class, membership_degree = fuzzy_artmap.predict_with_membership(training_out_value)
    assert torch.equal(out_class, predicted_class)
    assert membership_degree == 1.0

    predicted_class, membership_degree = fuzzy_artmap.predict_with_membership(test_in_value)
    assert torch.equal(in_class, predicted_class)
    assert membership_degree == 1.0

    predicted_class, membership_degree = fuzzy_artmap.predict_with_membership(test_out_value)
    assert torch.equal(out_class, predicted_class)
    assert membership_degree == 1.0


def test_fuzzy_artmap_grows_nodes() -> None:
    
    fuzzy_artmap = FuzzyArtMap(4, initial_number_of_category_nodes=1, baseline_vigilance=0.95)

    starting_weight_a_shape = fuzzy_artmap.weight_a.shape
    starting_weight_ab_shape = fuzzy_artmap.weight_ab.shape
    starting_A_and_w_shape = fuzzy_artmap.A_and_w.shape
    
    fuzzy_artmap.fit([training_in_value, training_out_value, test_in_value, test_out_value], [in_class, out_class, in_class, out_class])

    assert fuzzy_artmap.number_of_increases == 1

    assert starting_weight_a_shape[0] + fuzzy_artmap.node_increase_step == fuzzy_artmap.weight_a.shape[0]
    assert starting_weight_a_shape[1] == fuzzy_artmap.weight_a.shape[1]

    assert starting_weight_ab_shape[0] + fuzzy_artmap.node_increase_step == fuzzy_artmap.weight_ab.shape[0]
    assert starting_weight_ab_shape[1] == fuzzy_artmap.weight_ab.shape[1]

    assert starting_A_and_w_shape[0] + fuzzy_artmap.node_increase_step == fuzzy_artmap.A_and_w.shape[0]
    assert starting_A_and_w_shape[1] == fuzzy_artmap.A_and_w.shape[1]


def test_fuzzy_artmap_supports_max_nodes_mode() -> None:
    
    fuzzy_artmap = FuzzyArtMap(4, initial_number_of_category_nodes=1, baseline_vigilance=0.95, max_nodes=1)

    starting_weight_a_shape = fuzzy_artmap.weight_a.shape
    starting_weight_ab_shape = fuzzy_artmap.weight_ab.shape
    starting_A_and_w_shape = fuzzy_artmap.A_and_w.shape
    starting_rho_ab = fuzzy_artmap.rho_ab
    starting_beta_ab = fuzzy_artmap.beta_ab
    starting_rho_a_bar = fuzzy_artmap.rho_a_bar
    
    fuzzy_artmap.fit([training_in_value, training_out_value, test_in_value, test_out_value], [in_class, out_class, in_class, out_class])

    assert fuzzy_artmap.number_of_increases == 0

    assert starting_weight_a_shape[0] == fuzzy_artmap.weight_a.shape[0]
    assert starting_weight_a_shape[1] == fuzzy_artmap.weight_a.shape[1]

    assert starting_weight_ab_shape[0] == fuzzy_artmap.weight_ab.shape[0]
    assert starting_weight_ab_shape[1] == fuzzy_artmap.weight_ab.shape[1]

    assert starting_A_and_w_shape[0] == fuzzy_artmap.A_and_w.shape[0]
    assert starting_A_and_w_shape[1] == fuzzy_artmap.A_and_w.shape[1]

    # These values are specified in _resonance_search_vector
    # It is only important that they are less than the starting values.
    assert fuzzy_artmap.rho_ab < starting_rho_ab
    assert fuzzy_artmap.beta_ab < starting_beta_ab
    assert fuzzy_artmap.rho_a_bar < starting_rho_a_bar


def test_calculate_first_category_choice() -> None:
    fuzzy_artmap = FuzzyArtMap(4, initial_number_of_category_nodes=1)

    # Number of F2 nodes
    expected_N = 1

    # |I ^ wj| - The L1 norm of the input min the weight, initial weights are initialized to the 1s vector
    # so I ^ wj -> I, therefore expected S is the sum of the values in the vector I
    expected_S = torch.sum(training_in_value, 1)

    # The category choice function is the S term divided by alpha + the l1 norm of the weight vector for
    # the category (wj) - since no learning has taken place yet, this is a ones vector
    expected_T = expected_S / (fuzzy_artmap.alpha + torch.sum(torch.ones((1, 4)), 1))
    
    N, S, T = fuzzy_artmap._calculate_category_choice(training_in_value)

    assert N == expected_N
    assert torch.equal(S, expected_S)
    assert torch.equal(T, expected_T)


def test_fuzzy_artmap_matches_expected_learning() -> None:
    fuzzy_artmap = FuzzyArtMap(4, baseline_vigilance=0.9, initial_number_of_category_nodes=1)
    first_point = tensor([[0.1, 0.1]])
    complement_encoded_first_point = FuzzyArtMap.complement_encode(first_point)
    fuzzy_artmap.fit([complement_encoded_first_point], [in_class])
    assert torch.equal(fuzzy_artmap.weight_a, complement_encoded_first_point)
    assert torch.equal(fuzzy_artmap.weight_ab, in_class)
    
    fifth_point = tensor([[0.11, 0.1]])
    complement_encoded_fifth_point = FuzzyArtMap.complement_encode(fifth_point)
    fuzzy_artmap.fit([complement_encoded_fifth_point], [in_class])
    
    assert torch.allclose(fuzzy_artmap.weight_a, tensor([[0.1000, 0.1000, 0.8925, 0.9000]]))
    assert torch.equal(fuzzy_artmap.weight_ab, in_class)

    sixth_point = tensor([[0.13, 0.1]])
    complement_encoded_sixth_point = FuzzyArtMap.complement_encode(sixth_point)
    fuzzy_artmap.fit([complement_encoded_sixth_point], [in_class])

    assert torch.allclose(fuzzy_artmap.weight_a, tensor([[0.1000, 0.1000, 0.875625, 0.9000]]))
    assert torch.equal(fuzzy_artmap.weight_ab, in_class)

    seventh_point = tensor([[0.14, 0.1]])
    complement_encoded_seventh_point = FuzzyArtMap.complement_encode(seventh_point)
    fuzzy_artmap.fit([complement_encoded_seventh_point], [out_class])
    assert torch.allclose(fuzzy_artmap.weight_a[0, None], tensor([[0.1000, 0.1000, 0.875625, 0.9000]]))
    assert torch.equal(fuzzy_artmap.weight_ab[0, None], in_class)

    assert torch.allclose(fuzzy_artmap.weight_a[1, None], tensor([[0.14, 0.1 , 0.86, 0.9 ]]))
    assert torch.equal(fuzzy_artmap.weight_ab[1, None], out_class)
    