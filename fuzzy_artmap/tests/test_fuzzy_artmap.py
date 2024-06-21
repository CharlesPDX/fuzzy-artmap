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

    fuzzy_artmap = FuzzyArtMap(4, f2_size=2, rho_a_bar=0.95)

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
    fuzzy_artmap = FuzzyArtMap(4, f2_size=2, rho_a_bar=0.95)
    
    fuzzy_artmap.fit([training_in_value, training_out_value], [in_class, out_class])
    
    predicted_class = fuzzy_artmap.predict(training_in_value)
    assert torch.equal(in_class, predicted_class)

    predicted_class = fuzzy_artmap.predict(training_out_value)
    assert torch.equal(out_class, predicted_class)


def test_fuzzy_artmap_predicts_training_value_with_membership() -> None:    
    fuzzy_artmap = FuzzyArtMap(4, f2_size=2, rho_a_bar=0.95)
    
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
    
    fuzzy_artmap = FuzzyArtMap(4, f2_size=1, rho_a_bar=0.95)

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
    
    fuzzy_artmap = FuzzyArtMap(4, f2_size=1, rho_a_bar=0.95, max_nodes=1)

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

    assert fuzzy_artmap.rho_ab < starting_rho_ab
    assert fuzzy_artmap.beta_ab < starting_beta_ab
    assert fuzzy_artmap.rho_a_bar < starting_rho_a_bar


def test_calculate_activation() -> None:
    # Number of F2 nodes
    expected_N = 1


    expected_S = 0

    expected_T = 0

    fuzzy_artmap = FuzzyArtMap(4, f2_size=1, rho_a_bar=0.95)
    N, S, T = fuzzy_artmap._calculate_activation(training_in_value)

    # assert N == expected_N
    # assert S == expected_S
    # assert T == expected_T