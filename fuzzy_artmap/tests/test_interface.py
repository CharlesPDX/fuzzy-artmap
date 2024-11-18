import traceback
from sklearn.utils.estimator_checks import check_estimator

from fuzzy_artmap.fuzzy_artmap import FuzzyArtMap



def test_fuzzy_artmap_scikit_estimator_interface() -> None:
    """Cycles through the scikit estimator checks, skipping 2 because of scaling and complement encoding"""
    estimator = FuzzyArtMap(debugging=True, auto_scale=True, auto_complement_encode=True)
    expected_failures = {"check_methods_subset_invariance": "auto-normalization applied", "check_fit_idempotent": "auto-normalization applied"}
    checks = check_estimator(estimator, generate_only=True)
    for check in checks:
        try:
            check[1](check[0])
        except AssertionError as ae:
            # For now mimic the functionality that will appear in scikit-learn 1.6
            if warning := expected_failures.get(check[1].func.__name__):
               print(warning)
            else:
                stack_trace = ''.join(traceback.TracebackException.from_exception(ae).format())
                print(stack_trace)
                print(ae)
                raise ae
        except Exception as e:
            stack_trace = ''.join(traceback.TracebackException.from_exception(e).format())
            print(stack_trace)
            print(e)
            raise e
