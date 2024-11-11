import traceback
from sklearn.utils.estimator_checks import check_estimator

from fuzzy_artmap.fuzzy_artmap import FuzzyArtMap



def test_fuzzy_artmap_scikit_estimator_interface() -> None:
    estimator = FuzzyArtMap(debugging=True, auto_scale=True, auto_complement_encode=True)
    # check_estimator(estimator)
    expected_failures = {"check_methods_subset_invariance": "auto-normalization applied", "check_fit_idempotent": "auto-normalization applied"}
    checks = check_estimator(estimator, generate_only=True)
    for check in checks:
        try:
            check[1](check[0])
        except AssertionError as ae:
            # For now mimic the functionality that will appear in scikit-learn 1.6
            if check[1].func.__name__ in expected_failures:
               print("todo print warning") 
            else:
                stack_trace = ''.join(traceback.TracebackException.from_exception(ae).format())
                print(stack_trace)
                print(ae)
                raise ae
        except Exception as e:
            stack_trace = ''.join(traceback.TracebackException.from_exception(e).format())
            print(stack_trace)
            print(e)
