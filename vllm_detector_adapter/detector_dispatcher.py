# Standard
import functools

# global list to store all the registered functions with
# their types and qualified name
global_fn_list = dict()


def detector_dispatcher(types=None):
    """Decorator to dispatch to processing function based on type of the detector.

    This decorator allows us to reuse same function name for different types of detectors.
    For example, the same function name can be used for text chat and context analysis
    detectors. These decorated functions for these detectors will have different arguments
    and implementation but they share the same function name.

    NOTE: At the time of invoking these decorated function, the user needs to specify the type
    of the detector using fn_type argument.

    CAUTION: Since this decorator allow re-use of the name, one must take care of using different types
    for testing different functions.

    Args:
        types (list): Type of the detector this function applies to.
        args: Positional arguments passed to the processing function.
        kwargs: Keyword arguments passed to the processing function.

    Examples
    --------

    @detector_dispatcher(types=["foo"])
    def f(x):
        pass

    # Decorator can take multiple types as well
    @detector_dispatcher(types=["bar", "baz"])
    def f(x):
        pass

    When calling these functions, one can specify the type of the detector as follows:
    f(x, fn_type="foo")
    f(x, fn_type="bar")
    """
    global global_fn_list

    if not types:
        raise ValueError("Must specify types.")

    def decorator(func):
        fn_name = func.__qualname__

        if fn_name not in global_fn_list:
            # Associate each function with its type to create a dictionary of form:
            # {"fn_name": {type1: function, type2: function}
            # NOTE: "function" here are really function pointers
            global_fn_list[fn_name] = {t: func for t in types}
        elif fn_name in global_fn_list and (types & global_fn_list[fn_name].keys()):
            # Error out if the types function with same type declaration exist in the global
            # list already
            raise ValueError("Function already registered with the same types.")
        else:
            # Add the function to the global list with corresponding type
            global_fn_list[fn_name] |= {t: func for t in types}

        @functools.wraps(func)
        def wrapper(*args, fn_type=None, **kwargs):
            fn_name = func.__qualname__

            if not fn_type:
                raise ValueError("Must specify fn_type.")

            if fn_type not in global_fn_list[fn_name].keys():
                raise ValueError("Invalid fn_type.")

            # Grab the function using its fully qualified name and the specified type
            # and then call it
            return global_fn_list[fn_name][fn_type](*args, **kwargs)

        return wrapper

    return decorator
