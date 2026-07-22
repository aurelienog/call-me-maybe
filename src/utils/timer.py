import time
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def timer(func: Callable[P, R]) -> Callable[P, R]:
    """
    Measure and report the execution time of a function.

    Wraps a callable, measures its execution time using a
    high-resolution timer, prints the elapsed time to the
    console, and returns the original result.

    Args:
        func: Function to wrap.

    Returns:
        A wrapped function that preserves the original
        signature and behavior while reporting its
        execution time.
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed_time = time.perf_counter() - start
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60

        print("\n" + "=" * 50)
        if minutes >= 1:
            print(f"[TIMING] Proceso completado en: {int(minutes)}m"
                  f"{seconds:.2f}s")
        else:
            print(f"[TIMING] Proceso completado en: {seconds:.2f}s")
        print("=" * 50 + "\n")

        return result

    return wrapper
