import functools
from typing import TypeVar, Any

T = TypeVar("T")


def partial_class(cls: type[T], *args: Any, **kwargs: Any) -> type[T]:
    class NewClass(cls):
        def __init__(self, *inner_args: Any, **inner_kwargs: Any) -> None:
            functools.partial(cls.__init__, *args, **kwargs)(
                self, *inner_args, **inner_kwargs
            )

    return NewClass
