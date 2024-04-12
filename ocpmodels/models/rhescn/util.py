from typing import TYPE_CHECKING, Any, Literal

import jaxtyping
import torch
from einops import rearrange
from jaxtyping._storage import get_shape_memo, shape_str
from lovely_tensors import lovely
from typing_extensions import TypeVar


def rearrange_view(x: torch.Tensor, pattern: str, **axes_lengths: int):
    """
    https://github.com/arogozhnikov/einops/issues/296
    """
    result = rearrange(x, pattern, **axes_lengths)
    # if x.data_ptr() != result.data_ptr():
    #     raise ValueError(
    #         "rearrange_view could not be applied to the input tensor.\n"
    #         "This is likely because the input tensor is not contiguous "
    #         "or the rearrange pattern is not compatible with the input tensor without copying."
    #     )

    return result


T = TypeVar("T", infer_variance=True)


def _make_error_str(input: Any, t: Any) -> str:
    error_components: list[str] = []
    error_components.append("Type checking error:")
    if hasattr(t, "__instancecheck_str__"):
        error_components.append(t.__instancecheck_str__(input))
    if torch.is_tensor(input):
        error_components.append(repr(lovely(input)))
    error_components.append(shape_str(get_shape_memo()))

    return "\n".join(error_components)


@torch.jit.ignore()
def tassert(t: Any, input: T) -> T:
    """
    Typecheck the input against the given type.

    Args:
        t: Type to check against.
        input: Input to check.
    """

    assert isinstance(input, t), _make_error_str(input, t)
    return input


if TYPE_CHECKING:
    DtypeType = Literal[
        "Float",
        "Double",
        "Int",
        "UInt",
        "Byte",
        "Bool",
        "Shaped",
    ]
else:
    DtypeType = str


@torch.jit.ignore()
def tassert_jitsafe(
    dtype: DtypeType,
    shape: str,
    input: torch.Tensor,
) -> torch.Tensor:
    """
    Typecheck the input against the given type.

    Args:
        t: Type to check against.
        input: Input to check.
    """

    t: Any = getattr(jaxtyping, dtype)[torch.Tensor, shape]

    assert isinstance(input, t), _make_error_str(input, t)
    return input
