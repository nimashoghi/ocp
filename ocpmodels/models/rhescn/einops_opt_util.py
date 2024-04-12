from functools import wraps
from typing import Any, Callable

import opt_einsum_fx
import torch
import torch.fx
from einops import einsum
from typing_extensions import ParamSpec


def einops_to_torch(einsum_str: str):
    """
    E.g., einops einsum format: "first_dim second_dim, second_dim third_dim -> first_dim third_dim"
            => torch einsum format: "ij,jk->ik"
    """
    # First, find all the "lhs" of the einsum
    lhs, rhs = einsum_str.split("->")
    lhs = lhs.strip()
    rhs = rhs.strip()

    # Now, split the lhs into its components
    lhs_components = lhs.split(",")
    lhs_components = [x.strip() for x in lhs_components]

    # Now, split each component into its dimensions
    lhs_components = [
        [term.strip() for term in component.split(" ")]
        for component in lhs_components
    ]

    # Now, split the rhs into its components
    rhs_components = [term.strip() for term in rhs.split(" ")]

    # Now, we need to create a mapping from each dimension to a letter
    all_dims = set([dim for component in lhs_components for dim in component])
    all_dims.update(rhs_components)

    # Now, we need to create a mapping from each dimension to a letter
    dim_to_letter = {dim: chr(97 + i) for i, dim in enumerate(all_dims)}

    # Now, we need to convert the lhs and rhs into the torch einsum format
    lhs = [
        "".join([dim_to_letter[dim] for dim in component])
        for component in lhs_components
    ]
    rhs = "".join([dim_to_letter[dim] for dim in rhs_components])

    return f"{','.join(lhs)}->{rhs}"


def test_einops_einsum_to_torch_einsum():
    og_str = (
        "first_dim second_dim, second_dim third_dim -> first_dim third_dim"
    )
    torch_str = einops_to_torch(og_str)
    print(og_str, torch_str)

    # Let's construct tensors to test this
    first_dim = 24
    second_dim = 32
    third_dim = 64
    x = torch.randn(first_dim, second_dim)
    y = torch.randn(second_dim, third_dim)
    z_expected = einsum(x, y, og_str)

    # Now, let's test the torch einsum
    z = torch.einsum(torch_str, x, y)

    torch.testing.assert_close(z, z_expected)


P = ParamSpec("P")


def optimize_einsum(fn: Callable[P, Any]):
    @wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs):
        if kwargs:
            raise ValueError("kwargs not supported")

        graph_mod = torch.fx.symbolic_trace(fn)
        print("Original code:\n", graph_mod.code)
        graph_opt = opt_einsum_fx.optimize_einsums_full(
            model=graph_mod,
            example_inputs=tuple(args),
        )
        print(f"Optimized code for {fn.__name__}:\n", graph_opt.code)

    return inner
