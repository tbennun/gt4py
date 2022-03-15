import copy
from functools import lru_cache


try:
    import cupy
except ImportError:
    cupy = None
import dace
import hypothesis as hyp
import numpy as np
import pytest
from hypothesis import strategies as hyp_st

from gt4py import backend as gt_backend
from gt4py import gtscript
from gt4py import storage as gt_storage
from gtc.dace.nodes import StencilComputation

from .stencil_definitions import EXTERNALS_REGISTRY as externals_registry
from .stencil_definitions import REGISTRY as stencil_definitions


@lru_cache
def make_base_case(name, backend):
    ref_backend = "gtc:numpy"
    stencil_definition = stencil_definitions[name]
    externals = externals_registry[name]
    stencil = gtscript.stencil(backend=backend, definition=stencil_definition, externals=externals)
    ref_stencil = gtscript.stencil(
        backend=ref_backend, definition=stencil_definition, externals=externals
    )
    args = {}
    for k, v in stencil_definition.__annotations__.items():
        if isinstance(v, gtscript._FieldDescriptor):
            args[k] = gt_storage.ones(
                dtype=(v.dtype, v.data_dims) if v.data_dims else v.dtype,
                mask=gtscript.mask_from_axes(v.axes),
                backend=backend,
                shape=(23, 23, 23),
                default_origin=(10, 10, 10),
            )
        else:
            args[k] = v(1.5)
    sdfg: dace.SDFG = stencil.__sdfg__(**args, origin=(10, 10, 5), domain=(3, 3, 16))

    refout = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in args.items()}
    ref_stencil(**refout, origin=(10, 10, 5), domain=(3, 3, 16))

    for k, v in args.items():
        if isinstance(v, gt_storage.Storage):

            for i, (ax, m) in enumerate(zip("IJK", v.mask)):
                if m:
                    if f"__{k}_{ax}_stride" in sdfg.free_symbols:
                        sdfg.replace(f"__{k}_{ax}_stride", v.strides[sum(v.mask[:i])] // v.itemsize)
                        sdfg.remove_symbol(f"__{k}_{ax}_stride")
                    if f"__{k}_{ax}_size" in sdfg.free_symbols:
                        sdfg.replace(f"__{k}_{ax}_size", v.shape[sum(v.mask[:i])])
                        sdfg.remove_symbol(f"__{k}_{ax}_size")

    expansions_dict = dict()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, StencilComputation):
            expansions_dict[node.name] = list(node.valid_expansion_orders())
    return sdfg, args, refout, expansions_dict


def make_test_case(name, backend):
    sdfg, args, refout, expansions_dict = make_base_case(name, backend)
    next_data = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in args.items()}
    next_refout = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in refout.items()}
    next_sdfg = copy.deepcopy(sdfg)
    return next_data, next_refout, next_sdfg, expansions_dict


@pytest.mark.parametrize("name", stencil_definitions.keys())
@pytest.mark.parametrize(
    "backend", ["gtc:dace", pytest.param("gtc:dace:gpu", marks=[pytest.mark.requires_gpu])]
)
@pytest.mark.expensive
@hyp.given(data=hyp_st.data())
def test_generation(name, backend, data: hyp_st.DataObject):
    if name == "arithmetic_ops":
        pytest.skip("we know this case is bad, not due to expansion.")
    print()
    print("TESTING:")

    try:
        input_data, refout_data, sdfg, expansions_dict = make_test_case(name, backend)
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, StencilComputation):
                expansion = data.draw(hyp_st.sampled_from(expansions_dict[node.name]))
                print(expansion)
                node.expansion_specification = expansion
                print(node.expansion_specification)
    except Exception:
        print("FAILED (CONFIGURING)")
        raise
    try:
        sdfg.expand_library_nodes()
    except Exception:
        print("FAILED (EXPAND)")
        raise
    try:
        sdfg.build_folder = "/dev/shm"
        with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=-1):
            csdfg = sdfg.compile()
    except Exception:
        print("FAILED (COMPILE)")
        raise
    try:
        # this config is so that gt4py storages are allowed as arguments
        with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
            csdfg(**input_data)

        if gt_backend.from_name(backend).storage_info["device"] == "gpu":
            cupy.cuda.Device(0).synchronize()

    except Exception:
        print("FAILED (CALL)")
        raise

    for k, v in refout_data.items():
        try:
            if hasattr(input_data[k], "device_to_host"):
                input_data[k].device_to_host(force=True)
            np.testing.assert_allclose(
                np.array(v),
                np.array(input_data[k]),
            )
        except Exception:
            print("FAILED (VALIDATION)")
            raise
