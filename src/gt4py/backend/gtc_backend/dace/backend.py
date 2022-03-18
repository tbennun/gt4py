# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type

import dace
import numpy as np
from dace.sdfg.utils import fuse_states, inline_sdfgs
from dace.serialize import dumps
from dace.transformation.dataflow import MapCollapse

from eve import NodeVisitor, codegen
from eve.codegen import MakoTemplate as as_mako
from gt4py import backend as gt_backend
from gt4py import gt_src_manager
from gt4py.backend import BaseGTBackend, CLIBackendMixin, make_args_data_from_gtir
from gt4py.backend.gt_backends import (
    GTCUDAPyModuleGenerator,
    cuda_is_compatible_type,
    make_cuda_layout_map,
)
from gt4py.backend.gtc_backend.common import bindings_main_template, pybuffer_to_sid
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.ir import StencilDefinition
from gtc import common, gtir, gtir_to_oir
from gtc.common import CartesianOffset, LevelMarker
from gtc.dace.nodes import (
    HorizontalExecutionLibraryNode,
    StencilComputation,
    VerticalLoopLibraryNode,
)
from gtc.dace.oir_to_dace import OirSDFGBuilder
from gtc.dace.utils import array_dimensions, replace_strides
from gtc.passes.gtir_legacy_extents import compute_legacy_extents
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_optimizations.caches import FillFlushToLocalKCaches
from gtc.passes.oir_optimizations.horizontal_execution_merging import GreedyMerging
from gtc.passes.oir_optimizations.inlining import MaskInlining
from gtc.passes.oir_pipeline import DefaultPipeline


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


def specialize_transient_strides(sdfg: dace.SDFG, layout_map):
    repldict = replace_strides(
        [array for array in sdfg.arrays.values() if array.transient],
        layout_map,
    )
    sdfg.replace_dict(repldict)
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                for k, v in repldict.items():
                    if k in node.symbol_mapping:
                        node.symbol_mapping[k] = v
    for k, v in repldict.items():
        if k in sdfg.symbols:
            sdfg.remove_symbol(k)
        for sym in getattr(v, "free_symbols", set()):
            if str(sym) not in sdfg.symbols:
                sdfg.add_symbol(str(sym), dace.int32)


def post_expand_trafos(sdfg: dace.SDFG):
    while inline_sdfgs(sdfg) or fuse_states(sdfg):
        pass
    sdfg.simplify()
    sdfg.apply_transformations_repeated(
        MapCollapse, validate=False, permissive=False, print_report=True
    )

    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            node.collapse = len(node.range)


def to_device(sdfg: dace.SDFG, device):
    if device == "gpu":
        for array in sdfg.arrays.values():
            array.storage = (
                dace.StorageType.GPU_Global if not array.transient else dace.StorageType.GPU_Global
            )
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, StencilComputation):
                node.device = dace.DeviceType.GPU
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, VerticalLoopLibraryNode):
                node.device = dace.DeviceType.CPU


def pre_expand_trafos(sdfg: dace.SDFG):
    while inline_sdfgs(sdfg) or fuse_states(sdfg):
        pass
    sdfg.simplify()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, StencilComputation):
            print(node.name, node.has_splittable_regions())
            if node.oir_node.loop_order == common.LoopOrder.PARALLEL:
                expansion_priority = []
                if node.has_splittable_regions():
                    expansion_priority.append(["Sections", "Stages", "J", "I", "K"])

                expansion_priority.expand(
                    [
                        ["Sections", "Stages", "K", "J", "I"],
                        ["TileJ", "TileI", "Sections", "KMap", "Stages", "JMap", "IMap"],
                    ]
                )
            else:
                expansion_priority = [
                    ["J", "I", "Sections", "Stages", "K"],
                    ["TileJ", "TileI", "Sections", "KLoop", "Stages", "JMap", "IMap"],
                ]
            for exp in expansion_priority:

                try:
                    node.expansion_specification = exp
                    print("used", exp)
                    is_set = True
                except ValueError:
                    continue
                else:
                    break
            if not is_set:
                raise ValueError("No expansion compatible")


def expand_and_wrap_sdfg(
    gtir: gtir.Stencil, sdfg: dace.SDFG, layout_map
) -> dace.SDFG:  # noqa: C901

    args_data = make_args_data_from_gtir(GtirPipeline(gtir))

    # stencils without effect
    if all(info is None for info in args_data.field_info.values()):
        sdfg = dace.SDFG(gtir.name)
        sdfg.add_state(gtir.name)
        return sdfg

    for array in sdfg.arrays.values():
        if array.transient:
            array.lifetime = dace.AllocationLifetime.Persistent
    sdfg.simplify()
    sdfg.expand_library_nodes(recursive=True)
    specialize_transient_strides(sdfg, layout_map=layout_map)
    post_expand_trafos(sdfg)
    return sdfg


class GTCDaCeExtGenerator:
    def __init__(self, class_name, module_name, backend):
        self.class_name = class_name
        self.module_name = module_name
        self.backend = backend

    def __call__(self, definition_ir: StencilDefinition) -> Dict[str, Dict[str, str]]:
        gtir = GtirPipeline(DefIRToGTIR.apply(definition_ir)).full()
        base_oir = gtir_to_oir.GTIRToOIR().visit(gtir)
        oir_pipeline = self.backend.builder.options.backend_opts.get(
            "oir_pipeline",
            DefaultPipeline(),
        )
        oir_pipeline = DefaultPipeline(
            skip=oir_pipeline.skip
            + [
                GreedyMerging,
                MaskInlining,
                FillFlushToLocalKCaches,
            ]
        )
        oir = oir_pipeline.run(base_oir)
        sdfg = OirSDFGBuilder().visit(oir)
        to_device(sdfg, self.backend.storage_info["device"])
        pre_expand_trafos(sdfg)
        specialize_transient_strides(sdfg, layout_map=self.backend.storage_info["layout_map"])

        for tmp_sdfg in sdfg.all_sdfgs_recursive():
            tmp_sdfg.transformation_hist = []
            tmp_sdfg.orig_sdfg = None

        json_sdfgs = {
            self.backend.builder.module_name + ".sdfg": dumps(sdfg.to_json()),
        }

        if not self.backend.builder.options.backend_opts.get("disable_code_generation", False):
            sdfg = expand_and_wrap_sdfg(gtir, sdfg, self.backend.storage_info["layout_map"])

            for tmp_sdfg in sdfg.all_sdfgs_recursive():
                tmp_sdfg.transformation_hist = []
                tmp_sdfg.orig_sdfg = None

            with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=-1):
                implementation = DaCeComputationCodegen.apply(
                    gtir, sdfg, self.backend.storage_info["layout_map"]
                )

            bindings = DaCeBindingsCodegen.apply(
                gtir, sdfg, module_name=self.module_name, backend=self.backend
            )
            bindings_ext = ".cu" if self.backend.storage_info["device"] == "gpu" else ".cpp"

            computation = {"computation.hpp": implementation}
            bindings = {"bindings" + bindings_ext: bindings}
            json_sdfgs[self.backend.builder.module_name + "_expanded.sdfg"] = dumps(sdfg.to_json())
        else:
            computation = dict()
            bindings = dict()

        return {
            "computation": computation,
            "bindings": bindings,
            "info": json_sdfgs,
        }

        return implementation


class KOriginsVisitor(NodeVisitor):
    def visit_Stencil(self, node: gtir.Stencil):
        k_origins: Dict[str, int] = dict()
        self.generic_visit(node, k_origins=k_origins)
        return k_origins

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs):
        self.generic_visit(node, interval=node.interval, **kwargs)

    def visit_FieldAccess(
        self, node: gtir.FieldAccess, *, k_origins, interval: gtir.Interval
    ) -> None:
        if interval.start.level == LevelMarker.START:
            if not isinstance(node.offset, CartesianOffset):
                candidate = 0
            else:
                candidate = -interval.start.offset - node.offset.k
            candidate = max(0, candidate)
        else:
            candidate = 0
        k_origins[node.name] = max(k_origins.get(node.name, 0), candidate)


def compute_k_origins(node: gtir.Stencil) -> Dict[str, int]:
    return KOriginsVisitor().visit(node)


class DaCeComputationCodegen:

    template = as_mako(
        """
        auto ${name}(const std::array<gt::uint_t, 3>& domain) {
            return [domain](${",".join(functor_args)}) {
                const int __I = domain[0];
                const int __J = domain[1];
                const int __K = domain[2];
                ${name}_t dace_handle;
                auto allocator = gt::sid::make_cached_allocator(&${allocator}<char[]>);
                ${"\\n".join(tmp_allocs)}
                __program_${name}(${",".join(["&dace_handle", *dace_args])});
            };
        }
        """
    )

    def generate_tmp_allocs(self, sdfg):
        fmt = "dace_handle.__{sdfg_id}_{name} = allocate(allocator, gt::meta::lazy::id<{dtype}>(), {size})();"
        return [
            fmt.format(
                sdfg_id=array_sdfg.sdfg_id,
                name=name,
                dtype=array.dtype.ctype,
                size=array.total_size,
            )
            for array_sdfg, name, array in sdfg.arrays_recursive()
            if array.transient and array.lifetime == dace.AllocationLifetime.Persistent
        ]

    @classmethod
    def apply(cls, gtir, sdfg: dace.SDFG, make_layout):
        self = cls()
        code_objects = sdfg.generate_code()
        computations = code_objects[[co.title for co in code_objects].index("Frame")].clean_code
        lines = computations.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if '#include "../../include/hash.h' in line:
                lines = lines[0:i] + lines[i + 1 :]

        is_gpu = "CUDA" in {co.title for co in code_objects}
        if is_gpu:
            for i, line in enumerate(lines):
                if line.strip() == f"struct {sdfg.name}_t {{":
                    j = i + 1
                    while "};" not in lines[j].strip():
                        j += 1

                    lines = lines[0:i] + lines[j + 1 :]
                    break

            for i, line in enumerate(lines):
                if "#include <dace/dace.h>" in line:
                    cuda_code = [co.clean_code for co in code_objects if co.title == "CUDA"]
                    lines = lines[0:i] + cuda_code[0].split("\n") + lines[i + 1 :]
                    break
            for i, line in reversed(list(enumerate(lines))):
                line = line.strip()
                if line.startswith("DACE_EXPORTED") and line.endswith(");"):
                    lines = lines[0:i] + lines[i + 1 :]

        computations = codegen.format_source("cpp", "\n".join(lines), style="LLVM")
        if "Min(" in computations:
            lines = computations.split("\n")
            pos = lines.index("#include <dace/dace.h>")
            lines.insert(pos + 1, "#define Min(x, y) ((x) < (y) ? (x) : (y))")
            computations = "\n".join(lines)

        interface = cls.template.definition.render(
            name=sdfg.name,
            dace_args=self.generate_dace_args(gtir, sdfg, make_layout),
            functor_args=self.generate_functor_args(sdfg),
            tmp_allocs=self.generate_tmp_allocs(sdfg),
            allocator="gt::cuda_util::cuda_malloc" if is_gpu else "std::make_unique",
        )
        generated_code = f"""#include <gridtools/sid/sid_shift_origin.hpp>
                             #include <gridtools/sid/allocator.hpp>
                             #include <gridtools/stencil/cartesian.hpp>
                             {"#include <gridtools/common/cuda_util.hpp>" if is_gpu else ""}
                             namespace gt = gridtools;
                             {computations}

                             {interface}
                             """
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code

    def __init__(self):
        self._unique_index = 0

    def generate_dace_args(self, gtir, sdfg, make_layout):
        offset_dict: Dict[str, Tuple[int, int, int]] = {
            k: (-v[0][0], -v[1][0], -v[2][0]) for k, v in compute_legacy_extents(gtir).items()
        }
        k_origins = compute_k_origins(gtir)
        for name, origin in k_origins.items():
            offset_dict[name] = (offset_dict[name][0], offset_dict[name][1], origin)

        symbols = {f"__{var}": f"__{var}" for var in "IJK"}
        for name, array in sdfg.arrays.items():
            if array.transient:
                from gt4py.storage.utils import idx_from_order, strides_from_padded_shape

                mask = array_dimensions(array)
                layout_map = make_layout(mask)
                order_idx = idx_from_order([i for i in layout_map if i is not None])
                strides = strides_from_padded_shape(
                    array.shape, order_idx, dace.symbolic.pystr_to_symbolic("1")
                )
                dim_gen = (d for i, d in enumerate("IJK") if mask[i] is not None)
                for dim, stride in zip(dim_gen, strides):
                    symbols[f"__{name}_{dim}_stride"] = str(stride)
            else:
                dims = [dim for dim, select in zip("IJK", array_dimensions(array)) if select]
                data_ndim = len(array.shape) - len(dims)

                # api field strides
                fmt = "gt::sid::get_stride<{dim}>(gt::sid::get_strides(__{name}_sid))"

                symbols.update(
                    {
                        f"__{name}_{dim}_stride": fmt.format(
                            dim=f"gt::stencil::dim::{dim.lower()}", name=name
                        )
                        for dim in dims
                    }
                )
                symbols.update(
                    {
                        f"__{name}_d{dim}_stride": fmt.format(
                            dim=f"gt::integral_constant<int, {3 + dim}>", name=name
                        )
                        for dim in range(data_ndim)
                    }
                )

                # api field pointers
                fmt = """gt::sid::multi_shifted(
                             gt::sid::get_origin(__{name}_sid)(),
                             gt::sid::get_strides(__{name}_sid),
                             std::array<gt::int_t, {ndim}>{{{origin}}}
                         )"""
                origin = tuple(
                    -offset_dict[name][idx]
                    for idx, var in enumerate("IJK")
                    if any(
                        dace.symbolic.pystr_to_symbolic(f"__{var}") in s.free_symbols
                        for s in array.shape
                        if hasattr(s, "free_symbols")
                    )
                )
                symbols[name] = fmt.format(
                    name=name,
                    ndim=len(array.shape),
                    origin=",".join(str(o) for o in origin),
                )
        # the remaining arguments are variables and can be passed by name
        for sym in sdfg.signature_arglist(with_types=False, for_call=True):
            if sym not in symbols:
                symbols[sym] = sym

        # return strings in order of sdfg signature
        return [symbols[s] for s in sdfg.signature_arglist(with_types=False, for_call=True)]

    def generate_functor_args(self, sdfg: dace.SDFG):
        res = []
        for name, array in sdfg.arrays.items():
            if array.transient:
                continue
            res.append(f"auto && __{name}_sid")
        for name, dtype in (
            (n, d) for n, d in sorted(sdfg.symbols.items()) if not n.startswith("__")
        ):
            res.append(dtype.as_arg(name))
        return res


class DaCeBindingsCodegen:
    def __init__(self, backend):
        self.backend = backend
        self._unique_index: int = 0

    def unique_index(self) -> int:
        self._unique_index += 1
        return self._unique_index

    mako_template = bindings_main_template()

    def generate_entry_params(self, gtir: gtir.Stencil, sdfg: dace.SDFG):
        res = {}
        import dace.data

        for name in sdfg.signature_arglist(with_types=False, for_call=True):
            if name in sdfg.arrays:
                data = sdfg.arrays[name]
                assert isinstance(data, dace.data.Array)
                res[name] = "py::buffer {name}, std::array<gt::uint_t,{ndim}> {name}_origin".format(
                    name=name,
                    ndim=len(data.shape),
                )
            elif name in sdfg.symbols and not name.startswith("__"):
                assert name in sorted(sdfg.symbols)
                res[name] = "{dtype} {name}".format(dtype=sdfg.symbols[name].ctype, name=name)
        return list(res[node.name] for node in gtir.params if node.name in res)

    def generate_sid_params(self, sdfg: dace.SDFG):
        res = []
        import dace.data

        for name, array in sdfg.arrays.items():
            if array.transient:
                continue
            domain_dim_flags = tuple(
                True
                if any(
                    dace.symbolic.pystr_to_symbolic(f"__{dim.upper()}") in s.free_symbols
                    for s in array.shape
                    if hasattr(s, "free_symbols")
                )
                else False
                for dim in "ijk"
            )
            data_ndim = len(array.shape) - sum(array_dimensions(array))
            sid_def = pybuffer_to_sid(
                name=name,
                ctype=array.dtype.ctype,
                domain_dim_flags=domain_dim_flags,
                data_ndim=data_ndim,
                stride_kind_index=self.unique_index(),
                check_layout=False,
                backend=self.backend,
            )

            res.append(sid_def)
        # pass scalar parameters as variables
        for name in sorted(n for n in sdfg.symbols.keys() if not n.startswith("__")):
            res.append(name)
        return res

    def generate_sdfg_bindings(self, gtir, sdfg, module_name):

        return self.mako_template.render_values(
            name=sdfg.name,
            module_name=module_name,
            entry_params=self.generate_entry_params(gtir, sdfg),
            sid_params=self.generate_sid_params(sdfg),
        )

    @classmethod
    def apply(cls, gtir: gtir.Stencil, sdfg: dace.SDFG, module_name: str, *, backend) -> str:
        generated_code = cls(backend).generate_sdfg_bindings(gtir, sdfg, module_name=module_name)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


class DaCePyExtModuleGenerator(gt_backend.PyExtModuleGenerator):
    def generate_imports(self):
        res = super().generate_imports()
        return res + "\nimport dace\nimport copy"

    def generate_class_members(self):
        res = super().generate_class_members()
        filepath = self.builder.module_path.joinpath(
            os.path.dirname(self.builder.module_path),
            self.builder.module_name + "_pyext_BUILD",
            self.builder.module_name + ".sdfg",
        )
        res += """
_sdfg = None

def __new__(cls, *args, **kwargs):
    res = super().__new__(cls, *args, **kwargs)
    cls._sdfg = dace.SDFG.from_file('{filepath}')
    return res

@property
def sdfg(self) -> dace.SDFG:
    return copy.deepcopy(self._sdfg)


""".format(
            filepath=filepath
        )
        return res


class DaCeCUDAPyModuleGenerator(DaCePyExtModuleGenerator, GTCUDAPyModuleGenerator):
    pass


class BaseGTCDaceBackend(BaseGTBackend, CLIBackendMixin):
    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources(2) and not gt_src_manager.install_gt_sources(2):
            raise RuntimeError("Missing GridTools sources.")

        pyext_module_name: Optional[str]
        pyext_file_path: Optional[str]

        # TODO(havogt) add bypass if computation has no effect
        pyext_module_name, pyext_file_path = self.generate_extension()

        # Generate and return the Python wrapper class
        return self.make_module(
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )

    def generate_extension(self) -> Tuple[str, str]:
        return self.make_extension(
            gt_version=2,
            ir=self.builder.definition_ir,
            uses_cuda=self.storage_info["device"] == "gpu",
        )


def layout_maker_factory(base_layout):
    def layout_maker(mask):
        ranks = []
        for m, l in zip(mask, base_layout):
            if m:
                ranks.append(l)
        if len(mask) > 3:
            if base_layout[2] == 2:
                ranks.extend(3 + c for c in range(len(mask) - 3))
            else:
                ranks.extend(-c for c in range(len(mask) - 3))

        res_layout = [0] * len(ranks)
        for i, idx in enumerate(np.argsort(ranks)):
            res_layout[idx] = i
        return tuple(res_layout)

    return layout_maker


@gt_backend.register
class GTCDaceBackend(BaseGTCDaceBackend):
    """DaCe python backend using gtc."""

    name = "gtc:dace"
    GT_BACKEND_T = "dace"
    languages = {"computation": "c++", "bindings": ["python"]}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": layout_maker_factory((1, 0, 2)),
        "is_compatible_layout": lambda x: True,
        "is_compatible_type": lambda x: isinstance(x, np.ndarray),
    }
    MODULE_GENERATOR_CLASS = DaCePyExtModuleGenerator

    options = BaseGTBackend.GT_BACKEND_OPTS
    PYEXT_GENERATOR_CLASS = GTCDaCeExtGenerator  # type: ignore
    USE_LEGACY_TOOLCHAIN = False


@gt_backend.register
class GTCDaceGPUBackend(BaseGTCDaceBackend):
    """DaCe python backend using gtc."""

    name = "gtc:dace:gpu"
    GT_BACKEND_T = "dace"
    languages = {"computation": "c++", "bindings": ["python"]}
    storage_info = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": make_cuda_layout_map,
        "is_compatible_layout": lambda x: True,
        "is_compatible_type": cuda_is_compatible_type,
    }
    options = {
        **BaseGTBackend.GT_BACKEND_OPTS,
        "device_sync": {"versioning": True, "type": bool},
    }
    PYEXT_GENERATOR_CLASS = GTCDaCeExtGenerator  # type: ignore
    MODULE_GENERATOR_CLASS = DaCeCUDAPyModuleGenerator
    USE_LEGACY_TOOLCHAIN = False
