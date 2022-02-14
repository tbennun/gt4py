# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import dace
import dace.data
import dace.library
import dace.subsets
import numpy as np
import sympy

import eve
import gtc.common as common
import gtc.oir as oir
from eve import NodeTranslator, NodeVisitor, codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from gt4py import definitions as gt_def
from gt4py.definitions import Extent
from gtc import daceir as dcir
from gtc.dace.nodes import StencilComputation
from gtc.dace.utils import get_axis_bound_str, get_tasklet_symbol
from gtc.passes.oir_optimizations.utils import AccessCollector

from .utils import compute_dcir_access_infos, make_subset_str


def make_access_subset_dict(
    extent: gt_def.Extent, interval: oir.Interval, axes: List[dcir.Axis]
) -> Dict[dcir.Axis, Union[dcir.IndexWithExtent, dcir.DomainInterval]]:
    from gtc import daceir as dcir

    i_interval = dcir.DomainInterval(
        start=dcir.AxisBound(level=common.LevelMarker.START, offset=extent[0][0], axis=dcir.Axis.I),
        end=dcir.AxisBound(level=common.LevelMarker.END, offset=extent[0][1], axis=dcir.Axis.I),
    )
    j_interval = dcir.DomainInterval(
        start=dcir.AxisBound(level=common.LevelMarker.START, offset=extent[1][0], axis=dcir.Axis.J),
        end=dcir.AxisBound(level=common.LevelMarker.END, offset=extent[1][1], axis=dcir.Axis.J),
    )
    k_interval: Union[dcir.IndexWithExtent, dcir.DomainInterval]
    if isinstance(interval, dcir.IndexWithExtent):
        k_interval = interval
    else:
        k_interval = dcir.DomainInterval(
            start=dcir.AxisBound(
                level=interval.start.level, offset=interval.start.offset, axis=dcir.Axis.K
            ),
            end=dcir.AxisBound(
                level=interval.end.level, offset=interval.end.offset, axis=dcir.Axis.K
            ),
        )
    res = {dcir.Axis.I: i_interval, dcir.Axis.J: j_interval, dcir.Axis.K: k_interval}
    return {axis: res[axis] for axis in axes}


def make_read_accesses(node: common.Node, *, global_ctx, **kwargs):
    return compute_dcir_access_infos(
        node,
        block_extents=global_ctx.block_extents,
        oir_decls=global_ctx.library_node.declarations,
        collect_read=True,
        collect_write=False,
        **kwargs,
    )


def make_write_accesses(node: common.Node, *, global_ctx, **kwargs):
    return compute_dcir_access_infos(
        node,
        block_extents=global_ctx.block_extents,
        oir_decls=global_ctx.library_node.declarations,
        collect_read=False,
        collect_write=True,
        **kwargs,
    )


class TaskletCodegen(codegen.TemplatedGenerator):

    ScalarAccess = as_fmt("{name}")

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        is_target,
        targets,
        read_accesses: Dict[str, dcir.FieldAccessInfo],
        write_accesses: Dict[str, dcir.FieldAccessInfo],
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ):
        field_decl = sdfg_ctx.field_decls[node.name]
        is_offset_nil = (
            isinstance(node.offset, common.CartesianOffset)
            and self.visit(node.offset, is_dynamic_offset=False, field_decl=field_decl) == ""
        )

        if is_target or (node.name in targets and is_offset_nil):
            access_info = write_accesses[node.name]
            is_dynamic_offset = len(access_info.variable_offset_axes) > 0
        else:
            access_info = read_accesses[node.name]
            is_dynamic_offset = len(access_info.variable_offset_axes) > 0

        if is_target or (node.name in targets and (is_offset_nil or is_dynamic_offset)):
            targets.add(node.name)
            name = "__" + node.name
        elif is_dynamic_offset:
            name = node.name + "__"
        else:
            name = (
                node.name
                + "__"
                + self.visit(
                    node.offset,
                    is_dynamic_offset=False,
                    is_target=is_target,
                    targets=targets,
                    field_decl=field_decl,
                    access_info=access_info,
                    sdfg_ctx=sdfg_ctx,
                    read_accesses=read_accesses,
                    write_accesses=write_accesses,
                )
            )
        if node.data_index or is_dynamic_offset:
            offset_str = "["
            if is_dynamic_offset:
                offset_str += self.visit(
                    node.offset,
                    is_dynamic_offset=True,
                    is_target=is_target,
                    targets=targets,
                    field_decl=field_decl,
                    access_info=access_info,
                    sdfg_ctx=sdfg_ctx,
                    read_accesses=read_accesses,
                    write_accesses=write_accesses,
                )
            if node.data_index:
                offset_str += ",".join(self.visit(node.data_index))
            offset_str += "]"
        else:
            offset_str = ""
        return name + offset_str

    def visit_CartesianOffset(
        self,
        node: common.CartesianOffset,
        *,
        is_dynamic_offset,
        field_decl,
        access_info=None,
        **kwargs,
    ):
        if is_dynamic_offset:
            return self.visit_VariableKOffset(
                node,
                is_dynamic_offset=is_dynamic_offset,
                field_decl=field_decl,
                access_info=access_info,
                **kwargs,
            )
        else:
            res = []
            if node.i != 0:
                res.append(f'i{"m" if node.i<0 else "p"}{abs(node.i):d}')
            if node.j != 0:
                res.append(f'j{"m" if node.j<0 else "p"}{abs(node.j):d}')
            if node.k != 0:
                res.append(f'k{"m" if node.k<0 else "p"}{abs(node.k):d}')
            return "_".join(res)

    def visit_VariableKOffset(
        self,
        node: Union[oir.VariableKOffset, common.CartesianOffset],
        *,
        is_dynamic_offset,
        field_decl,
        **kwargs,
    ):
        assert is_dynamic_offset
        offset_strs = []
        for axis in field_decl.axes():
            if axis == dcir.Axis.K:
                offset_strs.append(axis.iteration_symbol() + f"+({self.visit(node.k, **kwargs)})")
            else:
                offset_strs.append(axis.iteration_symbol())
        res: dace.subsets.Range = StencilComputationSDFGBuilder._add_origin(
            field_decl.access_info, ",".join(offset_strs), add_for_variable=True
        )
        res.ranges = [res.ranges[list(field_decl.axes()).index(dcir.Axis.K)]]
        return str(res)

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs):
        right = self.visit(node.right, is_target=False, **kwargs)
        left = self.visit(node.left, is_target=True, **kwargs)
        return f"{left} = {right}"

    BinaryOp = as_fmt("({left} {op} {right})")

    UnaryOp = as_fmt("({op}{expr})")

    TernaryOp = as_fmt("({true_expr} if {cond} else {false_expr})")

    def visit_BuiltInLiteral(self, builtin: common.BuiltInLiteral, **kwargs: Any) -> str:
        if builtin == common.BuiltInLiteral.TRUE:
            return "True"
        elif builtin == common.BuiltInLiteral.FALSE:
            return "False"
        raise NotImplementedError("Not implemented BuiltInLiteral encountered.")

    Literal = as_fmt("{value}")

    Cast = as_fmt("{dtype}({expr})")

    def visit_NativeFunction(self, func: common.NativeFunction, **kwargs: Any) -> str:
        try:
            return {
                common.NativeFunction.ABS: "abs",
                common.NativeFunction.MIN: "min",
                common.NativeFunction.MAX: "max",
                common.NativeFunction.MOD: "fmod",
                common.NativeFunction.SIN: "dace.math.sin",
                common.NativeFunction.COS: "dace.math.cos",
                common.NativeFunction.TAN: "dace.math.tan",
                common.NativeFunction.ARCSIN: "asin",
                common.NativeFunction.ARCCOS: "acos",
                common.NativeFunction.ARCTAN: "atan",
                common.NativeFunction.SINH: "dace.math.sinh",
                common.NativeFunction.COSH: "dace.math.cosh",
                common.NativeFunction.TANH: "dace.math.tanh",
                common.NativeFunction.ARCSINH: "asinh",
                common.NativeFunction.ARCCOSH: "acosh",
                common.NativeFunction.ARCTANH: "atanh",
                common.NativeFunction.SQRT: "dace.math.sqrt",
                common.NativeFunction.POW: "dace.math.pow",
                common.NativeFunction.EXP: "dace.math.exp",
                common.NativeFunction.LOG: "dace.math.log",
                common.NativeFunction.GAMMA: "tgamma",
                common.NativeFunction.CBRT: "cbrt",
                common.NativeFunction.ISFINITE: "isfinite",
                common.NativeFunction.ISINF: "isinf",
                common.NativeFunction.ISNAN: "isnan",
                common.NativeFunction.FLOOR: "dace.math.ifloor",
                common.NativeFunction.CEIL: "ceil",
                common.NativeFunction.TRUNC: "trunc",
            }[func]
        except KeyError as error:
            raise NotImplementedError("Not implemented NativeFunction encountered.") from error

    NativeFuncCall = as_mako("${func}(${','.join(args)})")

    def visit_DataType(self, dtype: common.DataType, **kwargs: Any) -> str:
        if dtype == common.DataType.BOOL:
            return "dace.bool_"
        elif dtype == common.DataType.INT8:
            return "dace.int8"
        elif dtype == common.DataType.INT16:
            return "dace.int16"
        elif dtype == common.DataType.INT32:
            return "dace.int32"
        elif dtype == common.DataType.INT64:
            return "dace.int64"
        elif dtype == common.DataType.FLOAT32:
            return "dace.float32"
        elif dtype == common.DataType.FLOAT64:
            return "dace.float64"
        raise NotImplementedError("Not implemented DataType encountered.")

    def visit_UnaryOperator(self, op: common.UnaryOperator, **kwargs: Any) -> str:
        if op == common.UnaryOperator.NOT:
            return " not "
        elif op == common.UnaryOperator.NEG:
            return "-"
        elif op == common.UnaryOperator.POS:
            return "+"
        raise NotImplementedError("Not implemented UnaryOperator encountered.")

    Arg = as_fmt("{name}")

    Param = as_fmt("{name}")

    LocalScalar = as_fmt("{name}: {dtype}")

    def visit_Tasklet(self, node: dcir.Tasklet, **kwargs):
        return "\n".join(self.visit(node.stmts, targets=set(), **kwargs))

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs):
        mask_str = ""
        indent = ""
        if node.mask is not None:
            mask_str = f"if {self.visit(node.mask, is_target=False, **kwargs)}:"
            indent = "    "
        body_code = self.visit(node.body, **kwargs)
        body_code = [indent + b for b in body_code]
        return "\n".join([mask_str] + body_code)

    def visit_While(self, node: oir.While, **kwargs: Any):
        body = self.visit(node.body, **kwargs)
        body = [line for block in body for line in block.split("\n")]
        cond = self.visit(node.cond, is_target=False, **kwargs)
        init = "num_iter = 0"
        max_iter = 1000
        cond += f" and (num_iter < {max_iter})"
        body.append("num_iter += 1")
        indent = " " * 4
        delim = f"\n{indent}"
        code_as_str = f"{init}\nwhile {cond}:\n{indent}{delim.join(body)}"
        return code_as_str

    def visit_HorizontalMask(self, node: oir.HorizontalMask, **kwargs):
        clauses: List[str] = []
        imin = get_axis_bound_str(node.i.start, dcir.Axis.I.domain_symbol())
        if imin:
            clauses.append(f"{dcir.Axis.I.iteration_symbol()} >= {imin}")
        imax = get_axis_bound_str(node.i.end, dcir.Axis.I.domain_symbol())
        if imax:
            clauses.append(f"{dcir.Axis.I.iteration_symbol()} < {imax}")
        jmin = get_axis_bound_str(node.j.start, dcir.Axis.J.domain_symbol())
        if jmin:
            clauses.append(f"{dcir.Axis.J.iteration_symbol()} >= {jmin}")
        jmax = get_axis_bound_str(node.j.end, dcir.Axis.J.domain_symbol())
        if jmax:
            clauses.append(f"{dcir.Axis.J.iteration_symbol()} < {jmax}")
        return " and ".join(clauses)

    class RemoveCastInIndexVisitor(eve.NodeTranslator):
        def visit_FieldAccess(self, node: oir.FieldAccess):
            if node.data_index:
                return self.generic_visit(node, in_idx=True)
            else:
                return self.generic_visit(node)

        def visit_Cast(self, node: oir.Cast, in_idx=False):
            if in_idx:
                return node.expr
            else:
                return node

        def visit_Literal(self, node: oir.Cast, in_idx=False):
            if in_idx:
                return node
            else:
                return oir.Cast(dtype=node.dtype, expr=node)

    @classmethod
    def apply(cls, node: oir.HorizontalExecution, **kwargs: Any) -> str:
        preprocessed_node = cls.RemoveCastInIndexVisitor().visit(node)
        if not isinstance(node, oir.HorizontalExecution):
            raise ValueError("apply() requires oir.HorizontalExecution node")
        generated_code = super().apply(preprocessed_node)
        formatted_code = codegen.format_source("python", generated_code)
        return formatted_code


class DaCeIRBuilder(NodeTranslator):
    @dataclass
    class GlobalContext:
        library_node: StencilComputation
        block_extents: Dict[int, Extent]
        arrays: Dict[str, dace.data.Data]

        def get_dcir_decls(self, access_infos):
            return {
                field: self.get_dcir_decl(field, access_info)
                for field, access_info in access_infos.items()
            }

        def get_dcir_decl(self, field, access_info):
            oir_decl: oir.FieldDecl = self.library_node.declarations[field]
            dace_array = self.arrays[field]
            return dcir.FieldDecl(
                name=field,
                dtype=oir_decl.dtype,
                strides=[str(s) for s in dace_array.strides],
                data_dims=oir_decl.data_dims,
                access_info=access_info,
            )

    @dataclass
    class SymbolCollector:
        symbols: Dict[str, common.DataType]

    @dataclass
    class IterationContext:
        grid_subset: dcir.GridSubset

        def restricted_grid_subset(self, axis: dcir.Axis):
            grid_subset = self.grid_subset.restricted_to_index(axis)
            return DaCeIRBuilder.IterationContext(
                grid_subset=grid_subset,
            )

        def restricted_to_interval(self, axis: dcir.Axis, interval: dcir.DomainInterval):

            return DaCeIRBuilder.IterationContext(
                grid_subset=self.grid_subset.set_interval(axis, interval),
            )

    def _add_domain_loop(
        self,
        *dcir_nodes: eve.Node,
        axis: dcir.Axis,
        interval: Union[dcir.DomainInterval, oir.Interval, dcir.TileInterval],
        grid_subset: dcir.GridSubset,
        loop_order: common.LoopOrder,
    ):

        from .utils import union_node_access_infos

        read_accesses, write_accesses, _ = union_node_access_infos(list(dcir_nodes))
        read_accesses = {
            key: access_info.apply_iteration(dcir.GridSubset.from_interval(interval, axis))
            for key, access_info in read_accesses.items()
        }
        write_accesses = {
            key: access_info.apply_iteration(dcir.GridSubset.from_interval(interval, axis))
            for key, access_info in write_accesses.items()
        }

        stride = -1 if loop_order == common.LoopOrder.BACKWARD else 1
        if isinstance(interval, oir.Interval):
            start, end = (
                dcir.AxisBound.from_common(axis, interval.start),
                dcir.AxisBound.from_common(axis, interval.end),
            )
        else:
            start, end = interval.idx_range
        if loop_order == common.LoopOrder.BACKWARD:
            start, end = f"({end}{stride:+1})", f"({start}{stride:+1})"

        index_range = dcir.Range(var=axis.iteration_symbol(), start=start, end=end, stride=stride)

        return dcir.DomainLoop(
            index_range=index_range,
            loop_states=self.to_state(dcir_nodes, grid_subset=grid_subset),
            grid_subset=grid_subset,
            read_accesses=read_accesses,
            write_accesses=write_accesses,
        )

    def _add_domain_map(
        self,
        *dcir_nodes: eve.Node,
        axis: dcir.Axis,
        interval: dcir.DomainInterval,
        grid_subset: dcir.GridSubset,
        global_ctx: "DaCeIRBuilder.GlobalContext",
    ):
        from .utils import union_node_access_infos

        index_range = dcir.Range.from_axis_and_interval(axis, interval)

        read_accesses, write_accesses, _ = union_node_access_infos(list(dcir_nodes))

        read_accesses = {
            key: access_info.apply_iteration(dcir.GridSubset.from_interval(interval, axis))
            for key, access_info in read_accesses.items()
        }
        write_accesses = {
            key: access_info.apply_iteration(dcir.GridSubset.from_interval(interval, axis))
            for key, access_info in write_accesses.items()
        }

        return dcir.DomainMap(
            index_range=index_range,
            computations=self.to_dataflow(dcir_nodes, global_ctx=global_ctx),
            grid_subset=grid_subset,
            read_accesses=read_accesses,
            write_accesses=write_accesses,
        )

    def _add_domain_iteration(
        self,
        *dcir_nodes: eve.Node,
        axis: dcir.Axis,
        interval: dcir.DomainInterval,
        grid_subset: dcir.GridSubset,
        loop_order: common.LoopOrder,
        global_ctx: "DaCeIRBuilder.GlobalContext",
    ):
        if axis == dcir.Axis.K and not loop_order == common.LoopOrder.PARALLEL:
            return self._add_domain_loop(
                *dcir_nodes,
                axis=axis,
                interval=interval,
                grid_subset=grid_subset,
                loop_order=loop_order,
            )
        else:
            return self._add_domain_map(
                *dcir_nodes,
                axis=axis,
                interval=interval,
                grid_subset=grid_subset,
                global_ctx=global_ctx,
            )

    def _add_tiling_map(
        self,
        *dcir_nodes: eve.Node,
        global_ctx: "DaCeIRBuilder.GlobalContext",
    ):
        from .utils import union_node_access_infos, union_node_grid_subsets, untile_access_info_dict

        grid_subset = union_node_grid_subsets(list(dcir_nodes))
        read_accesses, write_accesses, _ = union_node_access_infos(list(dcir_nodes))
        computations = self.to_dataflow(dcir_nodes, global_ctx=global_ctx)
        for axis, tile_size in global_ctx.library_node.tile_sizes.items():
            axis = dcir.Axis(axis)
            grid_subset = grid_subset.untile(axis)
            read_accesses = untile_access_info_dict(read_accesses, axes=[axis])
            write_accesses = untile_access_info_dict(write_accesses, axes=[axis])
            computations = [
                dcir.DomainMap(
                    computations=computations,
                    index_range=dcir.Range(
                        var=axis.tile_symbol(),
                        start=dcir.AxisBound.from_common(axis, oir.AxisBound.start()),
                        end=dcir.AxisBound.from_common(axis, oir.AxisBound.end()),
                        stride=tile_size,
                    ),
                    read_accesses=read_accesses,
                    write_accesses=write_accesses,
                    grid_subset=grid_subset,
                )
            ]
        return computations

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        global_ctx: "DaCeIRBuilder.GlobalContext",
        iteration_ctx,
        expansion_order,
        loop_order,
        **kwargs,
    ):
        extent = global_ctx.block_extents[id(node)]
        decls = [self.visit(decl) for decl in node.declarations]
        stmts = [self.visit(stmt) for stmt in node.body]

        domain_subset = dcir.GridSubset.from_gt4py_extent(extent).set_interval(
            dcir.Axis.K, iteration_ctx.grid_subset.intervals[dcir.Axis.K]
        )

        if "Tiling" in expansion_order:
            local_subset = domain_subset.tile(global_ctx.library_node.tile_sizes)
        else:
            local_subset = domain_subset

        grid_subset = dcir.GridSubset.single_gridpoint()

        tasklet_read_accesses = make_read_accesses(
            node,
            global_ctx=global_ctx,
            grid_subset=grid_subset,
        )

        tasklet_write_accesses = make_write_accesses(
            node,
            global_ctx=global_ctx,
            grid_subset=grid_subset,
        )

        dcir_node = dcir.Tasklet(
            stmts=decls + stmts,
            read_accesses=tasklet_read_accesses,
            write_accesses=tasklet_write_accesses,
        )

        for axis in reversed(expansion_order):
            if axis not in set(dcir.Axis.dims_3d()):
                break
            axis = dcir.Axis(axis)
            interval = local_subset.intervals[axis]
            grid_subset = grid_subset.set_interval(axis, interval)
            dcir_node = self._add_domain_iteration(
                dcir_node,
                axis=axis,
                interval=interval,
                grid_subset=grid_subset,
                loop_order=loop_order,
                global_ctx=global_ctx,
            )
        return dcir_node

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoopSection,
        *,
        loop_order,
        iteration_ctx: "DaCeIRBuilder.IterationContext",
        global_ctx: "DaCeIRBuilder.GlobalContext",
        expansion_order: List[str],
        **kwargs,
    ):
        section_iteration_ctx = iteration_ctx.restricted_to_interval(dcir.Axis.K, node.interval)

        if expansion_order.index("K") < expansion_order.index("Stages"):
            inner_iteration_ctx = section_iteration_ctx.restricted_grid_subset(dcir.Axis.K)
        else:
            inner_iteration_ctx = section_iteration_ctx

        he_nodes = self.generic_visit(
            node.horizontal_executions,
            iteration_ctx=inner_iteration_ctx,
            global_ctx=global_ctx,
            expansion_order=expansion_order,
            loop_order=loop_order,
            **kwargs,
        )
        scope_nodes = [
            state
            for he in he_nodes
            for state in self.to_state(
                [he],
                grid_subset=he.grid_subset,
            )
        ]
        if expansion_order.index("K") < expansion_order.index("Stages"):
            return self._add_domain_iteration(
                *scope_nodes,
                axis=dcir.Axis.K,
                interval=node.interval,
                grid_subset=iteration_ctx.grid_subset,
                loop_order=loop_order,
                global_ctx=global_ctx,
            )
        else:
            return scope_nodes

    def to_dataflow(
        self,
        nodes,
        *,
        global_ctx: "DaCeIRBuilder.GlobalContext",
    ):
        from .utils import flatten_list

        nodes = flatten_list(nodes)
        if all(isinstance(n, (dcir.StateMachine, dcir.DomainMap, dcir.Tasklet)) for n in nodes):
            return nodes
        elif not all(isinstance(n, (dcir.ComputationState, dcir.DomainLoop)) for n in nodes):
            raise ValueError("Can't mix dataflow and state nodes on same level.")

        from .utils import union_node_access_infos

        read_accesses, write_accesses, field_accesses = union_node_access_infos(nodes)

        declared_symbols = set(
            n.name
            for node in nodes
            for n in node.iter_tree().if_isinstance(oir.ScalarDecl, oir.LocalScalar)
        )
        symbols = dict()

        for name in field_accesses.keys():
            for s in global_ctx.arrays[name].strides:
                for sym in dace.symbolic.symlist(s).values():
                    symbols[str(sym)] = common.DataType.INT32
        for node in nodes:
            for acc in node.iter_tree().if_isinstance(oir.ScalarAccess):
                if acc.name not in declared_symbols:
                    declared_symbols.add(acc.name)
                    symbols[acc.name] = acc.dtype

        for axis in dcir.Axis.dims_3d():
            if axis.domain_symbol() not in declared_symbols:
                declared_symbols.add(axis.domain_symbol())
                symbols[axis.domain_symbol()] = common.DataType.INT32

        return [
            dcir.StateMachine(
                field_decls=global_ctx.get_dcir_decls(field_accesses),
                symbols=symbols,
                # NestedSDFG must have same shape on input and output, matching corresponding
                # nsdfg.sdfg's array shape
                read_accesses={key: field_accesses[key] for key in read_accesses.keys()},
                write_accesses={key: field_accesses[key] for key in write_accesses.keys()},
                states=nodes,
            )
        ]

    def to_state(self, nodes, *, grid_subset: dcir.GridSubset):
        from .utils import flatten_list

        nodes = flatten_list(nodes)
        if all(isinstance(n, (dcir.ComputationState, dcir.DomainLoop)) for n in nodes):
            return nodes
        elif all(isinstance(n, (dcir.StateMachine, dcir.DomainMap, dcir.Tasklet)) for n in nodes):
            return [dcir.ComputationState(computations=nodes, grid_subset=grid_subset)]
        else:
            raise ValueError("Can't mix dataflow and state nodes on same level.")

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        *,
        global_ctx: "DaCeIRBuilder.GlobalContext",
        iteration_ctx: "DaCeIRBuilder.IterationContext",
        **kwargs,
    ):
        from .utils import flatten_list, union_node_access_infos

        computations = flatten_list(
            self.generic_visit(
                node.sections,
                loop_order=node.loop_order,
                global_ctx=global_ctx,
                iteration_ctx=iteration_ctx,
                **kwargs,
            )
        )

        if "Tiling" in global_ctx.library_node.expansion_order:
            computations = [
                self._add_tiling_map(
                    self.to_dataflow(computations, global_ctx=global_ctx), global_ctx=global_ctx
                )
            ]

        read_accesses, write_accesses, field_accesses = union_node_access_infos(computations)

        declared_symbols = set(
            n.name for n in node.iter_tree().if_isinstance(oir.ScalarDecl, oir.LocalScalar)
        )
        symbols = dict()
        for acc in node.iter_tree().if_isinstance(oir.ScalarAccess):
            if acc.name not in declared_symbols:
                declared_symbols.add(acc.name)
                symbols[acc.name] = acc.dtype
        for axis in dcir.Axis.dims_3d():
            if axis.domain_symbol() not in declared_symbols:
                declared_symbols.add(axis.domain_symbol())
                symbols[axis.domain_symbol()] = common.DataType.INT32
        for name in global_ctx.get_dcir_decls(field_accesses).keys():
            for s in global_ctx.arrays[name].strides:
                for sym in dace.symbolic.symlist(s).values():
                    symbols[str(sym)] = common.DataType.INT32

        return dcir.StateMachine(
            states=self.to_state(computations, grid_subset=iteration_ctx.grid_subset),
            field_decls=global_ctx.get_dcir_decls(field_accesses),
            read_accesses={key: field_accesses[key] for key in read_accesses.keys()},
            write_accesses={key: field_accesses[key] for key in write_accesses.keys()},
            symbols=symbols,
        )


class StencilComputationSDFGBuilder(NodeVisitor):
    @dataclass
    class NodeContext:
        input_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]
        output_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]

    @dataclass
    class SDFGContext:
        sdfg: dace.SDFG
        state: dace.SDFGState
        field_decls: Dict[str, dcir.FieldDecl] = dataclasses.field(default_factory=dict)
        state_stack: List[dace.SDFGState] = dataclasses.field(default_factory=list)

        def add_state(self):
            old_state = self.state
            state = self.sdfg.add_state()
            for edge in self.sdfg.out_edges(old_state):
                self.sdfg.remove_edge(edge)
                self.sdfg.add_edge(
                    state,
                    edge.dst,
                    edge.data,
                )
            self.sdfg.add_edge(old_state, state, dace.InterstateEdge())
            self.state = state
            return self

        def add_loop(self, index_range: dcir.Range):

            loop_state = self.sdfg.add_state()
            after_state = self.sdfg.add_state()
            for edge in self.sdfg.out_edges(self.state):
                self.sdfg.remove_edge(edge)
                self.sdfg.add_edge(
                    after_state,
                    edge.dst,
                    edge.data,
                )
            comparison_op = "<" if index_range.stride > 0 else ">"
            condition_expr = f"{index_range.var} {comparison_op} {index_range.end}"
            _, _, after_state = self.sdfg.add_loop(
                before_state=self.state,
                loop_state=loop_state,
                after_state=after_state,
                loop_var=index_range.var,
                initialize_expr=str(index_range.start),
                condition_expr=condition_expr,
                increment_expr=f"{index_range.var}+({index_range.stride})",
            )
            self.state_stack.append(after_state)
            self.state = loop_state
            return self

        def pop_loop(self):
            self.state = self.state_stack[-1]
            del self.state_stack[-1]

    @staticmethod
    def _interval_or_idx_range(
        node: Union[dcir.DomainInterval, dcir.IndexWithExtent]
    ) -> Tuple[str, str]:
        if isinstance(node, dcir.DomainInterval):
            return str(node.start), str(node.end)
        else:
            return f"{node.value}{node.extent[0]:+d}", f"{node.value}{node.extent[1]+1:+d}"

    @staticmethod
    def _get_memlets(
        node: dcir.ComputationNode,
        *,
        with_prefix: bool = True,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
    ):
        if with_prefix:
            prefix = "IN_"
        else:
            prefix = ""
        in_memlets = dict()
        for field, access_info in node.read_accesses.items():
            field_decl = sdfg_ctx.field_decls[field]
            in_memlets[prefix + field] = dace.Memlet.simple(
                field,
                subset_str=make_subset_str(
                    field_decl.access_info, access_info, field_decl.data_dims
                ),
                dynamic=access_info.is_dynamic,
            )

        if with_prefix:
            prefix = "OUT_"
        else:
            prefix = ""
        out_memlets = dict()
        for field, access_info in node.write_accesses.items():
            field_decl = sdfg_ctx.field_decls[field]
            out_memlets[prefix + field] = dace.Memlet.simple(
                field,
                subset_str=make_subset_str(
                    field_decl.access_info, access_info, field_decl.data_dims
                ),
                dynamic=access_info.is_dynamic,
            )

        return in_memlets, out_memlets

    @staticmethod
    def _add_edges(
        entry_node: dace.nodes.Node,
        exit_node: dace.nodes.Node,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
        in_memlets: Dict[str, dace.Memlet],
        out_memlets: Dict[str, dace.Memlet],
    ):
        for conn, memlet in in_memlets.items():
            sdfg_ctx.state.add_edge(
                *node_ctx.input_node_and_conns[memlet.data], entry_node, conn, memlet
            )
        if not in_memlets and None in node_ctx.input_node_and_conns:
            sdfg_ctx.state.add_edge(
                *node_ctx.input_node_and_conns[None], entry_node, None, dace.Memlet()
            )
        for conn, memlet in out_memlets.items():
            sdfg_ctx.state.add_edge(
                exit_node, conn, *node_ctx.output_node_and_conns[memlet.data], memlet
            )
        if not out_memlets and None in node_ctx.output_node_and_conns:
            sdfg_ctx.state.add_edge(
                exit_node, None, *node_ctx.output_node_and_conns[None], dace.Memlet()
            )

    @staticmethod
    def _add_origin(
        access_info: dcir.FieldAccessInfo,
        subset: Union[dace.subsets.Range, str],
        add_for_variable=False,
    ):
        if isinstance(subset, str):
            subset = dace.subsets.Range.from_string(subset)
        origin_strs = []
        for axis in access_info.axes():
            if axis in access_info.variable_offset_axes and not add_for_variable:
                origin_strs.append(str(0))
            elif add_for_variable:
                clamped_interval = access_info.clamp_full_axis(axis).grid_subset.intervals[axis]
                origin_strs.append(
                    f"-({get_axis_bound_str(clamped_interval.start, axis.domain_symbol())})"
                )
            else:
                interval = access_info.grid_subset.intervals[axis]
                if isinstance(interval, dcir.DomainInterval):
                    origin_strs.append(
                        f"-({get_axis_bound_str(interval.start, axis.domain_symbol())})"
                    )
                elif isinstance(interval, dcir.TileInterval):
                    origin_strs.append(
                        f"-({interval.axis.tile_symbol()}{interval.start_offset:+d})"
                    )
                else:
                    assert isinstance(interval, dcir.IndexWithExtent)
                    origin_strs.append(f"-({interval.value}{interval.extent[0]:+d})")

        sym = dace.symbolic.pystr_to_symbolic
        res_ranges = [
            (sym(f"({rng[0]})+({orig})"), sym(f"({rng[1]})+({orig})"), rng[2])
            for rng, orig in zip(subset.ranges, origin_strs)
        ]
        return dace.subsets.Range(res_ranges)

    def visit_Tasklet(
        self,
        node: dcir.Tasklet,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
    ):
        code = TaskletCodegen().visit(
            node,
            read_accesses=node.read_accesses,
            write_accesses=node.write_accesses,
            sdfg_ctx=sdfg_ctx,
        )
        access_collection = AccessCollector.apply(node)
        in_memlets = dict()
        for field, access_info in node.read_accesses.items():
            field_decl = sdfg_ctx.field_decls[field]
            for offset in access_collection.read_offsets()[field]:
                conn_name = get_tasklet_symbol(field, offset, is_target=False)
                subset_strs = []
                for axis in access_info.axes():
                    if axis in access_info.variable_offset_axes:
                        full_size = (
                            field_decl.access_info.clamp_full_axis(axis)
                            .grid_subset.intervals[axis]
                            .size
                        )
                        subset_strs.append(f"0:{full_size}")
                    else:
                        subset_strs.append(axis.iteration_symbol() + f"{offset[axis.to_idx()]:+d}")
                subset_str = ",".join(subset_strs)
                subset_str = str(
                    StencilComputationSDFGBuilder._add_origin(
                        access_info=field_decl.access_info, subset=subset_str
                    )
                )
                if sdfg_ctx.field_decls[field].data_dims:
                    subset_str += "," + ",".join(
                        f"0:{dim}" for dim in sdfg_ctx.field_decls[field].data_dims
                    )
                in_memlets[conn_name] = dace.Memlet.simple(
                    field, subset_str=subset_str, dynamic=node.read_accesses[field].is_dynamic
                )

        out_memlets = dict()
        for field, access_info in node.write_accesses.items():
            field_decl = sdfg_ctx.field_decls[field]
            conn_name = get_tasklet_symbol(field, (0, 0, 0), is_target=True)
            subset_strs = []
            for axis in access_info.axes():
                if axis in access_info.variable_offset_axes:
                    full_size = (
                        field_decl.access_info.clamp_full_axis(axis)
                        .grid_subset.intervals[axis]
                        .tile_size
                    )
                    subset_strs.append(f"0:{full_size}")
                else:
                    subset_strs.append(axis.iteration_symbol())
            subset_str = ",".join(subset_strs)
            subset_str = str(
                StencilComputationSDFGBuilder._add_origin(
                    access_info=field_decl.access_info, subset=subset_str
                )
            )
            if sdfg_ctx.field_decls[field].data_dims:
                subset_str += "," + ",".join(
                    f"0:{dim}" for dim in sdfg_ctx.field_decls[field].data_dims
                )
            out_memlets[conn_name] = dace.Memlet.simple(
                field, subset_str=subset_str, dynamic=node.write_accesses[field].is_dynamic
            )

        tasklet = dace.nodes.Tasklet(
            label=f"Tasklet_{id(node)}",
            code=code,
            inputs={inp for inp in in_memlets.keys()},
            outputs={outp for outp in out_memlets.keys()},
        )

        sdfg_ctx.state.add_node(tasklet)

        StencilComputationSDFGBuilder._add_edges(
            tasklet,
            tasklet,
            sdfg_ctx=sdfg_ctx,
            in_memlets=in_memlets,
            out_memlets=out_memlets,
            node_ctx=node_ctx,
        )

    def visit_Range(self, node: dcir.Range, **kwargs):
        return node.to_ndrange()

    def visit_DomainMap(
        self,
        node: dcir.DomainMap,
        *,
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
    ):
        map_entry, map_exit = sdfg_ctx.state.add_map(
            name=node.index_range.var + "_map",
            ndrange=self.visit(node.index_range),
        )

        input_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = dict()
        output_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = dict()
        for scope_node in node.computations:
            for field in node.read_accesses.keys():
                map_entry.add_in_connector("IN_" + field)
                map_entry.add_out_connector("OUT_" + field)
                input_node_and_conns[field] = (map_entry, "OUT_" + field)
            for field in node.write_accesses.keys():
                map_exit.add_in_connector("IN_" + field)
                map_exit.add_out_connector("OUT_" + field)
                output_node_and_conns[field] = (map_exit, "IN_" + field)
            if not input_node_and_conns:
                input_node_and_conns[None] = (map_entry, None)
            if not output_node_and_conns:
                output_node_and_conns[None] = (map_exit, None)
            inner_node_ctx = StencilComputationSDFGBuilder.NodeContext(
                input_node_and_conns=input_node_and_conns,
                output_node_and_conns=output_node_and_conns,
            )

            self.visit(scope_node, sdfg_ctx=sdfg_ctx, node_ctx=inner_node_ctx)

        in_memlets, out_memlets = self._get_memlets(node, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)

        self._add_edges(
            map_entry,
            map_exit,
            node_ctx=node_ctx,
            sdfg_ctx=sdfg_ctx,
            in_memlets=in_memlets,
            out_memlets=out_memlets,
        )

    def visit_DomainLoop(
        self,
        node: dcir.DomainLoop,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ):
        sdfg_ctx = sdfg_ctx.add_loop(node.index_range)
        self.visit(node.loop_states, sdfg_ctx=sdfg_ctx, **kwargs)
        sdfg_ctx.pop_loop()

    def visit_ComputationState(
        self,
        node: dcir.ComputationState,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ):

        sdfg_ctx = sdfg_ctx.add_state()
        read_acc_and_conn: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = dict()
        write_acc_and_conn: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = dict()
        for computation in node.computations:
            assert isinstance(computation, dcir.ComputationNode)
            for field in computation.read_accesses.keys():
                if field not in read_acc_and_conn:
                    read_acc_and_conn[field] = (sdfg_ctx.state.add_access(field), None)
            for field in computation.write_accesses.keys():
                if field not in write_acc_and_conn:
                    write_acc_and_conn[field] = (sdfg_ctx.state.add_access(field), None)
            node_ctx = StencilComputationSDFGBuilder.NodeContext(
                input_node_and_conns=read_acc_and_conn,
                output_node_and_conns=write_acc_and_conn,
            )
            self.visit(computation, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx, **kwargs)

    def visit_StateMachine(
        self,
        node: dcir.StateMachine,
        *,
        sdfg_ctx: Optional["StencilComputationSDFGBuilder.SDFGContext"] = None,
        node_ctx: Optional["StencilComputationSDFGBuilder.NodeContext"] = None,
    ):

        sdfg = dace.SDFG(f"StateMachine_{id(node)}")
        state = sdfg.add_state()
        symbol_mapping = {}
        for axis in [dcir.Axis.I, dcir.Axis.J, dcir.Axis.K]:
            sdfg.add_symbol(axis.domain_symbol(), stype=dace.int32)
            symbol_mapping[axis.domain_symbol()] = axis.domain_symbol()
        if sdfg_ctx is not None and node_ctx is not None:
            nsdfg = sdfg_ctx.state.add_nested_sdfg(
                sdfg=sdfg,
                parent=None,
                inputs=set(node.read_accesses.keys()),
                outputs=set(node.write_accesses.keys()),
                symbol_mapping=symbol_mapping,
            )
            in_memlets, out_memlets = StencilComputationSDFGBuilder._get_memlets(
                node, with_prefix=False, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
            )
            StencilComputationSDFGBuilder._add_edges(
                nsdfg,
                nsdfg,
                sdfg_ctx=sdfg_ctx,
                node_ctx=node_ctx,
                in_memlets=in_memlets,
                out_memlets=out_memlets,
            )
        else:
            nsdfg = dace.nodes.NestedSDFG(
                label=sdfg.label,
                sdfg=sdfg,
                inputs=set(node.read_accesses.keys()),
                outputs=set(node.write_accesses.keys()),
                symbol_mapping=symbol_mapping,
            )

        inner_sdfg_ctx = StencilComputationSDFGBuilder.SDFGContext(
            sdfg=sdfg,
            state=state,
            field_decls=node.field_decls,
        )

        for name, decl in node.field_decls.items():
            inner_sdfg_ctx.sdfg.add_array(
                name,
                shape=decl.shape,
                strides=[dace.symbolic.pystr_to_symbolic(s) for s in decl.strides],
                dtype=np.dtype(common.data_type_to_typestr(decl.dtype)).type,
            )
        for symbol, dtype in node.symbols.items():
            if symbol not in inner_sdfg_ctx.sdfg.symbols:
                inner_sdfg_ctx.sdfg.add_symbol(
                    symbol, stype=np.dtype(common.data_type_to_typestr(dtype)).type
                )
            nsdfg.symbol_mapping[symbol] = symbol

        for computation_state in node.states:
            self.visit(computation_state, sdfg_ctx=inner_sdfg_ctx)
        for sym in nsdfg.sdfg.free_symbols:
            if sym not in nsdfg.sdfg.symbols:
                nsdfg.sdfg.add_symbol(sym, stype=dace.int32)
            nsdfg.symbol_mapping.setdefault(str(sym), str(sym))

        return nsdfg


@dace.library.register_expansion(StencilComputation, "default")
class StencilComputationExpansion(dace.library.ExpandTransformation):
    environments: List = []

    @staticmethod
    def _solve_for_domain(field_decls: Dict[str, dcir.FieldDecl], outer_subsets):
        equations = []
        symbols = set()

        # Collect equations and symbols from arguments and shapes
        for field, decl in field_decls.items():
            inner_shape = [dace.symbolic.pystr_to_symbolic(s) for s in decl.shape]
            outer_shape = [
                dace.symbolic.pystr_to_symbolic(s) for s in outer_subsets[field].bounding_box_size()
            ]

            for inner_dim, outer_dim in zip(inner_shape, outer_shape):
                repldict = {}
                for sym in dace.symbolic.symlist(inner_dim).values():
                    newsym = dace.symbolic.symbol("__SOLVE_" + str(sym))
                    symbols.add(newsym)
                    repldict[sym] = newsym

                # Replace symbols with __SOLVE_ symbols so as to allow
                # the same symbol in the called SDFG
                if repldict:
                    inner_dim = inner_dim.subs(repldict)

                equations.append(inner_dim - outer_dim)
        if len(symbols) == 0:
            return {}

        # Solve for all at once
        results = sympy.solve(equations, *symbols, dict=True)
        result = results[0]
        return {str(k)[len("__SOLVE_") :]: str(v) for k, v in result.items()}

    @staticmethod
    def expansion(
        node: "StencilComputation", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.nodes.NestedSDFG:

        start, end = (
            node.oir_node.sections[0].interval.start,
            node.oir_node.sections[0].interval.end,
        )
        for section in node.oir_node.sections:
            start = min(start, section.interval.start)
            end = max(end, section.interval.end)

        overall_interval = dcir.DomainInterval(
            start=dcir.AxisBound(axis=dcir.Axis.K, level=start.level, offset=start.offset),
            end=dcir.AxisBound(axis=dcir.Axis.K, level=end.level, offset=end.offset),
        )
        overall_extent = Extent.zeros(ndims=3)
        for he in node.oir_node.iter_tree().if_isinstance(oir.HorizontalExecution):
            overall_extent = overall_extent.union(node.extents[id(he)])
        iteration_ctx = DaCeIRBuilder.IterationContext(
            grid_subset=dcir.GridSubset.from_gt4py_extent(overall_extent).set_interval(
                axis=dcir.Axis.K, interval=overall_interval
            )
        )
        daceir_builder_global_ctx = DaCeIRBuilder.GlobalContext(
            library_node=node, block_extents=node.extents, arrays=parent_sdfg.arrays
        )

        daceir: dcir.StateMachine = DaCeIRBuilder().visit(
            node.oir_node,
            global_ctx=daceir_builder_global_ctx,
            iteration_ctx=iteration_ctx,
            expansion_order=list(node.expansion_order),
        )

        #
        nsdfg = StencilComputationSDFGBuilder().visit(daceir)

        for in_edge in parent_state.in_edges(node):
            assert in_edge.dst_conn.startswith("IN_")
            in_edge.dst_conn = in_edge.dst_conn[len("IN_") :]
        for out_edge in parent_state.out_edges(node):
            assert out_edge.src_conn.startswith("OUT_")
            out_edge.src_conn = out_edge.src_conn[len("OUT_") :]

        subsets = dict()
        for edge in parent_state.in_edges(node):
            subsets[edge.dst_conn] = edge.data.subset
        for edge in parent_state.out_edges(node):
            subsets[edge.src_conn] = dace.subsets.union(
                edge.data.subset, subsets.get(edge.src_conn, edge.data.subset)
            )
        for edge in parent_state.in_edges(node):
            edge.data.subset = subsets[edge.dst_conn]
        for edge in parent_state.out_edges(node):
            edge.data.subset = subsets[edge.src_conn]

        nsdfg.symbol_mapping.update(
            **StencilComputationExpansion._solve_for_domain(daceir.field_decls, subsets)
        )
        for sym in nsdfg.free_symbols:
            if str(sym) not in parent_sdfg.symbols:
                parent_sdfg.add_symbol(str(sym), stype=dace.int32)
        return nsdfg
