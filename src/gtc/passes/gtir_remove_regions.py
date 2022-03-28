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

"""Removes horizontal executions that is never executed and computes the correct extents."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

import eve
from eve.iterators import iter_tree
from gt4py.definitions import Extent
from gtc import gtir
from gtc.passes import utils
from gtc.passes.gtir_legacy_extents import LegacyExtentsVisitor


class RemoveUnexecutedRegions(eve.NodeTranslator):
    @dataclass
    class Context:
        stmt_extents: Dict[str, Extent] = field(default_factory=dict)

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> gtir.Stencil:
        _, stmt_extents = LegacyExtentsVisitor().visit(node)
        ctx = self.Context(stmt_extents=stmt_extents)
        rev_vertical_loops = [
            self.visit(loop, ctx=ctx, params=node.params) for loop in reversed(node.vertical_loops)
        ]
        vertical_loops = [
            vloop for vloop in reversed(rev_vertical_loops) if isinstance(vloop, gtir.VerticalLoop)
        ]
        accessed_fields = (
            iter_tree(vertical_loops).if_isinstance(gtir.FieldAccess).getattr("name").to_set()
        )
        accessed_scalars = (
            iter_tree(vertical_loops).if_isinstance(gtir.ScalarAccess).getattr("name").to_set()
        )
        params = [
            decl
            for decl in node.params
            if (isinstance(decl, gtir.FieldDecl) and decl.name in accessed_fields)
            or (isinstance(decl, gtir.ScalarDecl) and decl.name in accessed_scalars)
        ]
        return gtir.Stencil(
            name=node.name,
            params=params,
            vertical_loops=vertical_loops,
        )

    def visit_VerticalLoop(
        self, node: gtir.VerticalLoop, *, params: List[gtir.FieldDecl], **kwargs: Any
    ) -> gtir.VerticalLoop:
        stmts = [self.visit(stmt, **kwargs) for stmt in node.body]
        stmts = [stmt for stmt in stmts if stmt is not eve.NOTHING]
        param_names = [param.name for param in params]
        if stmts and any(
            assign.name in param_names
            for assign in iter_tree(stmts).if_isinstance(gtir.ParAssignStmt).getattr("left")
        ):
            temporaries = [
                decl
                for decl in node.temporaries
                if decl.name
                in iter_tree(stmts).if_isinstance(gtir.FieldAccess).getattr("name").to_set()
            ]
            return gtir.VerticalLoop(
                interval=node.interval,
                loop_order=node.loop_order,
                temporaries=temporaries,
                body=stmts,
            )
        else:
            return eve.NOTHING

    def visit_HorizontalRegion(self, node: gtir.HorizontalRegion, *, ctx):
        res_stmts = []
        for stmt in node.block.body:
            new_stmt = self.visit(stmt, ctx=ctx)
            ext_diff = utils.compute_extent_difference(ctx.stmt_extents[id(stmt)], node.mask)
            if ext_diff is not None:
                res_stmts.append(new_stmt)
        if not res_stmts:
            return eve.NOTHING
        else:
            res_extent = None
            for stmt in node.block.body:
                if res_extent is None:
                    res_extent = ctx.stmt_extents[id(stmt)]
                else:
                    res_extent |= ctx.stmt_extents[id(stmt)]
            ctx.stmt_extents[id(node)] = res_extent
            return gtir.HorizontalRegion(mask=node.mask, block=gtir.BlockStmt(body=res_stmts))

    def visit_While(self, node: gtir.While, *, ctx):
        res = None
        stmts = node.body
        new_stmts = self.generic_visit(node.body, ctx=ctx)

        for stmt in stmts:
            extent = ctx.stmt_extents[id(stmt)]
            if res is None:
                res = extent
            else:
                res |= extent
        ctx.stmt_extents[id(node)] = res
        return gtir.While(cond=node.cond, body=new_stmts)

    def _visit_IfStmt(self, node, *, ctx):
        res = None
        stmts = node.true_branch.body
        new_true_branch = self.generic_visit(node.true_branch.body, ctx=ctx)
        if node.false_branch is not None:
            stmts += node.false_branch.body
            new_false_branch = self.generic_visit(node.false_branch.body, ctx=ctx)
        else:
            new_false_branch = None
        for stmt in stmts:
            self.visit(stmt, ctx=ctx)
            extent = ctx.stmt_extents[id(stmt)]
            if res is None:
                res = extent
            else:
                res |= extent
        ctx.stmt_extents[id(node)] = res
        return (
            gtir.BlockStmt(body=new_true_branch),
            gtir.BlockStmt(body=new_false_branch) if new_false_branch is not None else None,
        )

    def visit_FieldIfStmt(self, node: gtir.FieldIfStmt, **kwargs):
        new_true_branch, new_false_branch = self._visit_IfStmt(node, **kwargs)
        return gtir.FieldIfStmt(
            cond=node.cond, true_branch=new_true_branch, false_branch=new_false_branch
        )

    def visit_ScalarIfStmt(self, node: gtir.ScalarIfStmt, **kwargs):
        new_true_branch, new_false_branch = self._visit_IfStmt(node, **kwargs)
        return gtir.ScalarIfStmt(
            cond=node.cond, true_branch=new_true_branch, false_branch=new_false_branch
        )


def remove_regions(node):
    if len(iter_tree(node).if_isinstance(gtir.HorizontalRegion).to_list()) == 0:
        # TODO deal with complexity. To unlock satadjust, just return if no regions.
        return node
    return RemoveUnexecutedRegions().visit(node)
