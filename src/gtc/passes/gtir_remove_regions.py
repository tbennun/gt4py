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
from typing import Any, Dict

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
        rev_vertical_loops = [self.visit(loop, ctx=ctx) for loop in reversed(node.vertical_loops)]
        vertical_loops = [
            vloop for vloop in reversed(rev_vertical_loops) if isinstance(vloop, gtir.VerticalLoop)
        ]
        params = [
            decl
            for decl in node.params
            if decl.name
            in iter_tree(vertical_loops)
            .if_isinstance(gtir.FieldAccess, gtir.ScalarAccess)
            .getattr("name")
            .to_set()
        ]
        return gtir.Stencil(
            name=node.name,
            params=params,
            vertical_loops=vertical_loops,
        )

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs: Any) -> gtir.VerticalLoop:
        stmts = [self.visit(stmt, **kwargs) for stmt in node.body]
        stmts = [stmt for stmt in stmts if stmt is not eve.NOTHING]
        if stmts:
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
            ext_diff = utils.compute_extent_difference(ctx.stmt_extents[id(stmt)], node.mask)
            if ext_diff is not None:
                res_stmts.append(stmt)
        if not res_stmts:
            return eve.NOTHING
        else:
            return gtir.HorizontalRegion(mask=node.mask, block=gtir.BlockStmt(body=res_stmts))


def remove_regions(node):
    return RemoveUnexecutedRegions().visit(node)
