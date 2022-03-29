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
from typing import Any, Dict, Optional

import eve
from gt4py.definitions import Extent
from gtc import oir
from gtc.dace.utils import compute_horizontal_block_extents
from gtc.passes import utils


class RemoveUnexecutedRegions(eve.NodeTranslator):
    @dataclass
    class Context:
        block_extents: Dict[int, Extent] = field(default_factory=dict)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        block_extents = compute_horizontal_block_extents(node)
        ctx = self.Context(block_extents=block_extents)
        vertical_loops = [self.visit(loop, ctx=ctx) for loop in node.vertical_loops]
        vertical_loops = [loop for loop in vertical_loops if isinstance(loop, oir.VerticalLoop)]
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=vertical_loops,
            declarations=node.declarations,
        )

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        sections = [self.visit(section, **kwargs) for section in node.sections]
        sections = [section for section in sections if isinstance(section, oir.VerticalLoopSection)]
        if sections:
            # Clear caches
            return oir.VerticalLoop(loop_order=node.loop_order, sections=sections, caches=[])
        else:
            return eve.NOTHING

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoopSection,
        **kwargs: Any,
    ) -> Optional[oir.VerticalLoopSection]:
        executions = [self.visit(execution, **kwargs) for execution in node.horizontal_executions]
        if res_executions := [
            execution for execution in executions if isinstance(execution, oir.HorizontalExecution)
        ]:
            return oir.VerticalLoopSection(
                interval=node.interval, horizontal_executions=res_executions
            )
        else:
            return eve.NOTHING

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        ctx: Context,
        **kwargs: Any,
    ) -> Optional[oir.HorizontalExecution]:
        compute_extent = ctx.block_extents[id(node)]

        if filtered_body := self.visit(node.body, ctx=ctx, compute_extent=compute_extent, **kwargs):
            return oir.HorizontalExecution(
                body=filtered_body, declarations=node.declarations, **kwargs
            )
        else:
            return eve.NOTHING

    def visit_MaskStmt(
        self,
        node: oir.MaskStmt,
        *,
        compute_extent: Extent,
        **kwargs: Any,
    ) -> oir.MaskStmt:
        if masks := node.mask.iter_tree().if_isinstance(oir.HorizontalMask).to_list():
            assert len(masks) == 1
            mask = masks[0]
            dist_from_edge = utils.compute_extent_difference(compute_extent, mask)
            if dist_from_edge is None:
                return eve.NOTHING
        else:
            dist_from_edge = Extent.zeros()

        body = self.visit(
            node.body,
            compute_extent=(compute_extent - dist_from_edge),
            **kwargs,
        )

        return oir.MaskStmt(mask=node.mask, body=body)
