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

import warnings
from typing import Any

from eve import NodeTranslator
from gtc import common, oir


class AdjacentLoopMerging(NodeTranslator):
    @staticmethod
    def _mergeable(a: oir.VerticalLoop, b: oir.VerticalLoop) -> bool:
        if a.loop_order != b.loop_order:
            return False
        a_interval = a.sections[-1].interval
        b_interval = b.sections[0].interval
        if a.loop_order == common.LoopOrder.BACKWARD:
            a_lim = a_interval.start
            b_lim = b_interval.end
        else:
            a_lim = a_interval.end
            b_lim = b_interval.start
        return a_lim.level == b_lim.level and a_lim.offset == b_lim.offset

    @staticmethod
    def _merge(a: oir.VerticalLoop, b: oir.VerticalLoop) -> oir.VerticalLoop:
        sections = a.sections + b.sections
        if a.caches or b.caches:
            warnings.warn("AdjacentLoopMerging pass removed previously declared caches")
        return oir.VerticalLoop(
            loop_order=a.loop_order,
            sections=sections,
            caches=[],
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        if not node.vertical_loops:
            return self.generic_visit(node, **kwargs)
        vertical_loops = [self.visit(node.vertical_loops[0], **kwargs)]
        for vertical_loop in node.vertical_loops[1:]:
            vertical_loop = self.visit(vertical_loop, **kwargs)
            mergeable = self._mergeable(vertical_loops[-1], vertical_loop)
            if mergeable:
                vertical_loops[-1] = self._merge(vertical_loops[-1], vertical_loop)
            else:
                vertical_loops.append(vertical_loop)

        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=vertical_loops,
            declarations=node.declarations,
        )


class IdenticalSectionMerging(NodeTranslator):
    @staticmethod
    def _mergeable(
        order: common.LoopOrder, a: oir.VerticalLoopSection, b: oir.VerticalLoopSection
    ) -> bool:
        if order == common.LoopOrder.BACKWARD:
            a_lim = a.interval.start
            b_lim = b.interval.end
        else:
            a_lim = a.interval.end
            b_lim = b.interval.start
        return a_lim.level == b_lim.level and a_lim.offset == b_lim.offset

    @staticmethod
    def _contain_same_executions(a: oir.VerticalLoopSection, b: oir.VerticalLoopSection) -> bool:
        return a.horizontal_executions == b.horizontal_executions

    @staticmethod
    def _merge(
        order: common.LoopOrder, a: oir.VerticalLoopSection, b: oir.VerticalLoopSection
    ) -> oir.VerticalLoopSection:
        new_interval = a.interval
        if order == common.LoopOrder.BACKWARD:
            new_interval.start = b.interval.start
        else:
            new_interval.end = b.interval.end

        return oir.VerticalLoopSection(
            interval=new_interval,
            horizontal_executions=a.horizontal_executions,
        )

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        if len(node.sections) <= 1:
            return self.generic_visit(node, **kwargs)

        new_sections = []
        new_sections = [self.visit(node.sections[0], **kwargs)]
        for section in node.sections[1:]:
            section = self.visit(section, **kwargs)
            syntactic_mergeable = self._mergeable(node.loop_order, new_sections[-1], section)
            semantic_mergeable = self._contain_same_executions(new_sections[-1], section)
            if syntactic_mergeable and semantic_mergeable:
                new_sections[-1] = self._merge(node.loop_order, new_sections[-1], section)
            else:
                new_sections.append(section)

        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=new_sections,
            caches=node.caches,
            loc=node.loc,
        )
