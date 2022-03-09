from typing import Any, Dict, List, Tuple

import dace

import eve
from gtc import daceir as dcir

from .utils import union_node_access_infos


def set_grid_subset_idx(axis, grid_subset, idx):
    if isinstance(grid_subset.intervals[axis], dcir.IndexWithExtent):
        extent = grid_subset.intervals[axis].extent
    else:
        extent = [0, 0]
    return grid_subset.set_interval(
        axis,
        dcir.IndexWithExtent(
            axis=axis,
            value=idx,
            extent=extent,
        ),
    )


def set_idx(axis, access_infos, idx):
    res = dict()
    for name, info in access_infos.items():
        grid_subset = set_grid_subset_idx(axis, info.grid_subset, idx)
        res[name] = dcir.FieldAccessInfo(
            grid_subset=grid_subset,
            global_grid_subset=info.global_grid_subset,
            dynamic_access=info.dynamic_access,
            variable_offset_axes=info.variable_offset_axes,  # should actually never be True here
        )
    return res


class FieldAccessRenamer(eve.NodeMutator):
    def apply(self, node, *, local_name_map):
        return self.visit(node, local_name_map=local_name_map)

    def _rename_accesses(self, access_infos, *, local_name_map):
        return {
            local_name_map[name] if name in local_name_map else name: info
            for name, info in access_infos.items()
        }

    def visit_CopyState(self, node: dcir.CopyState, *, local_name_map):
        name_map = {local_name_map[k]: local_name_map[v] for k, v in node.name_map.items()}
        return dcir.CopyState(
            read_accesses=self._rename_accesses(node.read_accesses, local_name_map=local_name_map),
            write_accesses=self._rename_accesses(
                node.write_accesses, local_name_map=local_name_map
            ),
            name_map=name_map,
        )

    def visit_DomainMap(self, node: dcir.DomainMap, *, local_name_map):
        computations = self.visit(node.computations, local_name_map=local_name_map)
        return dcir.DomainMap(
            index_ranges=node.index_ranges,
            computations=computations,
            schedule=node.schedule,
            grid_subset=node.grid_subset,
            read_accesses=self._rename_accesses(node.read_accesses, local_name_map=local_name_map),
            write_accesses=self._rename_accesses(
                node.write_accesses, local_name_map=local_name_map
            ),
        )

    def visit_DomainLoop(self, node: dcir.DomainLoop, *, local_name_map):
        return dcir.DomainLoop(
            axis=node.axis,
            index_range=node.index_range,
            loop_states=self.visit(node.loop_states, local_name_map=local_name_map),
            grid_subset=node.grid_subset,
            read_accesses=self._rename_accesses(node.read_accesses, local_name_map=local_name_map),
            write_accesses=self._rename_accesses(
                node.write_accesses, local_name_map=local_name_map
            ),
        )

    def visit_StateMachine(self, node: dcir.StateMachine, *, local_name_map: Dict[str, str]):
        name_map = dict(node.name_map)
        for old_array_name in node.name_map.keys():
            if old_array_name in local_name_map:
                new_array_name = local_name_map[old_array_name]
                name_map[new_array_name] = name_map[old_array_name]
                del name_map[old_array_name]
        return dcir.StateMachine(
            label=node.label,
            field_decls=node.field_decls,  # don't rename, this is inside
            read_accesses=self._rename_accesses(node.read_accesses, local_name_map=local_name_map),
            write_accesses=self._rename_accesses(
                node.write_accesses, local_name_map=local_name_map
            ),
            symbols=node.symbols,
            states=node.states,
            name_map=name_map,
        )

    def visit_Tasklet(self, node: dcir.Tasklet, *, local_name_map):
        name_map = dict(node.name_map)
        for old_array_name in node.name_map.keys():
            if old_array_name in local_name_map:
                new_array_name = local_name_map[old_array_name]
                name_map[new_array_name] = name_map[old_array_name]
                del name_map[old_array_name]
        return dcir.Tasklet(
            read_accesses=self._rename_accesses(node.read_accesses, local_name_map=local_name_map),
            write_accesses=self._rename_accesses(
                node.write_accesses, local_name_map=local_name_map
            ),
            name_map=name_map,
            stmts=node.stmts,
        )


rename_field_accesses = FieldAccessRenamer().apply


class FieldDeclPropagator(eve.NodeMutator):
    def apply(self, node, *, decl_map):
        return self.visit(node, decl_map=decl_map)

    def visit_StateMachine(
        self, node: dcir.StateMachine, *, decl_map: Dict[str, Tuple[Any, dace.StorageType]]
    ):
        field_decls = dict(node.field_decls)
        for name, (strides, storage) in decl_map.items():
            if name in node.name_map:
                orig_field_decl = field_decls[node.name_map[name]]
                field_decls[node.name_map[name]] = dcir.FieldDecl(
                    name=orig_field_decl.name,
                    dtype=orig_field_decl.dtype,
                    strides=strides,
                    data_dims=orig_field_decl.data_dims,
                    access_info=orig_field_decl.access_info,
                    storage=storage,
                )
        return dcir.StateMachine(
            label=node.label,
            field_decls=field_decls,  # don't rename, this is inside
            read_accesses=node.read_accesses,
            write_accesses=node.write_accesses,
            symbols=node.symbols,
            states=node.states,
            name_map=node.name_map,
        )


propagate_field_decls = FieldDeclPropagator().apply


class MakeLocalCaches(eve.NodeTranslator):
    def _make_cache_init_state(self, loops, *, read_accesses, local_name_map):
        from .utils import union_node_access_infos, union_node_grid_subsets

        read_accesses, write_accesses, field_accesses = union_node_access_infos(
            loops[0].loop_states
        )

        axis = loops[0].axis
        cache_field_reads = {
            name: read_accesses[name] for name in local_name_map.keys() if name in read_accesses
        }
        if len(set(loop.index_range.stride for loop in loops)) == 1:
            offset = loops[0].index_range.stride
        else:
            offset = 0
        cache_populate_accesses = dict()
        for name, info in cache_field_reads.items():
            interval = info.grid_subset.intervals[axis]
            # if interval.extent[1] - interval.extent[0] <= 1:
            #     continue
            # if offset < 0:
            #     extent = interval.extent[0] - offset, interval.extent[1]
            # else:
            #     extent = interval.extent[0], interval.extent[1] - offset
            extent = interval.extent
            cache_populate_accesses[name] = dcir.FieldAccessInfo(
                grid_subset=info.grid_subset.set_interval(
                    axis, dcir.IndexWithExtent(axis=axis, value=interval.value, extent=extent)
                ),
                global_grid_subset=info.global_grid_subset,
                dynamic_access=info.dynamic_access,
                variable_offset_axes=info.variable_offset_axes,
            )

        start_grid_subset = set_grid_subset_idx(
            axis, loops[0].grid_subset, loops[0].index_range.start
        )

        def set_local_names(access_infos):
            return {local_name_map[k]: v for k, v in access_infos.items()}

        def set_start_idx(access_infos):
            return set_idx(axis, access_infos, loops[0].index_range.start)

        return dcir.CopyState(
            read_accesses=set_start_idx(cache_populate_accesses),
            write_accesses=set_local_names(cache_populate_accesses),
            name_map=local_name_map,
            grid_subset=start_grid_subset,
        )

    def _make_localcache_states(
        self,
        loop,
        loop_states,
        *,
        context_read_accesses,
        context_write_accesses,
        context_field_accesses,
        local_name_map,
    ):
        from .utils import union_node_access_infos, union_node_grid_subsets

        axis = loop.axis
        grid_subset = union_node_grid_subsets(list(loop_states))
        read_accesses, write_accesses, field_accesses = union_node_access_infos(list(loop_states))
        cache_field_reads = {
            name: read_accesses[name] for name in local_name_map.keys() if name in read_accesses
        }
        cache_field_writes = {
            name: write_accesses[name] for name in local_name_map.keys() if name in write_accesses
        }
        cache_populate_accesses = dict()
        for name, info in cache_field_reads.items():
            interval = info.grid_subset.intervals[axis]
            # # if loop.index_range.stride < 0:
            # #     extent = interval.extent[0] - loop.index_range.stride, interval.extent[1]
            # # else:
            # extent = interval.extent[0] - loop.index_range.stride, interval.extent[1] - loop.index_range.stride
            extent = interval.extent
            cache_populate_accesses[name] = dcir.FieldAccessInfo(
                grid_subset=info.grid_subset.set_interval(
                    axis, dcir.IndexWithExtent(axis=axis, value=interval.value, extent=extent)
                ),
                global_grid_subset=info.global_grid_subset,
                dynamic_access=info.dynamic_access,
                variable_offset_axes=info.variable_offset_axes,
            )
        cache_field_reads = {
            name: read_accesses[name] for name in local_name_map.keys() if name in read_accesses
        }
        next_value_accesses = dict()
        for name, info in cache_field_reads.items():
            interval = info.grid_subset.intervals[axis]
            if loop.index_range.stride < 0:
                extent = interval.extent[0], interval.extent[0]
            else:
                extent = interval.extent[1], interval.extent[1]
            next_value_accesses[name] = dcir.FieldAccessInfo(
                grid_subset=info.grid_subset.set_interval(
                    axis, dcir.IndexWithExtent(axis=axis, value=interval.value, extent=extent)
                ),
                global_grid_subset=info.global_grid_subset,
                dynamic_access=info.dynamic_access,
                variable_offset_axes=info.variable_offset_axes,
            )
        written_value_accesses = dict()
        for name, info in cache_field_writes.items():
            if name not in write_accesses:
                continue
            interval = info.grid_subset.intervals[axis]
            extent = [0, 0]
            written_value_accesses[name] = dcir.FieldAccessInfo(
                grid_subset=info.grid_subset.set_interval(
                    axis, dcir.IndexWithExtent(axis=axis, value=interval.value, extent=extent)
                ),
                global_grid_subset=info.global_grid_subset,
                dynamic_access=info.dynamic_access,
                variable_offset_axes=info.variable_offset_axes,
            )

        def set_local_names(access_infos):
            return {local_name_map[k]: v for k, v in access_infos.items()}

        def shift(access_infos, offset):
            res = dict()
            for name, info in access_infos.items():
                grid_subset = info.grid_subset.set_interval(
                    axis, info.grid_subset.intervals[axis].shifted(offset)
                )
                res[name] = dcir.FieldAccessInfo(
                    grid_subset=grid_subset,
                    global_grid_subset=info.global_grid_subset,
                    dynamic_access=info.dynamic_access,
                    variable_offset_axes=info.variable_offset_axes,  # should actually never be True here
                )
            return res

        fill_state = dcir.CopyState(
            grid_subset=grid_subset,
            read_accesses=next_value_accesses,
            write_accesses=set_local_names(next_value_accesses),
            name_map=local_name_map,
        )
        loop_states = rename_field_accesses(loop_states, local_name_map=local_name_map)
        flush_state = dcir.CopyState(
            grid_subset=grid_subset,
            read_accesses=set_local_names(written_value_accesses),
            write_accesses=written_value_accesses,
            name_map={v: k for k, v in local_name_map.items()},
        )
        shift_state = dcir.CopyState(
            read_accesses=set_local_names(cache_populate_accesses),
            write_accesses=shift(
                set_local_names(cache_populate_accesses), -loop.index_range.stride
            ),
            name_map={v: v for v in local_name_map.values()},
            grid_subset=grid_subset,
        )
        return fill_state, *loop_states, flush_state, shift_state

    def _process_state_sequence(self, states, *, localcache_infos):

        loops_and_states = []
        for loop in states:
            if isinstance(loop, dcir.DomainLoop):
                loops_and_states.append((loop, loop.loop_states))

        if not loops_and_states:
            return states, {}

        loops = [ls[0] for ls in loops_and_states]
        loop_states = [ls[1] for ls in loops_and_states]

        read_accesses, write_accesses, field_accesses = union_node_access_infos(
            [s for ls in loop_states for s in ls]
        )

        assert len(set(loop.axis for loop in loops)) == 1
        axis = loops[0].axis

        cache_fields = set()
        for name in read_accesses.keys():
            if axis not in field_accesses[name].grid_subset.intervals:
                continue
            interval = field_accesses[name].grid_subset.intervals[axis]
            if isinstance(interval, dcir.IndexWithExtent) and interval.size > 1:
                cache_fields.add(name)

        if not cache_fields:
            return states, {}

        cache_fields = set()
        for name in read_accesses.keys():
            interval = field_accesses[name].grid_subset.intervals[axis]
            if (
                name in localcache_infos.fields
                and isinstance(interval, dcir.IndexWithExtent)
                and interval.size > 1
            ):
                cache_fields.add(name)
        local_name_map = {k: f"__local_{k}" for k in cache_fields}

        res_states = []
        last_loop_node = None
        for loop_node, loop_state_nodes in loops_and_states:
            if not res_states or last_loop_node.index_range.end != loop_node.index_range.start:
                res_states.append(
                    self._make_cache_init_state(
                        [loop_node], read_accesses=read_accesses, local_name_map=local_name_map
                    )
                )
            (fill_state, *loop_states, flush_state, shift_state,) = self._make_localcache_states(
                loop_node,
                loop_state_nodes,
                context_read_accesses=read_accesses,
                context_write_accesses=write_accesses,
                context_field_accesses=field_accesses,
                local_name_map=local_name_map,
            )

            res_states.append(
                dcir.DomainLoop(
                    loop_states=[fill_state, *loop_states, flush_state, shift_state],
                    axis=loop_node.axis,
                    index_range=loop_node.index_range,
                    grid_subset=loop_node.grid_subset,
                    read_accesses=loop_node.read_accesses,
                    write_accesses=loop_node.write_accesses,
                )
            )
            last_loop_node = loop_node
        return res_states, local_name_map

    def visit_DomainLoop(self, node: dcir.DomainLoop, *, ctx_name_map, localcache_infos):
        loop_states = node.loop_states
        # first, recurse
        if any(isinstance(n, dcir.DomainLoop) for n in loop_states):
            start_idx = [i for i, n in enumerate(loop_states) if isinstance(n, dcir.DomainLoop)][0]
            end_idx = [
                i
                for i, n in enumerate(loop_states)
                if i > start_idx and not isinstance(n, dcir.DomainLoop)
            ]
            end_idx = end_idx[0] if end_idx else None
            loop_nodes = [n for n in loop_states if isinstance(n, dcir.DomainLoop)]
            if loop_nodes[0].axis in localcache_infos:
                domain_loop_nodes, local_name_map = self._process_state_sequence(
                    loop_nodes, localcache_infos=localcache_infos[loop_nodes[0].axis]
                )
                res_loop_states = [*loop_states[:start_idx], *domain_loop_nodes]
                if end_idx:
                    res_loop_states += loop_states[end_idx:]
                loop_states = res_loop_states
                ctx_name_map.update(local_name_map)

        inner_name_map = dict()
        loop_states = self.generic_visit(
            loop_states, localcache_infos=localcache_infos, ctx_name_map=inner_name_map
        )
        ctx_name_map.update(inner_name_map)

        from .utils import union_node_access_infos

        read_accesses, write_accesses, _ = union_node_access_infos(loop_states)
        return dcir.DomainLoop(
            grid_subset=node.grid_subset,
            read_accesses=read_accesses,
            write_accesses=write_accesses,
            axis=node.axis,
            index_range=node.index_range,
            loop_states=loop_states,
        )

    def visit_StateMachine(self, node: dcir.StateMachine, *, localcache_infos, **kwargs):
        states = node.states
        local_name_map = dict()
        # first, recurse
        is_add_caches = (
            all(isinstance(n, dcir.DomainLoop) for n in states)
            and states[0].axis in localcache_infos
        )

        if is_add_caches:
            axis = states[0].axis
            states, local_name_map = self._process_state_sequence(
                states, localcache_infos=localcache_infos[states[0].axis]
            )
        from .utils import union_node_access_infos

        inner_name_map = dict()
        states = self.generic_visit(
            states, localcache_infos=localcache_infos, ctx_name_map=inner_name_map
        )
        local_name_map.update(inner_name_map)

        _, _, inner_field_accesses = union_node_access_infos(
            [s for loop in states if isinstance(loop, dcir.DomainLoop) for s in loop.loop_states]
        )
        field_accesses = {
            k: v if k.startswith("__local_") else node.field_decls[k].access_info
            for k, v in inner_field_accesses.items()
        }
        for k in field_accesses.keys():
            if k not in local_name_map:
                local_name_map[k] = k

        field_decls = dict(node.field_decls)
        for name, cached_name in local_name_map.items():
            if cached_name in node.read_accesses or cached_name in node.write_accesses:
                continue
            while name.startswith("__local_"):
                name = name[len("__local_") :]
            main_decl = node.field_decls[name]

            stride = 1
            strides = []
            if cached_name.startswith("__local_"):
                shape = field_accesses[cached_name].overapproximated_shape
            else:
                shape = field_accesses[cached_name].shape
            for s in reversed(shape):
                strides = [stride, *strides]
                stride = f"({stride}) * ({s})"
            if is_add_caches:
                storage = dcir.StorageType.from_dace_storage(localcache_infos[axis].storage)
            else:
                storage = dcir.StorageType.Default
            field_decls[cached_name] = dcir.FieldDecl(
                name=cached_name,
                access_info=field_accesses[cached_name],
                dtype=main_decl.dtype,
                data_dims=main_decl.data_dims,
                strides=strides,
                storage=storage,
            )

        states = propagate_field_decls(
            states,
            decl_map={name: (decl.strides, decl.storage) for name, decl in field_decls.items()},
        )

        return dcir.StateMachine(
            label=node.label,
            states=states,
            read_accesses=node.read_accesses,
            write_accesses=node.write_accesses,
            field_decls=field_decls,
            symbols=node.symbols,
            name_map=node.name_map,
        )
