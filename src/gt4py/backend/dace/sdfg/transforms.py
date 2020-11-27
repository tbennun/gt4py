import copy
from collections import defaultdict
from typing import Dict, Set

import dace
from dace import registry
from dace import subsets
from dace import symbolic
from dace.properties import Property, make_properties
from dace import SDFG, SDFGState, nodes
from dace.sdfg import utils as sdutil

from dace.transformation.transformation import Transformation, PatternNode
from dace.transformation.interstate.loop_detection import DetectLoop, find_for_loop
from dace.transformation.interstate.loop_unroll import LoopUnroll

from gt4py.backend.dace.sdfg import library


def global_ij_tiling(sdfg, tile_size=(8, 8)):
    input_arrays = dict()
    output_arrays = dict()
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                if (
                    node.access is dace.AccessType.ReadOnly
                    or node.access is dace.AccessType.ReadWrite
                ) and not sdfg.arrays[node.data].transient:
                    num_accesses = input_arrays.get(node.data, 0)
                    input_arrays[node.data] = num_accesses + sum(
                        [e.data.num_accesses for e in state.out_edges(node)]
                    )

                if (
                    node.access is dace.AccessType.WriteOnly
                    or node.access is dace.AccessType.ReadWrite
                ) and not sdfg.arrays[node.data].transient:
                    num_accesses = output_arrays.get(node.data, 0)
                    output_arrays[node.data] = num_accesses + sum(
                        [e.data.num_accesses for e in state.in_edges(node)]
                    )

    # nest state
    import copy

    tmp_sdfg = copy.deepcopy(sdfg)
    for s in sdfg.nodes():
        sdfg.remove_node(s)
    state = sdfg.add_state()
    nsdfg_node = state.add_nested_sdfg(
        tmp_sdfg, sdfg, list(input_arrays.keys()), list(output_arrays.keys())
    )
    nsdfg_node.symbol_mapping.update(
        # I=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[0]}, I-tile_i*{tile_size[0]})"),
        # J=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[1]}, J-tile_j*{tile_size[1]})"),
        I=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[0]}, I-tile_i)"),
        J=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[1]}, J-tile_j)"),
    )
    # map
    map_entry, map_exit = state.add_map(
        "global_tiling",
        ndrange=dict(
            # tile_i=f"0:int_ceil(I, {tile_size[0]})", tile_j=f"0:int_ceil(J, {tile_size[1]})"
            tile_i=f"0:I:{tile_size[0]}",
            tile_j=f"0:J:{tile_size[1]}",
        ),
    )
    map_entry.map.collapse = 2

    # conn_id = 0
    for array_name, num_accesses in input_arrays.items():
        array = sdfg.arrays[array_name]

        if not array.transient:
            map_entry.add_in_connector("IN_" + array_name)
            map_entry.add_out_connector("OUT_" + array_name)

            state.add_edge(
                state.add_read(array_name),
                None,
                map_entry,
                "IN_" + array_name,
                # f"IN_{conn_id}",
                memlet=dace.Memlet.simple(
                    array_name,
                    subset_str=",".join(
                        [f"0:{limit}" if str(limit) != "1" else "0" for limit in array.shape]
                    ),
                    num_accesses=num_accesses,
                ),
            )
            from dace.data import Scalar

            if isinstance(array, dace.data.Scalar):
                subset_str = "0"
            else:
                frame_i = dace.symbolic.pystr_to_symbolic(str(array.shape[0]) + "-I")
                frame_j = dace.symbolic.pystr_to_symbolic(str(array.shape[1]) + "-J")
                subset_str = ",".join(
                    [
                        # f"{tile_size[0]}*tile_i:Min({tile_size[0]}*(tile_i+1),I)+{frame_i}",
                        # f"{tile_size[1]}*tile_j:Min({tile_size[1]}*(tile_j+1),J)+{frame_j}",
                        f"tile_i:Min(tile_i+{tile_size[0]},I)+{frame_i}",
                        f"tile_j:Min(tile_j+{tile_size[1]},J)+{frame_j}",
                        f"0:{array.shape[2]}",
                    ]
                )

            state.add_edge(
                map_entry,
                "OUT_" + array_name,
                nsdfg_node,
                array_name,
                memlet=dace.Memlet.simple(
                    array_name, subset_str=subset_str, num_accesses=num_accesses
                ),
            )
        # conn_id += 1
    # conn_id = 0
    for array_name, num_accesses in output_arrays.items():
        array = sdfg.arrays[array_name]

        if not array.transient:
            map_exit.add_in_connector("IN_" + array_name)
            map_exit.add_out_connector("OUT_" + array_name)
            state.add_edge(
                map_exit,
                "OUT_" + array_name,
                state.add_write(array_name),
                None,
                memlet=dace.Memlet.simple(
                    array_name,
                    subset_str=",".join(
                        [f"0:{limit}" if str(limit) != "1" else "0" for limit in array.shape]
                    ),
                    num_accesses=num_accesses,
                ),
            )
            from dace.data import Scalar

            if isinstance(array, dace.data.Scalar):
                subset_str = "0"
            else:
                frame_i = dace.symbolic.pystr_to_symbolic(str(array.shape[0]) + "-I")
                frame_j = dace.symbolic.pystr_to_symbolic(str(array.shape[1]) + "-J")
                subset_str = ",".join(
                    [
                        # f"{tile_size[0]}*tile_i:Min({tile_size[0]+1}*tile_i,I)+{frame_i}",
                        # f"{tile_size[1]}*tile_j:Min({tile_size[1]+1}*tile_j,J)+{frame_j}",
                        f"tile_i:Min(tile_i+{tile_size[0]},I)+{frame_i}",
                        f"tile_j:Min(tile_j+{tile_size[1]},J)+{frame_j}",
                        f"0:{array.shape[2]}",
                    ]
                )

            state.add_edge(
                nsdfg_node,
                array_name,
                map_exit,
                "IN_" + array_name,
                memlet=dace.Memlet.simple(
                    array_name, subset_str=subset_str, num_accesses=num_accesses
                ),
            )

    if len(input_arrays) == 0:
        state.add_edge(map_entry, None, nsdfg_node, None, dace.Memlet())
    if len(output_arrays) == 0:
        state.add_edge(nsdfg_node, None, map_exit, None, dace.Memlet())

    # dace.dtypes.StorageType.register("CPU_Threadprivate_Persistent")
    import sympy

    # symbols = dict(_tile_I=dace.symbol("_tile_I"), _tile_J=dace.symbol("_tile_J"))
    # symbols['_tile_I'].set(tile_size[0])
    # symbols['_tile_J'].set(tile_size[1])
    # tile_sizes = dict(I=tile_size[0], J=tile_size[1], K="K")
    for array_name, array in nsdfg_node.sdfg.arrays.items():
        if array.transient:
            # array.shape = [
            #     f"{tile_sizes[str(s)]}"
            #     if isinstance(s, dace.symbolic.symbol)
            #     else s.subs({a: tile_sizes[str(a)] for a in s.args if str(a) in "IJ"})
            #     for s in array.shape
            # ]
            array.tile_size = tile_size
            # print()
            array.storage = dace.dtypes.StorageType.CPU_ThreadLocal


@registry.autoregister_params(singlestate=True)
class PruneTransientOutputs(Transformation):

    _library_node = nodes.LibraryNode("")
    _access_node = nodes.AccessNode("")

    @staticmethod
    def expressions():
        return [
            dace.sdfg.utils.node_path_graph(
                PruneTransientOutputs._library_node, PruneTransientOutputs._access_node
            )
        ]

    @staticmethod
    def _overlap(subset_a: dace.memlet.subsets.Subset, subset_b: dace.memlet.subsets.Subset):


        subset_a = copy.deepcopy(subset_a)
        subset_b = copy.deepcopy(subset_b)
        ranges_a = list(subset_a.ranges[2])
        ranges_b = list(subset_b.ranges[2])
        if len(ranges_a[0].free_symbols) > 0:
            ranges_a[0] = ranges_a[0].replace(next(iter(ranges_a[0].free_symbols)),10000000)
        if len(ranges_a[1].free_symbols) > 0:
            ranges_a[1] = ranges_a[1].replace(next(iter(ranges_a[1].free_symbols)),10000000)
        if len(ranges_b[0].free_symbols) > 0:
            ranges_b[0] = ranges_b[0].replace(next(iter(ranges_b[0].free_symbols)),10000000)
        if len(ranges_b[1].free_symbols) > 0:
            ranges_b[1] = ranges_b[1].replace(next(iter(ranges_b[1].free_symbols)),10000000)
        subset_a.ranges[2] = ranges_a
        subset_b.ranges[2] = ranges_b
        res = dace.subsets.intersects(subset_a, subset_b)
        res = res if res is not None else True
        return res
        # import dace.subsets
        #

    @staticmethod
    def _check_reads(state: dace.SDFGState, candidate_subset, sorted_accesses):

        for acc in sorted_accesses:
            out_edges = state.out_edges(acc)
            if len(out_edges) == 0:
                assert acc.access == dace.dtypes.AccessType.WriteOnly
            for edge in out_edges:
                if not edge.data.data == acc.data:
                    return False
                if PruneTransientOutputs._overlap(edge.data.subset, candidate_subset):
                    return False
        return True

    @staticmethod
    def can_be_applied(
        graph: dace.sdfg.SDFGState, candidate, expr_index, sdfg: dace.SDFG, strict=False
    ):
        # TODO improvement: state-graphs that are not just sequences
        # TODO improvement: can still apply if read is shadowed by another write

        library_node: nodes.LibraryNode = graph.node(
            candidate[PruneTransientOutputs._library_node]
        )

        if not isinstance(library_node, library.StencilLibraryNode):
            return False
        access_node: nodes.AccessNode = graph.node(candidate[PruneTransientOutputs._access_node])

        edges = graph.edges_between(library_node, access_node)
        if len(edges) != 1:
            return False
        candidate_edge = edges[0]
        assert candidate_edge.data.data == access_node.data
        assert access_node.access != dace.dtypes.AccessType.ReadOnly

        candidate_subset = candidate_edge.data.subset
        if not sdfg.arrays[access_node.data].transient:
            return False

        import networkx as nx

        sorted_accesses = [access_node] + [
            node
            for node in nx.algorithms.dag.topological_sort(graph.nx)
            if isinstance(node, nodes.AccessNode) and node.data == access_node.data
        ]

        if not PruneTransientOutputs._check_reads(graph, candidate_subset, sorted_accesses):
            return False

        boundary_states = sdfg.successors(graph)
        visited_states = {graph}
        while len(boundary_states) == 1:
            state = boundary_states[0]
            if state in visited_states:
                return False  # currently only apply if is linear sequence of states.
            visited_states.add(state)
            sorted_accesses = [
                node
                for node in nx.algorithms.dag.topological_sort(state.nx)
                if isinstance(node, nodes.AccessNode) and node.data == access_node.data
            ]

            if not PruneTransientOutputs._check_reads(state, candidate_subset, sorted_accesses):
                return False

            boundary_states = sdfg.successors(state)

        return True

    def apply(self, sdfg: dace.SDFG):
        graph: dace.sdfg.SDFGState = sdfg.nodes()[self.state_id]
        library_node: library.StencilLibraryNode = graph.node(
            self.subgraph[PruneTransientOutputs._library_node]
        )
        access_node: nodes.AccessNode = graph.node(
            self.subgraph[PruneTransientOutputs._access_node]
        )
        edges = graph.edges_between(library_node, access_node)

        in_edge = edges[0]

        data = access_node.data

        library_node.remove_out_connector("OUT_" + data)
        library_node.outputs.remove(data)
        for name, acc in dict(library_node.write_accesses.items()).items():
            if acc.outer_name == data:
                del library_node.write_accesses[name]
        for int in library_node.intervals:
            # if data in int.input_extents:
            #     del int.input_extents[data]
            for state in int.sdfg.nodes():
                tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]
                assert len(tasklets) == 1
                tasklet: nodes.Tasklet = tasklets[0]
                remove_connectors = set()
                for conn in tasklet.out_connectors:
                    if conn.startswith(f"_gt_loc_out__{data}_"):
                        remove_connectors.add(conn)
                for conn in remove_connectors:
                    tasklet.remove_out_connector(conn)

                output_accessors = [
                    n
                    for n in state.nodes()
                    if isinstance(n, nodes.AccessNode)
                    and n.access != dace.dtypes.AccessType.ReadOnly
                    and n.data == data
                ]
                assert len(output_accessors) == 1
                acc = output_accessors[0]
                assert acc.access == dace.dtypes.AccessType.WriteOnly
                inner_in_edge = state.in_edges(acc)
                assert len(inner_in_edge) == 1
                state.remove_edge(inner_in_edge[0])
                state.remove_node(acc)
                if (
                    len(
                        [
                            n
                            for n in state.nodes()
                            if isinstance(n, nodes.AccessNode) and n.data == data
                        ]
                    )
                    == 0
                ):
                    int.sdfg.remove_data(data)
        graph.remove_edge(in_edge)
        if access_node.access == dace.dtypes.AccessType.ReadWrite:
            access_node.access = dace.dtypes.AccessType.WriteOnly
        if len(graph.out_edges(access_node)) == 0:
            graph.remove_node(access_node)

        remove = True
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data == data:
                    remove = False
        if remove:
            sdfg.remove_data(data)


@registry.autoregister_params(singlestate=True)
@make_properties
class TaskletAsKLoop(Transformation):
    """Docstring TODO"""

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _tasklet = nodes.Tasklet("")
    _map_exit = nodes.MapExit(nodes.Map("", [], []))

    # Properties
    init = Property(default=0, desc="initial value for k")
    condition = Property(default="k<K", desc="stopping condition for the loop")
    step = Property(default="k+1", desc="value assigned to k every step (e.g. increment k+1)")

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [
            dace.sdfg.utils.node_path_graph(
                TaskletAsKLoop._map_entry, TaskletAsKLoop._tasklet, TaskletAsKLoop._map_exit
            )
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    def _k_range(self):
        if "<" in self.condition:
            k_min = self.init
            _, k_max = self.condition.split("<")
            k_max = k_max + " - 1"
        else:
            k_max = str(self.init)
            _, k_min = self.condition.split(">=")
        return k_min, k_max

    def apply(self, sdfg):
        graph: dace.sdfg.SDFGState = sdfg.nodes()[self.state_id]
        map_entry: nodes.MapEntry = graph.nodes()[self.subgraph[TaskletAsKLoop._map_entry]]
        tasklet: nodes.Tasklet = graph.nodes()[self.subgraph[TaskletAsKLoop._tasklet]]
        map_exit: nodes.MapExit = graph.nodes()[self.subgraph[TaskletAsKLoop._map_exit]]

        k_min, k_max = self._k_range()
        # fix outer edges to ij map
        import sympy

        k_symbol = dace.symbolic.symbol("k")
        for e in graph.in_edges(map_entry) + graph.out_edges(map_exit):
            for i, r in enumerate(e.data.subset.ranges):
                e.data.subset.ranges[i] = (
                    r[0].subs(dace.symbolic.symbol("k"), k_min),
                    r[1].subs(dace.symbolic.symbol("k"), k_max),
                    r[2],
                )

        # node = nest_state_subgraph(sdfg, graph, dace.sdfg.ScopeSubgraphView(graph, [tasklet]))
        nsdfg: SDFG = dace.SDFG(f"nested_k_loop_{graph.name}")
        nstate = nsdfg.add_state()
        nstate.add_nodes_from([tasklet])
        # nsdfg.add_nodes_from(dace.sdfg.ScopeSubgraphView(graph, [nstate]))

        in_prefix = f"__in_"
        out_prefix = f"__out_"

        nsdfg_in_arrays = set()
        for e in graph.out_edges(map_entry):
            nsdfg_in_arrays.add(in_prefix + e.data.data)
        nsdfg_out_arrays = set()
        for e in graph.in_edges(map_exit):
            nsdfg_out_arrays.add(out_prefix + e.data.data)

        for name in set(
            n.data
            for n in graph.nodes()
            if isinstance(n, nodes.AccessNode) and n.access == dace.dtypes.AccessType.ReadOnly
        ):
            nsdfg.add_datadesc(in_prefix + name, copy.deepcopy(sdfg.arrays[name]))
        for name in set(
            n.data
            for n in graph.nodes()
            if isinstance(n, nodes.AccessNode) and n.access == dace.dtypes.AccessType.WriteOnly
        ):
            nsdfg.add_datadesc(out_prefix + name, copy.deepcopy(sdfg.arrays[name]))

        read_accessors = dict()
        for name in nsdfg_in_arrays:
            read_accessors[name] = nstate.add_read(name)
        write_accessors = dict()
        for name in nsdfg_out_arrays:
            write_accessors[name] = nstate.add_write(name)

        for e in graph.out_edges(map_entry):
            nstate.add_edge(
                read_accessors[in_prefix + e.data.data],
                None,
                tasklet,
                e.dst_conn,
                memlet=dace.Memlet.simple(
                    in_prefix + e.data.data,
                    subset_str=str(e.data.subset),
                    num_accesses=e.data.num_accesses,
                ),
            )
        for e in graph.in_edges(map_exit):
            nstate.add_edge(
                tasklet,
                e.src_conn,
                write_accessors[out_prefix + e.data.data],
                None,
                memlet=dace.Memlet.simple(
                    out_prefix + e.data.data,
                    subset_str=str(e.data.subset),
                    num_accesses=e.data.num_accesses,
                ),
            )

        node = graph.add_nested_sdfg(nsdfg, sdfg, nsdfg_in_arrays, nsdfg_out_arrays)
        nstate = nsdfg.nodes()[0]

        conn_map_entry_to_nsdfg = dict()
        subsets_map_entry_to_nsdfg = dict()
        num_map_entry_to_nsdfg = dict()
        for e in graph.out_edges(map_entry):
            conn_map_entry_to_nsdfg[e.src_conn] = e.data.data

            subset = subsets_map_entry_to_nsdfg.get(e.data.data, e.data.subset)
            num = num_map_entry_to_nsdfg.get(e.data.data, e.data.num_accesses)
            for i, r in enumerate(subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        min(subset.ranges[i][0], e.data.subset[i][0]),
                        max(subset.ranges[i][1], e.data.subset[i][1]),
                        1,
                    )
                elif "k" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        0,
                        dace.symbolic.pystr_to_symbolic("K-1"),
                        1,
                    )  # graph.edges_between(
                    #     [
                    #         n
                    #         for n in graph.nodes()
                    #         if isinstance(n, nodes.AccessNode)
                    #         and n.access == dace.AccessType.ReadOnly
                    #         and n.data == e.data.data
                    #     ][0],
                    #     map_entry,
                    # )[0].data.subset.ranges[i]
                subsets_map_entry_to_nsdfg[e.data.data] = subset
                num_map_entry_to_nsdfg[e.data.data] = num + e.data.num_accesses

        conn_map_exit_to_nsdfg = dict()
        for e in graph.in_edges(map_exit):
            conn_map_exit_to_nsdfg[e.dst_conn] = e.data.data

        for conn in map_entry.out_connectors:
            data_name = conn_map_entry_to_nsdfg[conn]
            graph.add_edge(
                map_entry,
                conn,
                node,
                in_prefix + conn_map_entry_to_nsdfg[conn],
                memlet=dace.Memlet.simple(
                    data=data_name,
                    subset_str=str(subsets_map_entry_to_nsdfg[data_name]),
                    num_accesses=num_map_entry_to_nsdfg[data_name],
                ),
            )

        conn_nsdfg_to_map_exit = dict()
        subsets_nsdfg_to_map_exit = dict()
        num_nsdfg_to_map_exit = dict()
        for e in graph.in_edges(map_exit):
            conn_nsdfg_to_map_exit[e.dst_conn] = e.data.data

            subset = subsets_nsdfg_to_map_exit.get(e.data.data, e.data.subset)
            num = num_nsdfg_to_map_exit.get(e.data.data, e.data.num_accesses)
            for i, r in enumerate(subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        min(subset.ranges[i][0], e.data.subset[i][0]),
                        max(subset.ranges[i][1], e.data.subset[i][1]),
                        1,
                    )
                elif "k" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        0,
                        dace.symbolic.pystr_to_symbolic("K-1"),
                        1,
                    )  # graph.edges_between(
                    #     map_exit,
                    #     [
                    #         n
                    #         for n in graph.nodes()
                    #         if isinstance(n, nodes.AccessNode)
                    #         and n.access == dace.AccessType.WriteOnly
                    #         and n.data == e.data.data
                    #     ][0],
                    # )[0].data.subset.ranges[i]
                subsets_nsdfg_to_map_exit[e.data.data] = subset
                num_nsdfg_to_map_exit[e.data.data] = num + e.data.num_accesses
        for conn in map_exit.in_connectors:
            data_name = conn_nsdfg_to_map_exit[conn]
            graph.add_edge(
                node,
                out_prefix + conn_map_exit_to_nsdfg[conn],
                map_exit,
                conn,
                memlet=dace.Memlet.simple(
                    data=data_name,
                    subset_str=str(subsets_nsdfg_to_map_exit[data_name]),
                    num_accesses=num_nsdfg_to_map_exit[data_name],
                ),
            )
        for e in graph.in_edges(map_entry) + graph.out_edges(map_exit):
            if len(e.data.subset.ranges) >= 3 and "k" in dace.symbolic.symlist(
                e.data.subset.ranges[2]
            ):
                e.data.subset.ranges[2] = (0, dace.symbolic.pystr_to_symbolic("K-1"), 1)

        for e in nstate.in_edges(tasklet):
            outer_subset = subsets_map_entry_to_nsdfg[e.data.data[len(in_prefix) :]]
            for i, r in enumerate(e.data.subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    e.data.subset.ranges[i] = (
                        r[0] - outer_subset.ranges[i][0],
                        r[1] - outer_subset.ranges[i][0],
                        1,
                    )

        for e in nstate.out_edges(tasklet):
            outer_subset = subsets_nsdfg_to_map_exit[e.data.data[len(out_prefix) :]]
            for i, r in enumerate(e.data.subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    e.data.subset.ranges[i] = (
                        r[0] - outer_subset.ranges[i][0],
                        r[1] - outer_subset.ranges[i][0],
                        1,
                    )

        # Create a loop inside the nested SDFG
        nsdfg.add_loop(None, nstate, None, "k", self.init, self.condition, self.step)
        graph.remove_node(tasklet)
        # outer_in_edges = {e.dst_conn: e for e in graph.in_edges(node)}
        # outer_out_edges = {e.src_conn: e for e in graph.out_edges(node)}
        #
        # for e in nstate.in_edges(tasklet):
        #     assert all(r == (0, 0, 1) for r in e.data.subset.ranges)
        #     assert e.src.data in outer_in_edges
        #     outer_edge = outer_in_edges[e.src.data]
        #     for i, r in enumerate(outer_edge.data.subset.ranges):
        #         e.data.subset.ranges[i] = r
        #
        # for e in nstate.out_edges(tasklet):
        #     assert all(r == (0, 0, 1) for r in e.data.subset.ranges)
        #     assert e.dst.data in outer_out_edges
        #     outer_edge = outer_out_edges[e.dst.data]
        #     for i, r in enumerate(outer_edge.data.subset.ranges):
        #         e.data.subset.ranges[i] = r

        #     e.data.subset.ranges[i] = r
        # if len(e.data.subset.ranges) > 2:
        #     e.data.subset.ranges[2] = (
        #         dace.symbolic.pystr_to_symbolic("k"),
        #         dace.symbolic.pystr_to_symbolic("k"),
        #         dace.symbolic.pystr_to_symbolic("1"),
        #     )


class EnhancedDetectLoop(DetectLoop):
    """Detects a for-loop construct from an SDFG, with added utility function for finding
    context states."""

    def _get_context_subgraph(self, sdfg):
        # Obtain loop information
        guard: dace.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_guard])
        begin: dace.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_begin])
        after_state: dace.SDFGState = sdfg.node(self.subgraph[DetectLoop._exit_state])

        # Obtain iteration variable, range, and stride
        guard_inedges = sdfg.in_edges(guard)
        condition_edge = sdfg.edges_between(guard, begin)[0]
        itervar = list(guard_inedges[0].data.assignments.keys())[0]
        condition = condition_edge.data.condition_sympy()
        rng = LoopUnroll._loop_range(itervar, guard_inedges, condition)

        # Find the state prior to the loop
        if rng[0] == dace.symbolic.pystr_to_symbolic(guard_inedges[0].data.assignments[itervar]):
            before_state: dace.SDFGState = guard_inedges[0].src
            last_state: dace.SDFGState = guard_inedges[1].src
        else:
            before_state: dace.SDFGState = guard_inedges[1].src
            last_state: dace.SDFGState = guard_inedges[0].src

        return guard, begin, last_state, before_state, after_state


@registry.autoregister
@make_properties
class RemoveTrivialLoop(EnhancedDetectLoop):

    count = 1

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        if not EnhancedDetectLoop.can_be_applied(graph, candidate, expr_index, sdfg, strict):
            return False

        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])

        # Obtain iteration variable, range, and stride
        guard_inedges = graph.in_edges(guard)
        condition_edge = graph.edges_between(guard, begin)[0]
        itervar = list(guard_inedges[0].data.assignments.keys())[0]
        condition = condition_edge.data.condition_sympy()

        # If loop cannot be detected, fail
        rng = LoopUnroll._loop_range(itervar, guard_inedges, condition)
        if not rng:
            return False

        start, end, step = rng

        try:
            return bool(start == end)
        except TypeError:
            return False

    def apply(self, sdfg):
        guard, first_state, last_state, before_state, after_state = self._get_context_subgraph(
            sdfg
        )
        # guard_inedges = sdfg.in_edges(guard)
        # condition_edge = sdfg.edges_between(guard, first_state)[0]
        # itervar = list(guard_inedges[0].data.assignments.keys())[0]
        # condition = condition_edge.data.condition_sympy()

        init_edges = sdfg.edges_between(before_state, guard)
        assert len(init_edges) == 1
        init_edge = init_edges[0]
        sdfg.add_edge(
            before_state,
            first_state,
            dace.InterstateEdge(
                condition=init_edge.data.condition, assignments=init_edge.data.assignments
            ),
        )
        sdfg.remove_edge(init_edge)
        # add edge from pred directly to loop states

        # sdfg.add_edge(before_state, first_state, dace.InterstateEdge(assignments=init_edge.assignments))
        exit_edge = sdfg.edges_between(last_state, guard)[0]
        sdfg.add_edge(
            last_state, after_state, dace.InterstateEdge(assignments=exit_edge.data.assignments)
        )
        sdfg.remove_edge(exit_edge)

        # remove guard
        sdfg.remove_edge(sdfg.edges_between(guard, first_state)[0])
        sdfg.remove_edge(sdfg.edges_between(guard, after_state)[0])
        sdfg.remove_node(guard)


#
# def eliminate_trivial_k_loop(sdfg: dace.SDFG, state: dace.SDFGState):
#     sdfg.predecessor_states(state)
#     if not len(sdfg.successors(state)) == 2:
#         return
#     if not len(sdfg.predecessors(state)) == 2:
#         return
#     init, condition, step = None, None, None
#     for s in sdfg.predecessors(state):
#         edges = sdfg.edges_between(s, state)
#         if not len(edges) == 1:
#             return
#         if edges[0].data.condition.as_string == "" and s in sdfg.predecessor_states(state):
#             init = edges[0].data.assignments["k"]
#             init_state = s
#         elif not edges[0].data.condition.as_string == "":
#             return
#         else:
#             step = edges[0].data.assignments["k"]
#             loop_end_state = s
#     for s in sdfg.successors(state):
#         edges = sdfg.edges_between(state, s)
#         if edges:
#             if not len(edges) == 1:
#                 return
#             if not edges[0].data.condition.as_string == "":
#                 condition = edges[0].data.condition
#                 loop_start_state = s
#             else:
#                 exit_state = s
#
#     if "<" in condition.as_string:
#         k_min = init
#         _, k_max = condition.as_string.split("<")
#         k_max = k_max + " - 1"
#     else:
#         k_max = str(init)
#         _, k_min = condition.as_string.split(">=")
#
#     if not dace.symbolic.pystr_to_symbolic(f"({k_min})-({k_max})") == 0:
#         return
#
#     # add edge from pred directly to loop states
#     sdfg.add_edge(init_state, loop_start_state, dace.InterstateEdge(assignments={"k": init}))
#     # add edge from loop states directly to succ
#     sdfg.add_edge(loop_end_state, exit_state, dace.InterstateEdge())
#     # remove guard & edges involving guard
#     for s in sdfg.successors(state):
#         for edge in sdfg.edges_between(state, s):
#             sdfg.remove_edge(edge)
#     for s in sdfg.predecessors(state):
#         for edge in sdfg.edges_between(s, state):
#             sdfg.remove_edge(edge)
#     sdfg.remove_node(state)


def outer_k_loop_to_inner_map(sdfg: dace.SDFG, state: dace.SDFGState):
    sdfg.predecessor_states(state)
    if not len(sdfg.successors(state)) == 2:
        return
    if not len(sdfg.predecessors(state)) == 2:
        return
    init, condition, step = None, None, None
    for s in sdfg.predecessors(state):
        edges = sdfg.edges_between(s, state)
        if not len(edges) == 1:
            return
        if edges[0].data.condition.as_string == "" and s in sdfg.predecessor_states(state):
            init = edges[0].data.assignments["k"]
            init_state = s
        elif not edges[0].data.condition.as_string == "":
            return
        else:
            step = edges[0].data.assignments["k"]
            loop_end_state = s
    for s in sdfg.successors(state):
        edges = sdfg.edges_between(state, s)
        if edges:
            if not len(edges) == 1:
                return
            if not edges[0].data.condition.as_string == "":
                condition = edges[0].data.condition
                loop_start_state = s
            else:
                exit_state = s
    # for state in loop...
    loop_states = []
    s = loop_start_state
    while s is not state:
        if not len(sdfg.successors(s)) == 1:
            return
        else:
            loop_states.append(s)
            s = sdfg.successors(s)[0]
    assert loop_end_state is loop_states[-1]

    # replace tasklet with nestedsdfg
    for s in loop_states:
        sdfg.apply_transformations(
            TaskletAsKLoop,
            states=[s],
            validate=False,
            options=dict(init=init, step=step, condition=condition.as_string),
        )
    # add edge from pred directly to loop states
    sdfg.add_edge(init_state, loop_start_state, dace.InterstateEdge())
    # add edge from loop states directly to succ
    sdfg.add_edge(loop_end_state, exit_state, dace.InterstateEdge())
    # remove guard & edges involving guard
    for s in sdfg.successors(state):
        for edge in sdfg.edges_between(state, s):
            sdfg.remove_edge(edge)
    for s in sdfg.predecessors(state):
        for edge in sdfg.edges_between(s, state):
            sdfg.remove_edge(edge)
    sdfg.remove_node(state)



@registry.autoregister
@make_properties
class LoopBufferCache(DetectLoop):
    storage_type = Property(
        dtype=dace.dtypes.StorageType,
        default=dace.dtypes.StorageType.Default,
        desc="the StorageType of local buffers",
    )

    @staticmethod
    def collect_subset_info(state, name, var_idx):
        in_subsets = set()
        out_subsets = set()
        for edge in state.edges():
            if isinstance(edge.dst, nodes.CodeNode) and edge.data.data == name:
                in_subsets.add(copy.deepcopy(edge.data.subset))
            if (
                isinstance(edge.dst, nodes.AccessNode)
                and edge.dst.access == dace.dtypes.AccessType.ReadWrite
                and edge.data.data == name
            ):
                in_subsets.add(copy.deepcopy(edge.data.subset))
            if isinstance(edge.src, nodes.CodeNode) and edge.data.data == name:
                out_subsets.add(copy.deepcopy(edge.data.subset))
            if (
                isinstance(edge.src, nodes.AccessNode)
                and edge.src.access == dace.dtypes.AccessType.ReadWrite
                and edge.data.data == name
            ):
                out_subsets.add(copy.deepcopy(edge.data.subset))

        outer_in_subsets = set()
        outer_out_subsets = set()

        for edge in state.edges():
            if (
                isinstance(edge.src, nodes.AccessNode)
                and edge.src.access == dace.dtypes.AccessType.ReadOnly
                and edge.data.data == name
            ):
                outer_in_subsets.add(copy.deepcopy(edge.data.subset))
            if (
                isinstance(edge.dst, nodes.AccessNode)
                and edge.dst.access == dace.dtypes.AccessType.WriteOnly
                and edge.data.data == name
            ):
                outer_out_subsets.add(copy.deepcopy(edge.data.subset))

        indices = set(subs.ranges[var_idx][0] for subs in in_subsets | out_subsets)
        indices |= set(subs.ranges[var_idx][1] for subs in in_subsets | out_subsets)
        length = max(indices) - min(indices) + 1

        outer_in_subset = None
        if len(outer_in_subsets) > 0:
            outer_in_subset = next(iter(outer_in_subsets))
            for subset in outer_in_subsets:
                outer_in_subset = dace.subsets.union(outer_in_subset, subset)
            for i in range(3):
                if i != var_idx:
                    for subset in in_subsets:
                        subset.ranges[i] = outer_in_subset.ranges[i]

        outer_out_subset = None
        if len(outer_out_subsets) > 0:
            outer_out_subset = next(iter(outer_out_subsets))
            for subset in outer_out_subsets:
                outer_out_subset = dace.subsets.union(outer_out_subset, subset)
            for i in range(3):
                if i != var_idx:
                    for subset in out_subsets:
                        subset.ranges[i] = outer_out_subset.ranges[i]

        if len(out_subsets) == 1:
            subset = next(iter(out_subsets))
            assert subset.ranges[var_idx][0] == subset.ranges[var_idx][1]
            write_idx = subset.ranges[var_idx][0]
        else:
            assert len(out_subsets) == 0
            write_idx = None

        if outer_in_subset is not None and outer_out_subset is not None:
            unified_subset = dace.subsets.union(outer_in_subset, outer_out_subset)
        else:
            unified_subset = outer_in_subset or outer_out_subset

        min_idx = unified_subset[var_idx][0]
        max_idx = unified_subset[var_idx][1]

        return dict(
            length=length,
            unified_subset=unified_subset,
            in_subsets=in_subsets,
            out_subsets=out_subsets,
            min_idx=min_idx,
            max_idx=max_idx,
            write_idx=write_idx,
        )

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        if not DetectLoop.can_be_applied(graph, candidate, expr_index, sdfg, strict):
            return False

        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])

        loop_states = list(
            dace.sdfg.utils.dfs_conditional(
                graph, sources=[begin], condition=lambda _, child: child != guard
            )
        )
        if len(loop_states) != 1:
            return False

        itervar, _, (_, loop_state) = find_for_loop(sdfg, guard, begin)
        var_idx = "ijk".index(str(itervar))
        for name in graph.arrays.keys():
            subset_info = LoopBufferCache.collect_subset_info(loop_state, name, var_idx=var_idx)
            if subset_info["length"] > 1:
                return True

        return False

    def apply(self, sdfg):
        ####################################################################
        # Obtain loop information
        guard: dace.SDFGState = sdfg.node(self.subgraph[self._loop_guard])
        begin: dace.SDFGState = sdfg.node(self.subgraph[self._loop_begin])
        after_state: dace.SDFGState = sdfg.node(self.subgraph[self._exit_state])

        # Obtain iteration variable, range, and stride
        guard_inedges = sdfg.in_edges(guard)
        condition_edge = sdfg.edges_between(guard, begin)[0]

        sdfg.validate()
        assert sdfg.parent is None or sdfg.parent_nsdfg_node in sdfg.parent.nodes()

        if sdfg.parent is not None:
            loads = {e.data.data for e in sdfg.parent.in_edges(sdfg.parent_nsdfg_node)}
            stores = {e.data.data for e in sdfg.parent.out_edges(sdfg.parent_nsdfg_node)}
        else:
            loads = stores = set()

        itervar, rng, ([*_, before_state], loop_state) = find_for_loop(sdfg, guard, begin)
        itervar_sym = dace.symbolic.pystr_to_symbolic(itervar)
        condition = condition_edge.data.condition_sympy()

        var_idx = "ijk".index(str(itervar))

        subset_infos = {}
        for name in sdfg.arrays.keys():
            subset_infos[name] = self.collect_subset_info(loop_state, name, var_idx=var_idx)

        fill_state = sdfg.add_state(sdfg.label + "_fill_state")
        edge = sdfg.edges_between(before_state, guard)[0]
        sdfg.add_edge(
            before_state,
            fill_state,
            dace.InterstateEdge(condition=edge.data.condition, assignments=edge.data.assignments),
        )
        sdfg.add_edge(fill_state, guard, dace.InterstateEdge())
        sdfg.remove_edge(edge)

        load_state = sdfg.add_state(sdfg.label + "_load_state")
        edge = sdfg.edges_between(guard, loop_state)[0]
        sdfg.add_edge(
            guard,
            load_state,
            dace.InterstateEdge(condition=edge.data.condition, assignments=edge.data.assignments),
        )
        sdfg.add_edge(load_state, loop_state, dace.InterstateEdge())
        sdfg.remove_edge(edge)

        store_shift_state = sdfg.add_state(sdfg.label + "_store_shift_state")
        edge = sdfg.edges_between(loop_state, guard)[0]
        sdfg.add_edge(loop_state, store_shift_state, dace.InterstateEdge())
        sdfg.add_edge(
            store_shift_state,
            guard,
            dace.InterstateEdge(condition=edge.data.condition, assignments=edge.data.assignments),
        )
        sdfg.remove_edge(edge)

        for name, array in list(sdfg.arrays.items()):
            subset_info = subset_infos[name]
            if subset_info["length"] <= 1:
                continue

            shape = list(array.shape)
            shape[var_idx] = subset_info["length"]
            sdfg.add_array(
                f"_loc_buf_{name}",
                shape=shape,
                dtype=array.dtype,
                storage=self.storage_type,
                transient=True,
                lifetime=dace.dtypes.AllocationLifetime.SDFG,
            )

            def relative(subset):
                result = copy.deepcopy(subset)
                offset = [0] * result.dims()
                offset[var_idx] = subset_info["unified_subset"][var_idx][0]
                result.offset(offset, negative=True)
                return result

            if name in loads:
                read = fill_state.add_read(name)
                write = fill_state.add_write(f"_loc_buf_{name}")
                src_subset = copy.deepcopy(subset_info["unified_subset"])
                dst_subset = relative(src_subset)
                fill_state.add_edge(
                    read,
                    None,
                    write,
                    None,
                    dace.Memlet(data=name, subset=src_subset, other_subset=dst_subset),
                )

                read = load_state.add_read(name)
                write = load_state.add_write(f"_loc_buf_{name}")
                select_load = min if rng[2] < 0 else max
                src_subset = copy.deepcopy(
                    select_load(subset_info["in_subsets"], key=lambda s: s[var_idx])
                )
                dst_subset = relative(src_subset)
                load_state.add_edge(
                    read,
                    None,
                    write,
                    None,
                    dace.Memlet(data=name, subset=src_subset, other_subset=dst_subset),
                )

            read = store_shift_state.add_read(f"_loc_buf_{name}")
            sdfg.add_datadesc(f"_tmp_loc_buf_{name}", sdfg.arrays[f"_loc_buf_{name}"])
            write = store_shift_state.add_access(f"_tmp_loc_buf_{name}")
            src_subset = relative(subset_info["unified_subset"])
            dst_subset = src_subset
            store_shift_state.add_edge(
                read,
                None,
                write,
                None,
                dace.Memlet(data=f"_loc_buf_{name}", subset=src_subset, other_subset=dst_subset),
            )

            read = write
            write = store_shift_state.add_write(f"_loc_buf_{name}")
            src_subset = copy.deepcopy(subset_info["unified_subset"])
            src_subset.ranges[var_idx] = 1, subset_info["length"] - 1, 1
            dst_subset = copy.deepcopy(src_subset)
            dst_subset.ranges[var_idx] = 0, subset_info["length"] - 2, 1
            if rng[2] < 0:
                src_subset, dst_subset = dst_subset, src_subset
            store_shift_state.add_edge(
                read,
                None,
                write,
                None,
                dace.Memlet(
                    data=f"_tmp_loc_buf_{name}", subset=src_subset, other_subset=dst_subset
                ),
            )

            if name in stores:
                write = store_shift_state.add_write(name)
                assert len(subset_info["out_subsets"]) == 1
                dst_subset = copy.deepcopy(next(iter(subset_info["out_subsets"])))
                src_subset = relative(dst_subset)
                store_shift_state.add_edge(
                    read,
                    None,
                    write,
                    None,
                    dace.Memlet(
                        data=f"_tmp_loc_buf_{name}", subset=src_subset, other_subset=dst_subset
                    ),
                )

            for edge in loop_state.edges():
                if edge.data.data == name:
                    assert edge.data.other_subset is None
                    edge.data.data = f"_loc_buf_{name}"
                    edge.data.subset = relative(edge.data.subset)

            for node in loop_state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data == name:
                    node.data = f"_loc_buf_{name}"

        return sdfg


@registry.autoregister_params(singlestate=True)
class OnTheFlyMapFusion(Transformation):
    _first_map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _first_tasklet = nodes.Tasklet("")
    _first_map_exit = nodes.MapExit(nodes.Map("", [], []))
    _array_access = nodes.AccessNode("")
    _second_map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _second_tasklet = nodes.Tasklet("")

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                OnTheFlyMapFusion._first_map_entry,
                OnTheFlyMapFusion._first_tasklet,
                OnTheFlyMapFusion._first_map_exit,
                OnTheFlyMapFusion._array_access,
                OnTheFlyMapFusion._second_map_entry,
                OnTheFlyMapFusion._second_tasklet,
            )
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        first_map_entry = graph.node(candidate[OnTheFlyMapFusion._first_map_entry])
        first_tasklet = graph.node(candidate[OnTheFlyMapFusion._first_tasklet])
        first_map_exit = graph.node(candidate[OnTheFlyMapFusion._first_map_exit])
        array_access = graph.node(candidate[OnTheFlyMapFusion._array_access])

        if len(first_map_exit.in_connectors) != 1:
            return False

        if graph.in_degree(array_access) != 1 or graph.out_degree(array_access) != 1:
            return False
        return True

    @staticmethod
    def _memlet_offsets(base_memlet, offset_memlet):
        """Compute subset offset of `offset_memlet` relative to `base_memlet`."""

        def offset(base_range, offset_range):
            b0, e0, s0 = base_range
            b1, e1, s1 = offset_range
            assert e1 - e0 == b1 - b0 and s0 == s1
            return int(e1 - e0)

        return tuple(
            offset(b, o) for b, o in zip(base_memlet.subset.ranges, offset_memlet.subset.ranges)
        )

    @staticmethod
    def _update_map_connectors(state, array_access, first_map_entry, second_map_entry):
        """Remove unused connector (of the to-be-replaced array) from second
        map entry, add new connectors to second map entry for the inputs
        used in the first map’s tasklets.
        """
        # Remove edges and connectors from arrays access to second map entry
        for edge in state.edges_between(array_access, second_map_entry):
            state.remove_edge_and_connectors(edge)
        state.remove_node(array_access)

        # Add new connectors to second map
        # TODO: implement for the general case with random naming
        for edge in state.in_edges(first_map_entry):
            if second_map_entry.add_in_connector(edge.dst_conn):
                state.add_edge(edge.src, edge.src_conn, second_map_entry, edge.dst_conn, edge.data)

    @staticmethod
    def _read_offsets(state, array_name, first_map_exit, second_map_entry):
        """Compute offsets of read accesses in second map."""
        # Get output memlet of first tasklet
        output_edges = state.in_edges(first_map_exit)
        assert len(output_edges) == 1
        write_memlet = output_edges[0].data

        # Find read offsets by looping over second map entry connectors
        offsets = defaultdict(list)
        for edge in state.out_edges(second_map_entry):
            if edge.data.data == array_name:
                second_map_entry.remove_out_connector(edge.src_conn)
                state.remove_edge(edge)
                offset = OnTheFlyMapFusion._memlet_offsets(write_memlet, edge.data)
                offsets[offset].append(edge)

        return offsets

    @staticmethod
    def _copy_first_map_contents(sdfg, state, first_map_entry, first_map_exit):
        nodes = list(state.all_nodes_between(first_map_entry, first_map_exit) - {first_map_entry})
        new_nodes = [copy.deepcopy(node) for node in nodes]
        tmp_map = dict()
        for node in new_nodes:
            if isinstance(node, dace.nodes.AccessNode):
                data = sdfg.arrays[node.data]
                if isinstance(data, dace.data.Scalar) and data.transient:
                    tmp_name = sdfg.temp_data_name()
                    sdfg.add_scalar(tmp_name, data.dtype, transient=True)
                    tmp_map[node.data] = tmp_name
                    node.data = tmp_name
            state.add_node(node)
        id_map = {state.node_id(old): state.node_id(new) for old, new in zip(nodes, new_nodes)}

        def map_node(node):
            return state.node(id_map[state.node_id(node)])

        def map_memlet(memlet):
            memlet = copy.deepcopy(memlet)
            memlet.data = tmp_map.get(memlet.data, memlet.data)
            return memlet

        for edge in state.edges():
            if edge.src in nodes or edge.dst in nodes:
                src = map_node(edge.src) if edge.src in nodes else edge.src
                dst = map_node(edge.dst) if edge.dst in nodes else edge.dst
                edge_data = map_memlet(edge.data)
                state.add_edge(src, edge.src_conn, dst, edge.dst_conn, edge_data)

        return new_nodes

    def _replicate_first_map(
        self, sdfg, array_access, first_map_entry, first_map_exit, second_map_entry
    ):
        """Replicate tasklet of first map for reach read access in second map."""
        state = sdfg.node(self.state_id)
        array_name = array_access.data
        array = sdfg.arrays[array_name]

        read_offsets = self._read_offsets(state, array_name, first_map_exit, second_map_entry)

        # Replicate first map tasklets once for each read offset access and
        # connect them to other tasklets accordingly
        for offset, edges in read_offsets.items():
            nodes = self._copy_first_map_contents(sdfg, state, first_map_entry, first_map_exit)
            tmp_name = sdfg.temp_data_name()
            sdfg.add_scalar(tmp_name, array.dtype, transient=True)
            tmp_access = state.add_access(tmp_name)

            for node in nodes:
                for edge in state.edges_between(node, first_map_exit):
                    state.add_edge(
                        edge.src, edge.src_conn, tmp_access, None, dace.Memlet(tmp_name)
                    )
                    state.remove_edge(edge)

                for edge in state.edges_between(first_map_entry, node):
                    memlet = copy.deepcopy(edge.data)
                    memlet.subset.offset(list(offset), negative=False)
                    second_map_entry.add_out_connector(edge.src_conn)
                    state.add_edge(second_map_entry, edge.src_conn, node, edge.dst_conn, memlet)
                    state.remove_edge(edge)

            for edge in edges:
                state.add_edge(tmp_access, None, edge.dst, edge.dst_conn, dace.Memlet(tmp_name))

    def apply(self, sdfg: dace.SDFG):
        state = sdfg.node(self.state_id)
        first_map_entry = state.node(self.subgraph[self._first_map_entry])
        first_tasklet = state.node(self.subgraph[self._first_tasklet])
        first_map_exit = state.node(self.subgraph[self._first_map_exit])
        array_access = state.node(self.subgraph[self._array_access])
        second_map_entry = state.node(self.subgraph[self._second_map_entry])

        self._update_map_connectors(state, array_access, first_map_entry, second_map_entry)

        self._replicate_first_map(
            sdfg, array_access, first_map_entry, first_map_exit, second_map_entry
        )

        state.remove_nodes_from(
            state.all_nodes_between(first_map_entry, first_map_exit) | {
                first_map_entry, first_map_exit }
        )


@registry.autoregister_params(singlestate=True)
class IJMapFusion(Transformation):
    map_entry = PatternNode(nodes.EntryNode)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(IJMapFusion.map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entries = [n for n in graph.nodes() if isinstance(n, nodes.MapEntry)]
        if len(map_entries) <= 1:
            return False

        if any(set(m.map.params) != {"i", "j"} for m in map_entries):
            return False

        return True

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]

        map_entries = [n for n in graph.nodes() if isinstance(n, nodes.MapEntry)]
        map_exits = [n for n in graph.nodes() if isinstance(n, nodes.MapExit)]
        if len(map_entries) <= 1:
            return sdfg

        if any(set(m.map.params) != {"i", "j"} for m in map_entries):
            return sdfg

        i, j = dace.symbol("i"), dace.symbol("j")
        column_subset = [(i, i, 1), (j, j, 1)]
        for map_entry in map_entries:
            for out_edge in graph.out_edges(map_entry):
                if out_edge.data.subset[:2] != column_subset:
                    return sdfg
        for map_exit in map_exits:
            for in_edge in graph.in_edges(map_exit):
                if in_edge.data.subset[:2] != column_subset:
                    return sdfg

        new_entry = nodes.MapEntry(map=copy.deepcopy(map_entries[0].map))
        new_exit = nodes.MapExit(map=map_entry.map)

        nsdfg = dace.SDFG("ij_fused")
        for name, val in sdfg.symbols.items():
            nsdfg.add_symbol(name, val)
        for name, val in sdfg.constants.items():
            nsdfg.add_constant(name, val)

        sources = {n.data for n in graph.source_nodes() if isinstance(n, nodes.AccessNode)}
        sinks = {n.data for n in graph.sink_nodes() if isinstance(n, nodes.AccessNode)}
        for name, array in sdfg.arrays.items():
            narray = copy.deepcopy(array)
            if array.shape[:2] == (dace.symbol("I"), dace.symbol("J")):
                nshape = list(array.shape)
                nshape[:2] = [1, 1]
                narray.shape = tuple(nshape)
            if name in sources or name in sinks:
                narray.transient = False
            else:
                narray.storage = dace.StorageType.Register
            nsdfg.add_datadesc(name, narray)

        ngraph = copy.deepcopy(graph)
        ngraph.instrument = dace.InstrumentationType.No_Instrumentation
        ngraph.parent = nsdfg
        nsdfg.add_node(ngraph)

        remove = set()
        for node in ngraph.nodes():
            if isinstance(node, nodes.MapEntry):
                remove.add(node)
                in_edges = {e.data.data: e for e in ngraph.in_edges(node)}
                out_edges = {e.data.data: e for e in ngraph.out_edges(node)}
                for (inp, in_edge), (out, out_edge) in zip(
                    sorted(in_edges.items()), sorted(out_edges.items())
                ):
                    assert inp == out
                    ngraph.add_edge(
                        in_edge.src,
                        in_edge.src_conn,
                        out_edge.dst,
                        out_edge.dst_conn,
                        out_edge.data,
                    )
            elif isinstance(node, nodes.MapExit):
                remove.add(node)
                in_edges = {e.data.data: e for e in ngraph.in_edges(node)}
                out_edges = {e.data.data: e for e in ngraph.out_edges(node)}
                for (inp, in_edge), (out, out_edge) in zip(
                    sorted(in_edges.items()), sorted(out_edges.items())
                ):
                    assert inp == out
                    ngraph.add_edge(
                        in_edge.src,
                        in_edge.src_conn,
                        out_edge.dst,
                        out_edge.dst_conn,
                        in_edge.data,
                    )
        ngraph.remove_nodes_from(remove)
        for edge in ngraph.edges():
            assert edge.data.subset.ranges[:2] == column_subset
            edge.data.subset.ranges[:2] = [(0, 0, 1), (0, 0, 1)]
        nsdfg_node = graph.add_nested_sdfg(nsdfg, sdfg, sources, sinks)

        def fix_sdfg_list_recursive(sdfg):
            sdfg._sdfg_list = list(sdfg.all_sdfgs_recursive())
            for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, nodes.NestedSDFG):
                        fix_sdfg_list_recursive(node.sdfg)
            sdfg.update_sdfg_list([])

        fix_sdfg_list_recursive(sdfg)
        nsdfg.validate()

        graph.remove_nodes_from(n for n in graph.nodes() if n is not nsdfg_node)

        graph.add_node(new_entry)
        graph.add_node(new_exit)

        for inp in nsdfg_node.in_connectors:
            access = nodes.AccessNode(inp, dace.AccessType.ReadOnly)
            graph.add_memlet_path(
                access,
                new_entry,
                nsdfg_node,
                memlet=dace.Memlet(data=inp, subset="i,j,0:K"),
                dst_conn=inp,
            )
        for out in nsdfg_node.out_connectors:
            access = nodes.AccessNode(out, dace.AccessType.WriteOnly)
            graph.add_memlet_path(
                nsdfg_node,
                new_exit,
                access,
                memlet=dace.Memlet(data=out, subset="i,j,0:K"),
                src_conn=out,
            )
        sdfg.validate()

        return sdfg


@registry.autoregister_params(singlestate=True)
@make_properties
class RefineMappedAccess(Transformation):
    """ 
    If a data descriptor is accessed only within the context of a single map
    and the accesses are unique per map iteration, reduces the dimensionality
    of the data descriptor and locates it internally to the map.
    """

    make_register = Property(
        dtype=bool, default=True,
        desc='Sets the storage type of reduced data to Register')

    entry = PatternNode(nodes.MapEntry)


    @staticmethod
    def expressions():
        return [dace.sdfg.utils.node_path_graph(RefineMappedAccess.entry)]

    @staticmethod
    def _candidates(state: SDFGState, me: nodes.MapEntry) -> Set[str]:
        sdfg = state.parent
        mx = state.exit_node(me)
        # Get candidates from map inputs/outputs
        candidates = set()
        for e in state.in_edges(me):
            if isinstance(e.src, nodes.AccessNode) and state.in_degree(e.src) == 0:
                if all(ee.dst is me for ee in state.out_edges(e.src)):
                    candidates.add(e.src)
        for e in state.out_edges(mx):
            if isinstance(e.dst, nodes.AccessNode) and state.out_degree(e.dst) == 0:
                if all(ee.src is mx for ee in state.in_edges(e.dst)):
                    candidates.add(e.dst)

        candidates = set(c for c in candidates if sdfg.arrays[c.data].transient)

        subgraph = state.scope_subgraph(me)

        # Check that all other instances of the access nodes only appear inside
        # the map
        names = set(n.data for n in candidates)
        for dnode in state.data_nodes():
            if dnode in candidates: continue
            if dnode.data not in names: continue
            if dnode not in subgraph.nodes():
                names.remove(dnode.data)
                continue
        # Check for uses in other states
        for other_state in sdfg.nodes():
            if other_state is state: continue
            for dnode in other_state.data_nodes():
                if dnode.data in names:
                    names.remove(dnode.data)
                    continue

        # (memlets) Mapping between data and indices of certain map indices
        accesses: Dict[str, Dict[str, int]] = {}

        # Check internal memlets for accesses that can be refined (i.e.,
        # include map parameters)
        map_params = me.map.params
        for e in subgraph.edges():
            if e.data.data not in names: continue
            syms: Dict[str, int] = {}
            for i, (rb, re, rs) in enumerate(e.data.subset):
                # If this is not a map parameter index, it can be anything
                if (len(set(map(str, rb.free_symbols)) & set(map_params)) == 0 and
                        len(set(map(str, re.free_symbols)) & set(map_params)) == 0):
                    continue

                r = str(rb)

                # Map parameter index must be exact (i, j)
                if r not in map_params:
                    break
                # Index cannot appear more than once
                if r in syms:
                    break
                # Index cannot be a range
                if re != rb or rs != 1:
                    break
                syms[r] = i

                # Check that all memlets access the map indices the same way
                if e.data.data in accesses:
                    if accesses[e.data.data][r] != i:
                        break
            else:
                # If not all map indices are involved, bad access
                if len(syms) == len(map_params):
                    # No break occurred - well-defined access
                    accesses[e.data.data] = syms
                    continue

            # Some break occured - bad access
            names.remove(e.data.data)

        # Filter out removed names
        candidates = set(cand for cand in candidates if cand.data in names)
        return candidates, accesses

    @staticmethod
    def can_be_applied(graph: SDFGState,
                       candidate: Dict[PatternNode, int],
                       expr_index: int,
                       sdfg: SDFG,
                       strict: bool = False) -> bool:

        me = graph.node(candidate[RefineMappedAccess.entry])
        candidates, _ = RefineMappedAccess._candidates(graph, me)
        if len(candidates) > 0:
            return True

        return False

    def apply(self, sdfg: SDFG):
        state = sdfg.node(self.state_id)
        me = self.entry(sdfg)
        mx = state.exit_node(me)

        to_remove, accesses = RefineMappedAccess._candidates(state, me)

        ######################################
        # Modify graph structure
        src_removals = [n for n in to_remove if any(e.src is n
                        for e in state.in_edges(me))]
        sink_removals = [n for n in to_remove if any(e.dst is n
                         for e in state.out_edges(mx))]
        # Source nodes
        for node in src_removals:
            # Remove outer edges
            for e in list(state.out_edges(node)):
                state.remove_edge_and_connectors(e)
            # Reconnect node to map from the inside
            state.add_nedge(me, node, dace.Memlet())
            # Redirect inner edges
            for e in list(state.out_edges(me)):
                if e.data.data == node.data:
                    state.remove_edge(e)
                    me.remove_out_connector(e.src_conn)
                    state.add_edge(node, None, e.dst, e.dst_conn, e.data)
        # Sink nodes
        for node in sink_removals:
            # Remove outer edges
            for e in list(state.in_edges(node)):
                state.remove_edge_and_connectors(e)
            # Reconnect node to map from the inside
            state.add_nedge(node, mx, dace.Memlet())
            # Redirect inner edges
            for e in list(state.in_edges(mx)):
                if e.data.data == node.data:
                    state.remove_edge(e)
                    mx.remove_in_connector(e.dst_conn)
                    state.add_edge(e.src, e.src_conn, node, None, e.data)

        ######################################
        # Modify data descriptors
        names = set(n.data for n in to_remove)
        for data in names:
            desc = sdfg.arrays[data]
            # Reduce shape in the right indices
            new_shape = list(desc.shape)
            for ind in accesses[data].values():
                new_shape[ind] = 1
            desc.shape = new_shape
            # Reset strides and total size
            desc.strides = [
                dace.data._prod(new_shape[i + 1:])
                for i in range(len(new_shape))
            ]
            desc.total_size = dace.data._prod(desc.shape)
            # Change storage type (if set)
            if self.make_register:
                desc.storage = dace.StorageType.Register

        ######################################
        # Modify edges
        subgraph = state.scope_subgraph(me)
        # Construct subsets to offset by
        offsets = {}
        for data in names:
            rng = [(0, 0, 1)] * len(sdfg.arrays[data].shape)
            for param, ind in accesses[data].items():
                sym = symbolic.pystr_to_symbolic(param)
                rng[ind] = (sym, sym, 1)
            offsets[data] = subsets.Range(rng)

        for e in subgraph.edges():
            if e.data.data in offsets:
                e.data.subset.offset(offsets[e.data.data], True)
