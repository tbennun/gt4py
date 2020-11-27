import dace

from gt4py import backend as gt_backend

from .base_backend import (
    CudaDaceOptimizer,
    DaceBackend,
    DacePyModuleGenerator,
    dace_is_compatible_layout,
    dace_is_compatible_type,
    dace_layout,
)


dace.SDFG


class GPUDacePyModuleGenerator(DacePyModuleGenerator):
    @property
    def array_interface_name(self):
        return "__cuda_array_interface__"


class GPUDaceOptimizer(CudaDaceOptimizer):

    description = ""

    def transform_library(self, sdfg):
        from dace.transformation.interstate import StateFusion
        from gt4py.backend.dace.sdfg.transforms import PruneTransientOutputs

        # sdfg.apply_transformations_repeated([PruneTransientOutputs, StateFusion], validate=False, strict=True)
        return sdfg

    def transform_optimize(self, sdfg):
        # import dace
        from dace.transformation.dataflow import MapCollapse
        from dace.transformation.interstate import StateFusion, RefineNestedAccess

        from gt4py.backend.dace.sdfg.transforms import (
            OnTheFlyMapFusion,
            LoopBufferCache,
            IJMapFusion,
        )

        sdfg.apply_transformations_repeated(MapCollapse, validate=False)
        # sdfg.apply_transformations_repeated(IJMapFusion, validate=False)
        sdfg.apply_transformations_repeated(RefineNestedAccess, validate=False, strict=True)
        sdfg.apply_transformations_repeated(OnTheFlyMapFusion, validate=False)
        sdfg.apply_transformations_repeated(LoopBufferCache, validate=False)
        # sdfg.apply_strict_transformations(validate=False)

        for name, array in sdfg.arrays.items():
            if array.transient:
                array.lifetime = dace.dtypes.AllocationLifetime.Persistent



        # sdfg.apply_transformations_repeated(LoopBufferCache, validate=False)

        from dace.sdfg.graph import SubgraphView
        from dace.transformation.subgraph.subgraph_fusion import SubgraphFusion

        for graph in sdfg.nodes():
            subgraph = SubgraphView(
                graph, [node for node in graph.nodes() if graph.out_degree(node) > 0]
            )
            if SubgraphFusion.can_be_applied(sdfg, subgraph):
                fusion = SubgraphFusion(subgraph)
                fusion.transient_allocation = dace.dtypes.StorageType.Register
                fusion.apply(sdfg)
                for name, array in sdfg.arrays.items():
                    if array.transient:
                        if array.storage == dace.dtypes.StorageType.GPU_Global:
                            array.lifetime = dace.dtypes.AllocationLifetime.Persistent

                        for node in graph.nodes():
                            if isinstance(node, dace.nodes.NestedSDFG):
                                for inner_name, inner_array in node.sdfg.arrays.items():
                                    if inner_name == name:
                                        inner_array.storage = array.storage
                                        inner_array.strides = array.strides

        dace.sdfg.utils.consolidate_edges(sdfg)

        for sd, name, array in sdfg.arrays_recursive():
            if (array.lifetime is dace.AllocationLifetime.Persistent
                    and array.storage is dace.StorageType.Register):
                array.lifetime = dace.AllocationLifetime.Scope

        return sdfg


@gt_backend.register
class GPUDaceBackend(DaceBackend):
    name = "dacecuda"
    storage_info = {
        "alignment": 1,
        "device": "gpu",
        "layout_map": dace_layout,
        "is_compatible_layout": dace_is_compatible_layout,
        "is_compatible_type": dace_is_compatible_type,
    }
    MODULE_GENERATOR_CLASS = GPUDacePyModuleGenerator
    DEFAULT_OPTIMIZER = GPUDaceOptimizer()
