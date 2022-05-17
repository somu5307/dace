import ast
import copy
import itertools
import numpy as np
import random
import string

from typing import Any, Generator, Tuple, List

from dace import SDFG, nodes
from dace.transformation import helpers as xfh
from dace.sdfg.analysis import cutout as cutter
from dace.optimization.transfer.transfer_space import TransferSpace

class MapPermutationSpace(TransferSpace):

    def name(self) -> str:
        return 'MapPermutationSpace'

    def apply_on_cutout(self, cutout: SDFG, config: Any, make_copy: bool = True) -> SDFG:
        if make_copy:
            cutout_ = copy.deepcopy(cutout)
        else:
            cutout_ = cutout

        map_entry = MapPermutationSpace.top_map(cutout_)        
        new_order = [map_entry.map.params[i] for i in config]
        map_entry.range.ranges = [
            r for list_param in new_order for map_param, r in zip(map_entry.map.params, map_entry.range.ranges)
            if list_param == map_param
        ]
        map_entry.map.params = new_order

        return cutout_

    def apply_on_target(self, sdfg: SDFG, cutout: SDFG, config: Any) -> None:
        state_id = int(cutout.name.split("_")[-1])
        state = sdfg.node(state_id)
        
        cmap_entry = MapPermutationSpace.top_map(cutout)
        node_id = state.node_id(cmap_entry)

        map_entry = state.node(node_id)
        
        new_order = [map_entry.map.params[i] for i in config]
        map_entry.range.ranges = [
            r for list_param in new_order for map_param, r in zip(map_entry.map.params, map_entry.range.ranges)
            if list_param == map_param
        ]
        map_entry.map.params = new_order

    def encode_config(self, config: Any) -> str:
        return str(config)

    def decode_config(self, config: str) -> Any:
        return ast.literal_eval(config)

    def extract_patterns(self, sdfg: SDFG, cutout: SDFG, config: str) -> Tuple[Tuple[Any, str]]:
        state_id = int(cutout.name.split("_")[-1])
        state = sdfg.node(state_id)
        
        map_entry = MapPermutationSpace.top_map(cutout)
        node_id = state.node_id(map_entry)

        letters = string.ascii_lowercase
        embedding = ''.join(random.choice(letters) for i in range(10))
        pattern = {
            # "embedding": np.zeros((2,), dtype=np.float32),
            "embedding": embedding,
            "config": config,
            "sdfg": sdfg.hash_sdfg(),
            "target": [(state_id, node_id)]
        }

        return (
            pattern,
        )

    def cutouts(self, sdfg: SDFG) -> Generator[SDFG, None, None]:
        for state in sdfg.nodes():
            for node in state.nodes():
                if not isinstance(node, nodes.MapEntry) or xfh.get_parent_map(state, node) is not None:
                    continue
            
                state_id = sdfg.node_id(state)
                subgraph_nodes = state.scope_subgraph(node).nodes()
                cutout = cutter.cutout_state(state, *subgraph_nodes, make_copy=False)
                yield cutout

    def configurations(self, cutout: SDFG) -> Generator[Any, None, None]:
        map_entry = MapPermutationSpace.top_map(cutout)

        positions = tuple(range(len(map_entry.map.params)))
        for config in itertools.permutations(positions):
            if config == positions:
                continue

            yield config

    @staticmethod
    def top_map(cutout: SDFG):
        map_entry = None
        for node in cutout.start_state.nodes():
            if not isinstance(node, nodes.MapEntry) or xfh.get_parent_map(cutout.start_state, node) is not None:
                continue

            map_entry = node
            break
    
        return map_entry
