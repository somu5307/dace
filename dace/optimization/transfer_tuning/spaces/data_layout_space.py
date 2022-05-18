import copy
import ast
import itertools

from typing import Any, Generator, Tuple

from dace import SDFG, symbolic
from dace.optimization.transfer_tuning.transfer_space import TransferSpace

class DataLayoutSpace(TransferSpace):

    def name(self) -> str:
        return 'DataLayoutSpace'

    def apply_config(self, cutout: SDFG, config: Any, make_copy: bool = True) -> SDFG:
        if make_copy:
            cutout_ = copy.deepcopy(cutout)
        else:
            cutout_ = cutout

        cutout_._arrays = config
        return cutout_

    def translate_config(self, cutout: SDFG, sdfg: SDFG, config: Any) -> Any:
        return config

    def encode_config(self, config: Any) -> str:
        dict_str = ','.join([f'"{k}": "{v.strides}"' for k, v in config.items() if not v.transient])
        dict_str = '{' + dict_str + '}'
        return dict_str

    def decode_config(self, config: str) -> Any:
        return ast.literal_eval(config)

    def extract_patterns(self, sdfg: SDFG, cutout: SDFG, config: Any) -> Tuple[Tuple[Any, str]]:
        patterns = []
        for array in config:
            desc = ""
            data_layout = ""

            patterns.append((desc, data_layout))
        
        return patterns

    def cutouts(self, sdfg: SDFG) -> Generator[SDFG, None, None]:
        yield sdfg

    def configurations(self, cutout: SDFG) -> Generator[Any, None, None]:
        groups = self.group_arrays(cutout)

        group_configs = [itertools.permutations(list(range(dims))) for (_, dims), _ in groups]
        global_configs = itertools.product(*group_configs)

        for config in global_configs:
            new_arrays = copy.deepcopy(cutout.arrays)

            for i in range(len(groups)):
                group_config = config[i]
                _, group = groups[i]

                for member in group:
                    desc = new_arrays[member]
                    strides, total_size = desc.strides_from_layout(*group_config)
                    new_arrays[member].strides = strides
                    new_arrays[member].total_size = total_size

            yield new_arrays

    def group_arrays(self, cutout: SDFG):
        groups = {}

        visited = set()
        for state in cutout.nodes():
            for dnode in state.data_nodes():
                if cutout.arrays[dnode.data].transient or dnode.data in visited:
                    continue

                dims = len(dnode.desc(cutout).shape)
                dims = symbolic.evaluate(dims, cutout.constants)
                if state.in_degree(dnode) == 0:
                    type = "input"
                elif state.out_degree(dnode) == 0:
                    type = "output"
                else:
                    type = dnode.data
                
                group = (type, dims)
                if group not in groups:
                    groups[group] = []
                
                groups[group].append(dnode.data)
        
        return list(groups.items())
