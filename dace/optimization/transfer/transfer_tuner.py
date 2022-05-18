import math
import json

from typing import Dict, Any, List, Sequence, Union
from pathlib import Path

from pydoc import locate

from dace import SDFG, InstrumentationType, DataInstrumentationType, nodes
from dace.optimization.transfer.transfer_space import TransferSpace
from dace.optimization.transfer.utils import measure

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x

class TransferTuner():

    def __init__(self, config: Union[str, Path]) -> None:
        with open(config, "r") as handle:
            self._config = json.load(handle)

            self._name = self._config["name"]
            self._stages = []
            for stage_desc in self._config["stages"]:
                stage = []
                for space in stage_desc:
                    space_class = locate(f'dace.optimization.transfer.spaces.{space}')
                    stage.append(space_class())
                
                self._stages.append(stage)

    def _search(self, stage: List[TransferSpace], sdfg: SDFG, dreport, instrumentation_type: InstrumentationType, search_cache: Dict, save_results: bool = True):        
        current_space = stage[0]
        for i, cutout in enumerate(current_space.cutout(sdfg)):
            if i not in search_cache:
                # Cutout not in cache, measure baseline
                cutout.instrument = instrumentation_type
                search_cache[i] = {}
                search_cache[i]["base_runtime"] = measure(cutout, dreport)
            
            base_runtime = search_cache[i]["base_runtime"]
            if base_runtime < 0 or base_runtime == math.inf:
                continue

            # Iterate through config space and measure configs if necessary
            best_runtime = base_runtime
            best_config = None
            new_configs = []
            for config in current_space.configurations(cutout):
                key = current_space.encode_config(config)
                if key in search_cache[i]:
                    # a) Results from cache
                    runtime = search_cache[i][key]["runtime"]
                    if runtime < 0 or runtime == math.inf:
                        continue

                    if runtime < best_runtime:
                        best_runtime = runtime
                        best_config = key
                    
                    continue

                # b) Measure config
                search_cache[i][key]["runtime"] = math.inf
                search_cache[i][key]["subspace"] = {}

                cutout_ = current_space.apply_on_cutout(cutout, config, make_copy=True)
                cutout_.instrument = instrumentation_type
                if len(stage) > 1:
                    subspace_cache = search_cache[i][config]["subspace"]
                    
                    self.run_stage(stage[:1], cutout_, dreport, instrumentation_type, search_cache=subspace_cache, save_results=False)
                    
                    search_cache[i][key]["runtime"] = subspace_cache

                runtime = measure(cutout_, dreport)
                search_cache[i][key]["runtime"] = runtime
                new_configs.append(key)
                
                if runtime < 0 or runtime == math.inf:
                    continue

                if runtime < best_runtime:
                    best_runtime = runtime
                    best_config = key

            # Store best config
            search_cache[i]["best_config"] = best_config
            search_cache[i]["best_runtime"] = best_runtime

        if save_results:
            self._write_cache(stage, search_cache)

        self._apply_best_config(stage, sdfg, search_cache)

        return search_cache

    def _apply_best_config(self, stage: List[TransferSpace], sdfg: SDFG, search_cache: Dict):
        current_space = stage.pop(0)
        for i, cutout in enumerate(current_space.cutout(sdfg)):
            key = search_cache[i]["best_config"]
            config = search_cache[i][key]
            current_space.apply_on_target(sdfg, cutout, config)

            if len(stage) > 0:
                # Recursively apply on SDFG
                pass

    def _write_cache(self, stage: List[TransferSpace], search_cache: Dict):
        pass

    def _load_cache(self, path: Union[str, Path]):
        return {}

    def optimize(self, sdfg: SDFG, dreport, instrumentation_type: InstrumentationType = InstrumentationType.Timer, search_cache_path: Union[str, Path] = None) -> Dict[str, Dict[Any, str]]:
        search_cache = {}
        if search_cache is not None:
            search_cache = self._load_cache(search_cache_path)

        for stage in self._stages:
            self._search(stage, sdfg, dreport, instrumentation_type, search_cache=search_cache, save_results=True)

    @staticmethod
    def dry_run(sdfg: SDFG, *args, **kwargs) -> Any:
        # Check existing instrumented data for shape mismatch
        kwargs.update({aname: a for aname, a in zip(sdfg.arg_names, args)})

        dreport = sdfg.get_instrumented_data()
        if dreport is not None:
            for data in dreport.keys():
                rep_arr = dreport.get_first_version(data)
                sdfg_arr = sdfg.arrays[data]
                # Potential shape mismatch
                if rep_arr.shape != sdfg_arr.shape:
                    # Check given data first
                    if hasattr(kwargs[data], 'shape') and rep_arr.shape != kwargs[data].shape:
                        sdfg.clear_data_reports()
                        dreport = None
                        break

        # If there is no valid instrumented data available yet, run in data instrumentation mode
        if dreport is None:
            for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, nodes.AccessNode) and not node.desc(sdfg).transient:
                        node.instrument = DataInstrumentationType.Save

            result = sdfg(**kwargs)

            # Disable data instrumentation from now on
            for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, nodes.AccessNode):
                        node.instrument = DataInstrumentationType.No_Instrumentation

        return dreport
