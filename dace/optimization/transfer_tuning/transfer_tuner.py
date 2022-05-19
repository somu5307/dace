import math
import json

from typing import Dict, Any, List, Union
from pathlib import Path

from pydoc import locate

from dace import SDFG, InstrumentationType, DataInstrumentationType, nodes
from dace.optimization.transfer_tuning.transfer_space import TransferSpace
from dace.optimization.transfer_tuning.utils import measure

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x

class TransferTuner():
    """
    
    """

    def __init__(self, config: Union[str, Path]) -> None:
        with open(config, "r") as handle:
            self._config = json.load(handle)

            self._name = self._config["name"]
            self._stages = []
            for stage_desc in self._config["stages"]:
                stage = []
                for space in stage_desc:
                    space_class = locate(f'dace.optimization.transfer_tuning.spaces.{space}')
                    stage.append(space_class())
                
                self._stages.append(stage)

    def _write_cache(self, stage: List[TransferSpace], search_cache: Dict) -> None:
        pass

    def _load_cache(self, path: Union[str, Path]) -> Dict:
        return {}

    def _search(self, stage: List[TransferSpace], sdfg: SDFG, dreport, instrumentation_type: InstrumentationType, search_cache: Dict, save_cache: bool = True) -> Dict:
        """
        Searches the space of the stage brute-force.
        The best config of each cutout is directly applied to the SDFG.

        :param stage: the stage.
        :param sdfg: the sdfg.
        :param instrumentation_type: the instrumentation type defines the metric to compare configurations.
        :param search_cache: a (partial) search cache from previous runs.
        :param save_cache: whether to write the search cache to the dacecache folder of the SDFG.
        :return: returns the updated search cache.
        """
        current_space = stage[0]
        for i, cutout in tqdm(list(enumerate(current_space.cutouts(sdfg)))):
            if i not in search_cache:
                # Cutout not in cache, measure baseline
                cutout.instrument = instrumentation_type
                search_cache[i] = {}
                search_cache[i]["base_runtime"] = measure(cutout, dreport)
            
            base_runtime = search_cache[i]["base_runtime"]
            if base_runtime < 0 or base_runtime == math.inf:
                continue

            print(f"New cutout with: {base_runtime:.2f} ms")

            # Iterate through config space and measure configs if necessary
            best_runtime = base_runtime
            best_config = None
            for config in tqdm(list(current_space.configurations(cutout))):
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
                search_cache[i][key] = {}
                search_cache[i][key]["runtime"] = math.inf
                search_cache[i][key]["subspace"] = {}

                cutout_ = current_space.apply_config(cutout, config, make_copy=True)
                cutout_.instrument = instrumentation_type
                if len(stage) > 1:
                    subspace_cache = search_cache[i][config]["subspace"]
                    
                    self.run_stage(stage[:1], cutout_, dreport, instrumentation_type, search_cache=subspace_cache, save_cache=False)
                    
                    search_cache[i][key]["subspace"] = subspace_cache

                runtime = measure(cutout_, dreport)
                search_cache[i][key]["runtime"] = runtime
                
                if runtime < 0 or runtime == math.inf:
                    continue

                print(f"{key} with {runtime:.2f} ms")

                if runtime < best_runtime:
                    best_runtime = runtime
                    best_config = key

            # Store best config
            search_cache[i]["best_config"] = best_config
            search_cache[i]["best_runtime"] = best_runtime

        if save_cache:
            self._write_cache(stage, search_cache)

        # Update SDFG with best configs
        self._apply_best_configs(stage, sdfg, search_cache)

        print(search_cache)
        return search_cache

    def _apply_best_configs(self, stage: List[TransferSpace], sdfg: SDFG, search_cache: Dict, context: List[SDFG] = None):
        if context is None:
            context = [sdfg]
        
        current_space = stage[0]
        current_sdfg = context[-1]
        # Apply best config to each cutout
        for i, cutout in enumerate(current_space.cutouts(current_sdfg)):
            key = search_cache[i]["best_config"]
            if key is None:
                continue

            config = search_cache[i][key]

            # Translate config back to SDFG and apply
            current_context = cutout
            for i in range(len(context)):
                parent_context = context[-(i + 1)]

                config = current_space.translate_config(current_context, parent_context, config)
                current_space.apply_config(parent_context, config)

                current_context = parent_context

            # If nested configs, translate configs over cutout back to SDFG recursively and apply
            # cutout needs to have the configs applied as well to remain equivalent to SDFG
            if len(stage) > 1:
                nested_stages = stage.copy()
                nested_stages.pop(0)

                nested_search_cache = search_cache[i][key]["subspace"]

                nested_context = context.copy()
                nested_context.append(cutout)

                self._apply_best_configs(nested_stages, sdfg, nested_search_cache, nested_context)

    def tune(self, sdfg: SDFG, dreport, instrumentation_type: InstrumentationType = InstrumentationType.Timer, search_cache_path: Union[str, Path] = None) -> None:
        """
        Auto-tunes the SDFG with stages defined in the tuning config.
        Stages run sequentially and each stage tunes multiple cutouts of the SDFG.

        :param sdfg: the SDFG.
        :param dreport: the data report (obtained from dry_run function).
        :param instrumentation_type: the instrumentation type defines the metric to compare configurations.
        :param search_cache_path: path to a search cache obtained from previous runs.
        """
        search_cache = {}
        if search_cache is not None:
            search_cache = self._load_cache(search_cache_path)

        for stage in self._stages:
            self._search(stage, sdfg, dreport, instrumentation_type, search_cache=search_cache, save_cache=True)

    @staticmethod
    def dry_run(sdfg: SDFG, *args, **kwargs) -> Any:
        """
        A dry run executes the SDFG with data instrumentation and stores the data of
        all non-transient arrays in the dacecache. The method checks whether a valid data
        report is already available.
        
        The data report is also returned at the end of the function.

        :param sdfg: the SDFG.
        :param args: args to the SDFG.
        :param kwargs: kwargs to the SDFG.
        :return: the data report
        """
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
