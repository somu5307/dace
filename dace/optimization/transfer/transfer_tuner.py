import math
import json

from typing import Dict, Any, List, Sequence, Union
from pathlib import Path

from pydoc import locate

from dace import SDFG, InstrumentationType, DataInstrumentationType, nodes
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

    def optimize(self, sdfg: SDFG, dreport, instrumentation_type: InstrumentationType = InstrumentationType.Timer) -> Dict[str, Dict[Any, str]]:
        for stage in self._stages:
            current_space = stage.pop(0)
            for cutout in tqdm(list(current_space.cutouts(sdfg))):
                cutout.instrument = instrumentation_type
                base_runtime = measure(cutout, dreport)

                print(f"New cutout with base runtime: {base_runtime:.2f}")

                best_config = None
                best_runtime = base_runtime
                for config in tqdm(list(current_space.configurations(cutout))):
                    cutout_ = current_space.apply_on_cutout(cutout, config, make_copy=True)
                    cutout_.instrument = instrumentation_type

                    runtime = measure(cutout_, dreport)
                    if runtime < 0 or runtime == math.inf:
                        continue

                    print(f"Tested {config} with {base_runtime - runtime:.2f} delta")

                    if runtime >= best_runtime:
                        continue
            
                    best_runtime = runtime
                    best_config = config
                
                if best_config is None:
                    continue

                patterns = current_space.extract_patterns(sdfg, cutout=cutout, config=best_config)

                print(f"Applying {best_config} with {base_runtime - best_runtime:.2f} delta")
                current_space.apply_on_target(sdfg, cutout, best_config)

    def transfer(self, sdfg: SDFG, database: Dict[str, Dict[Any, str]], strict: bool = False) -> None:
        for stage in self._stages:
            stage_database = database[stage.name()]
            stage.transfer(sdfg, database=stage_database, strict=strict)

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
