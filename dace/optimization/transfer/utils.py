import math
import numpy as np

import traceback
import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method("spawn")

from dace import SDFG
from dace import data as dt

def measure(cutout: SDFG, dreport, repeat: int = 30, timeout: float = 300.0) -> float:    
    queue = mp.Queue()
    proc = MeasureProcess(target=_measure, args=(cutout.to_json(), dreport, repeat, queue))
    proc.start()
    proc.join(timeout)

    if proc.exitcode != 0:
        print("Error occured during measuring")
        return math.inf

    if proc.exception:
        error, traceback = proc.exception
        print(traceback)
        print("Error occured during measuring: ", error)
        runtime = math.inf

    try:
        runtime = queue.get(block=True, timeout=10)
    except:
        return math.inf

    return runtime

def _measure(cutout_json, dreport, repetitions: int, queue: mp.Queue) -> float:
    cutout = SDFG.from_json(cutout_json)
    
    arguments = {}
    if len(cutout.free_symbols) > 0:
        raise ValueError("Free symbols found")

    for state in cutout.nodes():
        for dnode in state.data_nodes():
            array = cutout.arrays[dnode.data]
            if array.transient:
               continue

            try:
                data = dreport[dnode.data]
                arguments[dnode.data] = dt.make_array_from_descriptor(array, data, constants=cutout.constants)
            except KeyError:
                arguments[dnode.data] = dt.make_array_from_descriptor(array, constants=cutout.constants)


    for name, array in list(cutout.arrays.items()):
        if array.transient:
            continue

        if not name in arguments:
            del cutout.arrays[name]


    import dace
    with dace.config.set_temporary('debugprint', value=False):
        with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
            with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
                csdfg = cutout.compile()
                for _ in range(repetitions):
                    csdfg(**arguments)

                csdfg.finalize()

    # TODO: Validate

    report = cutout.get_latest_report()
    durations = next(iter(next(iter(report.durations.values())).values()))
    queue.put(np.median(np.array(durations)))

class MeasureProcess(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

