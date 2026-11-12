"""Consumer sidecar: select only the second TP2 blocks call in each process."""
import atexit
import ctypes
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def install():
    if os.environ.get("H3_AB_ROI") != "1":
        return
    target = "minimax_h3_amd_sp.parallel_blocks"

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target_module=None):
            if fullname != target:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            delegate = spec.loader

            class Loader(importlib.abc.Loader):
                def create_module(self, module_spec):
                    return delegate.create_module(module_spec)

                def exec_module(self, module):
                    delegate.exec_module(module)
                    original = module.run_blocks
                    lib = ctypes.CDLL("/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib/librocprofiler-sdk-roctx.so.1")
                    for name in ("roctxProfilerResume", "roctxProfilerPause"):
                        function = getattr(lib, name)
                        function.argtypes = [ctypes.c_uint32]
                        function.restype = ctypes.c_int
                    lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
                    lib.roctxRangePop.argtypes = []
                    rows = []
                    count = 0
                    process_rank = None

                    def run_blocks(*args, **kwargs):
                        nonlocal count, process_rank
                        count += 1
                        rank = kwargs.get("rank", args[6] if len(args) > 6 else None)
                        process_rank = rank
                        selected = count == 2
                        if rank == 1 and selected:
                            lib.roctxProfilerResume(0)
                        name = f"h3/rank{rank}/blocks/request{count}"
                        start = time.perf_counter_ns()
                        lib.roctxRangePushA(name.encode())
                        try:
                            return original(*args, **kwargs)
                        finally:
                            if rank == 1 and selected:
                                # Finish this rank's submitted kernels before pausing.
                                import torch
                                torch.cuda.synchronize()
                            lib.roctxRangePop()
                            stop = time.perf_counter_ns()
                            if rank == 1 and selected:
                                lib.roctxProfilerPause(0)
                            rows.append({"name": name, "rank": rank, "pid": os.getpid(),
                                         "selected": selected, "start_ns": start,
                                         "duration_ns": stop - start,
                                         "time_basis": "cpu_submission"})

                    module.run_blocks = run_blocks

                    def save():
                        if process_rank is not None:
                            root = Path(os.environ["RDNASCOPE_RUN_DIR"])
                            (root / f"roi-rank{process_rank}.json").write_text(
                                json.dumps({"pid": os.getpid(), "rank": process_rank,
                                            "calls": count, "scopes": rows}, indent=2) + "\n")
                    atexit.register(save)

            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, Finder())
