import atexit
import logging
import os
import subprocess
import sys
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist

from .h3_forward import run_rank_zero
from .cache_dit import make_cache


class DualGpuRuntime:
    def __init__(self, model_path, weight_dtype, devices, port):
        if len(devices) != 2:
            raise ValueError("exactly two devices are required")
        self.model_path = model_path
        self.weight_dtype = weight_dtype
        self.devices = tuple(devices)
        self.port = port
        self.process = None
        self.object_group = None
        self.last_timestep = None
        self.cache_dit = make_cache()
        self.log_path = os.path.join(
            os.environ.get("AMD_H3_SP_LOGDIR", tempfile.gettempdir()),
            "amd_h3_sp_worker_1.log",
        )
        self._start()
        atexit.register(self.shutdown)

    def _start(self):
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        comfy_root = os.path.dirname(os.path.abspath(__import__("folder_paths").__file__))
        worker = os.path.join(package_root, "minimax_h3_amd_sp", "worker.py")
        environment = os.environ.copy()
        if torch.version.hip is not None:
            environment.pop("CUDA_VISIBLE_DEVICES", None)
            environment.pop("HIP_VISIBLE_DEVICES", None)
            environment.pop("ROCR_VISIBLE_DEVICES", None)
            environment["ROCR_VISIBLE_DEVICES"] = self.devices[1]
        else:
            environment.pop("HIP_VISIBLE_DEVICES", None)
            environment.pop("ROCR_VISIBLE_DEVICES", None)
            environment["CUDA_VISIBLE_DEVICES"] = self.devices[1]
        environment["PYTHONPATH"] = (
            comfy_root
            + os.pathsep
            + package_root
            + os.pathsep
            + environment.get("PYTHONPATH", "")
        )

        log = open(self.log_path, "w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                sys.executable,
                worker,
                "--rank",
                "1",
                "--world-size",
                "2",
                "--port",
                str(self.port),
                "--model-path",
                self.model_path,
                "--weight-dtype",
                self.weight_dtype,
                "--comfy-root",
                comfy_root,
            ],
            cwd=comfy_root,
            env=environment,
            stdout=log,
            stderr=log,
        )

        if not dist.is_initialized():
            dist.init_process_group(
                "nccl",
                init_method=f"tcp://127.0.0.1:{self.port}",
                rank=0,
                world_size=2,
                timeout=timedelta(minutes=40),
                device_id=torch.device("cuda", 0),
            )
        self.object_group = dist.new_group(
            backend="gloo",
            timeout=timedelta(minutes=40),
        )
        dist.barrier()
        from .cache_dit import CacheDiTConfig
        cache_identity = self.cache_dit.identity() if self.cache_dit else CacheDiTConfig().identity()
        identities = [None, None]
        dist.all_gather_object(identities, cache_identity, group=self.object_group)
        if identities[0] != identities[1]:
            raise RuntimeError("Rank Cache-DiT configuration/dependency mismatch")
        self._check_worker()
        logging.info(
            "[amd-h3-sp] two-rank runtime ready on %s",
            self.devices,
        )

    def _check_worker(self):
        if self.process is None:
            self._start()
            return
        return_code = self.process.poll()
        if return_code is not None:
            logging.warning(
                "[amd-h3-sp] worker exited with code %d; restarting it",
                return_code,
            )
            self.shutdown()
            self._start()

    def forward(
        self,
        dit,
        x,
        timestep,
        context,
        transformer_options,
        minimax_payload,
    ):
        self._check_worker()
        current_timestep = float(timestep.flatten()[0])
        reset_cache = (
            self.last_timestep is None
            or current_timestep >= self.last_timestep
        )
        self.last_timestep = current_timestep
        try:
            return run_rank_zero(
                dit,
                x,
                timestep,
                context,
                transformer_options,
                minimax_payload,
                self.object_group,
                reset_cache=reset_cache,
                profile=os.environ.get("AMD_H3_SP_PROFILE") == "1",
                cache_dit=self.cache_dit,
            )
        except Exception:
            logging.exception("[amd-h3-sp] distributed forward failed")
            # A failed peer cannot participate in the shutdown broadcast.
            if self.process is not None:
                self.process.kill()
                self.process.wait(timeout=15)
            self.process = None
            self.object_group = None
            raise

    def finish_cache_request(self):
        if self.cache_dit is None:
            return
        logging.info("[amd-h3-sp] Cache-DiT request: %s", self.cache_dit.summary())
        self.cache_dit.release()
        self.last_timestep = None
        if self.process is not None and self.process.poll() is None and self.object_group is not None:
            dist.broadcast_object_list([{"op": "cache_release"}], src=0, group=self.object_group)

    def shutdown(self):
        if self.cache_dit is not None:
            self.cache_dit.release()
        if self.process is None:
            return
        try:
            if self.process.poll() is None and self.object_group is not None:
                dist.broadcast_object_list(
                    [{"op": "shutdown"}],
                    src=0,
                    group=self.object_group,
                )
                self.process.wait(timeout=15)
        except Exception:
            if self.process.poll() is None:
                self.process.kill()
        finally:
            self.process = None
            self.object_group = None
            self.last_timestep = None
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
