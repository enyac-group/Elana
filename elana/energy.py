import os
import time
import logging
from typing import Optional
from multiprocessing import Process, Event, Manager
from datetime import datetime

# to profile gpu power
try:
    from pynvml import *
    _NVML_AVAILABLE = True
except Exception as e:
    _NVML_AVAILABLE = False

try:
    from jtop import jtop
    _JTOP_AVAILABLE = True
except Exception:
    _JTOP_AVAILABLE = False

import torch
import torch.nn as nn

logger = logging.getLogger(os.path.basename(__file__))


def get_visible_gpus():
    """
    Returns a list of physical GPU indices (as seen by nvidia-smi)
    that are visible to this process, honoring CUDA_VISIBLE_DEVICES.

    On Jetson (jtop) this will still return [0] if a GPU is present.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")

    if _NVML_AVAILABLE:
        # First, compute the visible list purely from CUDA / env
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible is not None:
            # e.g. CUDA_VISIBLE_DEVICES="4,5,6,7" -> [4, 5, 6, 7]
            visible_list = [int(v.strip()) for v in visible.split(",") if v.strip()]
        else:
            # No env restriction: all physical devices [0..N-1]
            visible_list = list(range(torch.cuda.device_count()))
    elif _JTOP_AVAILABLE:
        # Jetson: usually a single integrated GPU, index 0
        # visible_list from CUDA is already correct; this branch is only
        # here to preserve compatibility if you later want to add jtop-specific logic.
        if not visible_list:
            visible_list = [0]
    else:
        # No NVML, no jtop: we already checked torch.cuda.is_available()
        # so visible_list should still be valid; no extra work needed.
        pass

    return visible_list


"""
{
  "rail": { ... },   # per-rail sensor readings from the INA3221 chips, lists individual voltage rails that power different functional blocks of the SoC.
  "tot": { ... }     # total board (VDD_IN) power. Represents the total input power drawn by the Jetson module
}
"""
# Rails that often carry GPU power on Jetson
GPU_RAIL_CANDIDATES = [
    "VDD_GPU_SOC",  # Orin
    "VDD_SYS_GPU",  # TX2
    "VDD_GPU",      # Nano / Xavier
    "GPU",          # Generic alias in some tools
]

def find_nano_gpu_rail_name(rail_dict):
    """
    Given power_data['rail'] (a dict of {rail_name: {power, voltage, current}}),
    return the best-matching rail name for GPU power.
    On the Jetson Orin Nano:
        •	You cannot directly separate GPU-only power, because it shares the regulator with CPU and CV blocks (VDD_CPU_GPU_CV).
        •	The closest approximation to “GPU power” is VDD_CPU_GPU_CV, but it also rises when CPU workloads are active.
    """
    # 1) Exact/known names first
    for name in GPU_RAIL_CANDIDATES:
        if name in rail_dict:
            return name
    # 2) Fuzzy match: any rail name that contains 'GPU'
    for name in rail_dict: # we get "VDD_CPU_GPU_CV" here
        if "GPU" in name.upper():
            return name
    return None

def read_nano_gpu_power_once(power_data):
    """
    Returns GPU power in Watts if found, else None.
    Expects power_data from jetson.power (dict with keys 'rail' and 'tot').
    """
    rail = power_data.get("rail")
    if not isinstance(rail, dict):
        return None
    gpu_rail = find_nano_gpu_rail_name(rail)
    if not gpu_rail:
        return None

    entry = rail.get(gpu_rail, {})
    # jtop reports mW for 'power'
    mw = entry.get("power")
    if mw is None:
        return None
    return float(mw) / 1000.0  # W


def log_gpu_stats(
    gpu_index=0,
    log_interval=0.1,
    log_file="gpu_power_log.csv",
    stop_event: Optional[Event] = None,
    shared_power_list=None,
):
    """
    Continuously logs power, utilization, and temperature.
    Appends power readings (in Watts) to shared_power_list if provided.
    Stops when stop_event is set.
    """
    if _NVML_AVAILABLE:
        nvmlInit()
        try:
            device_count = nvmlDeviceGetCount()
            if gpu_index >= device_count:
                logger.error(f"Error: GPU index {gpu_index} out of range (0..{device_count-1})")
                return

            handle = nvmlDeviceGetHandleByIndex(gpu_index)

            with open(log_file, "w") as f:
                f.write("timestamp,gpu_index,power_w,utilization_pct,temperature_c\n")

            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0 # mW to W
                util  = nvmlDeviceGetUtilizationRates(handle).gpu
                temp  = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

                # write to CSV
                with open(log_file, "a") as f:
                    f.write(f"{ts},{gpu_index},{power:.2f},{util},{temp}\n")

                # record in shared list (if exists)
                if shared_power_list is not None:
                    shared_power_list.append(power)

                time.sleep(log_interval)
        finally:
            nvmlShutdown()
    
    # Jetson Nano
    elif _JTOP_AVAILABLE:
        try:
            with jtop() as jetson:
                # Show what rails are available
                rails = list(jetson.power.get("rail", {}).keys())
                logger.info("Connected to Jetson via jtop.")
                if rails:
                    logger.info(f"Available rails: {rails}")
                else:
                    logger.info(f"(none)")

                while jetson.ok():
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    pwr_w = read_nano_gpu_power_once(jetson.power)
                    if pwr_w is not None:
                        # logger.info(f"GPU Power: {pwr_w:.2f} W")
                        # write to CSV
                        with open(log_file, "a") as f:
                            f.write(f"{ts},{gpu_index},{pwr_w:.2f}\n")
                        # record in shared list (if exists)
                        if shared_power_list is not None:
                            shared_power_list.append(pwr_w)
                    else:
                        # Help debug: show rails once in a while
                        pd = jetson.power
                        logger.warning("GPU rail not found. Keys:",
                            {"rail": list(pd.get("rail", {}).keys()),
                            "tot": list(pd.get("tot", {}).keys()) if isinstance(pd.get("tot"), dict) else pd.get("tot")})
                    time.sleep(log_interval)
        finally:
            # process stop
            pass


def launch_energy_logger_process():
    gpu_index = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
    stop_event = Event()
    manager = Manager()

    gpu_power = manager.list()

    logger.info(f"[Rank {gpu_index}] Launching GPU energy logger on GPU {gpu_index}")

    proc = Process(
        target=log_gpu_stats,
        kwargs=dict(
            gpu_index=gpu_index,
            log_interval=0.1,
            log_file=f"gpu{gpu_index}_power_log.csv",
            stop_event=stop_event,
            shared_power_list=gpu_power,
        ),
    )
    proc.start()

    return stop_event, [gpu_power], [proc]


def stop_energy_logger_process(proc_list, stop_event):
    # Signal all logger processes to stop
    stop_event.set()

    # Wait for them to exit cleanly, then hard-kill if needed
    for proc in proc_list:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()