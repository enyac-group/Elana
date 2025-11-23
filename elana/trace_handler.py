import os
import time
import gzip
import json
import socket
import logging

import torch
from hta.trace_analysis import TraceAnalysis

logger = logging.getLogger(os.path.basename(__file__))

def trace_handler(prof: torch.profiler.profile, dir_name="torch_profile_output",
                  worker_name = None, use_gzip: bool = False,
                  file_prefix="prefilling", device="cuda:0"):
    if not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            raise RuntimeError("Can't create directory: " + dir_name) from e
    if not worker_name:
        worker_name = f"{socket.gethostname()}_{os.getpid()}"
    # Use nanosecond here to avoid naming clash when exporting the trace
    timestamp = time.time_ns()
    file_name = f"{file_prefix}.{worker_name}.{timestamp}.pt.trace.json"
    if use_gzip:
        file_name = file_name + ".gz"
    prof.export_chrome_trace(os.path.join(dir_name, file_name))
    # Fix the rank issue for  HolisticTraceAnalysis
    # reference: https://github.com/facebookresearch/HolisticTraceAnalysis/issues/107
    # FIXME: This does not work for json.gz
    # rn_rank = np.random.randint(low=0, high=16, dtype=int) # If there are multiple traces files, then each file should have a unique rank value.
    if use_gzip:
        with gzip.open(os.path.join(dir_name, file_name), mode="rt") as fin:
            data = json.loads(fin.read())
        data["distributedInfo"] = {"rank": 0} # must use 0. I don't know why. If there are multiple traces files, then each file should have a unique rank value.
        with gzip.open(os.path.join(dir_name, file_name), 'w') as fout:
            fout.write(json.dumps(data).encode('utf-8')) 
    else:
        with open(os.path.join(dir_name, file_name), "r") as fin:
            data = json.load(fin)
        data["distributedInfo"] = {"rank": 0} # must use 0. I don't know why. If there are multiple traces files, then each file should have a unique rank value.
        with open(os.path.join(dir_name, file_name), "w") as fout:
            json.dump(data, fout, indent=2)

    analyzer = TraceAnalysis(trace_files={0: file_name}, trace_dir=dir_name)
    kernel_type_metrics_df, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown(visualize=False, num_kernels=100)
    kernel_type_metrics_df.to_csv(os.path.join(dir_name, f'kernel_type_metrics.{file_prefix}.{timestamp}.csv'), index=False)
    kernel_metrics_df.to_csv(os.path.join(dir_name, f'kernel_metrics.{file_prefix}.{timestamp}.csv'), index=False)
    # this feature is at https://github.com/facebookresearch/HolisticTraceAnalysis/pull/209
    # To get accurate kernel results, checkout this branch https://github.com/hychiang-git/HolisticTraceAnalysis/tree/dev/no_merge_cpu_kernels
    if hasattr(analyzer, "get_gpu_user_annotation_breakdown"):
        try:
            user_annotation_kernel_type_metrics_df, user_annotation_metrics_df = analyzer.get_gpu_user_annotation_breakdown(visualize=False, num_kernels=100)
            user_annotation_kernel_type_metrics_df.to_csv(os.path.join(dir_name, f'user_annotation_kernel_type_metrics.{file_prefix}.{timestamp}.csv'), index=False)
            user_annotation_metrics_df.to_csv(os.path.join(dir_name, f'user_annotation_metrics.{file_prefix}.{timestamp}.csv'), index=False)
        except Exception as e:
            logger.warning(f"Failed to get user annotation breakdown: {e}")
    # Construct the memory timeline file.
    # !!! This does not work for graph cache !!!
    html_name = f"{file_prefix}.{worker_name}.{timestamp}.html"
    prof.export_memory_timeline(os.path.join(dir_name, html_name), device=device)

