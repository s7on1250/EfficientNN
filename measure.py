import argparse
import csv
import gc
import json
import math
import platform
import random
from statistics import median
import time
from pathlib import Path

import pynvml
import torch

from equations import bytes_moved, flops, memory
from models import CNN


BASE_SIZES = (32, 64, 128, 224, 256, 384, 512)
BASE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CSV_COLUMNS = (
    "image_size",
    "batch",
    "status",
    "latency_s",
    "memory_bytes",
    "energy_j",
    "energy_method",
    "is_validation",
    "flops_formula",
    "memory_formula_bytes",
    "bytes_formula",
    "gpu_total_memory_bytes",
    "error",
)


def make_grid(seed):
    rng = random.Random(seed)
    extra_sizes = rng.sample(
        [s for s in range(32, 513, 16) if s not in BASE_SIZES], 4
    )
    extra_batches = rng.sample(
        [b for b in range(1, 257) if b not in BASE_BATCHES], 3
    )

    sizes = sorted(BASE_SIZES + tuple(extra_sizes))
    batches = sorted(BASE_BATCHES + tuple(extra_batches))
    pairs = [(s, b) for s in sizes for b in batches]
    held_out = set(random.Random(seed + 1).sample(range(len(pairs)), round(0.2 * len(pairs))))

    return [(s, b, int(i in held_out)) for i, (s, b) in enumerate(pairs)]


def is_out_of_memory(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and ("out of memory" in str(error).lower()
             or "CUDNN_STATUS_ALLOC_FAILED" in str(error))
    )


def run_forwards(model, x, count):
    for _ in range(count):
        y = model(x)
        del y

    torch.cuda.synchronize()


def measure_one(model, image_size, batch, warmup, repeats, energy_handle, energy_window_s):
    x = None
    y = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        x = torch.randn(batch, 3, image_size, image_size, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            run_forwards(model, x, warmup)

            torch.cuda.reset_peak_memory_stats()
            y = model(x)
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated()
            del y
            y = None

            times = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                y = model(x)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                del y
                y = None

            median_s = median(times)

            count = max(1, min(100_000, math.ceil(energy_window_s / median_s)))
            before_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(energy_handle)
            run_forwards(model, x, count)
            after_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(energy_handle)
            energy_j = (after_mj - before_mj) / 1000.0 / count

        return peak_bytes, median_s, energy_j
    finally:
        del y, x
        gc.collect()
        torch.cuda.empty_cache()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "measurements.csv")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--energy-window-s", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = arguments()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cuda_index = torch.cuda.current_device()
    device = torch.cuda.get_device_properties(cuda_index)
    model = CNN().cuda().eval()
    pynvml.nvmlInit()
    energy_handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_index)

    grid = make_grid(args.seed)
    if args.limit is not None:
        grid = grid[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.parent / "metadata.json"
    metadata = {
        "gpu_name": device.name,
        "gpu_total_memory_bytes": device.total_memory,
        "model_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "cuda_device_index": cuda_index,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "seed": args.seed,
        "warmup": args.warmup,
        "latency_repeats": args.repeats,
        "energy_window_s": args.energy_window_s,
        "configurations_completed": len(grid),
    }

    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for s, b, is_validation in grid:
            row = {
                "image_size": s,
                "batch": b,
                "is_validation": is_validation,
                "flops_formula": float(flops(s, b)),
                "memory_formula_bytes": float(memory(s, b)),
                "bytes_formula": float(bytes_moved(s, b)),
                "gpu_total_memory_bytes": device.total_memory,
            }
            try:
                peak, latency_s, energy_j = measure_one(
                    model, s, b, args.warmup, args.repeats, energy_handle, args.energy_window_s,
                )
                row.update(status="ok", latency_s=latency_s, memory_bytes=peak,
                           energy_j=energy_j, energy_method="nvml_total_energy",
                           error="")
            except RuntimeError as error:
                if not is_out_of_memory(error):
                    raise
                row["status"] = "OOM"
                row["error"] = str(error).replace("\n", " ")[:400]
                row["energy_method"] = "unavailable"

            writer.writerow(row)

    pynvml.nvmlShutdown()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
