import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import least_squares, lsq_linear

from equations import bytes_moved, energy, flops, latency, memory


HERE = Path(__file__).resolve().parent
GIGA = 1_000_000_000


def number(value):
    return float(value) if value not in (None, "") else None


def load_measurements(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            size = int(raw["image_size"])
            batch = int(raw["batch"])
            validation = raw["is_validation"] in ("1", "True")
            row = {
                "image_size": size,
                "batch": batch,
                "status": raw["status"].lower(),
                "is_validation": validation,
                "latency_s": number(raw.get("latency_s")),
                "memory_bytes": number(raw.get("memory_bytes")),
                "energy_j": number(raw.get("energy_j")),
                "energy_method": raw.get("energy_method", ""),
                "gpu_total_memory_bytes": number(raw.get("gpu_total_memory_bytes")),
                "flops_formula": float(flops(size, batch)),
                "memory_formula_bytes": float(memory(size, batch)),
                "bytes_formula": float(bytes_moved(size, batch)),
            }
            rows.append(row)

    return rows


def fit_latency(rows):
    train = [r for r in rows if r["status"] == "ok" and not r["is_validation"]
             and r["latency_s"] is not None and r["latency_s"] > 0]
    f_g = np.array([r["flops_formula"] / GIGA for r in train])
    d_g = np.array([r["bytes_formula"] / GIGA for r in train])
    measured_ms = np.array([r["latency_s"] * 1000 for r in train])

    def residual(x):
        startup_ms, ms_per_gflop, ms_per_gb = x
        estimate_ms = startup_ms + np.maximum(f_g * ms_per_gflop, d_g * ms_per_gb)
        return estimate_ms - measured_ms

    compute_guess = np.median(measured_ms) / max(np.median(f_g), 1e-12)
    memory_guess = np.median(measured_ms) / max(np.median(d_g), 1e-12)
    fits = []
    for fraction, scale in [(0.01, 0.5), (0.10, 1.0), (0.25, 2.0), (0.50, 4.0)]:
        start = [max(np.min(measured_ms) * fraction, 1e-9),
                 compute_guess * scale, memory_guess / scale]
        fits.append(least_squares(residual, start, bounds=(1e-12, np.inf), max_nfev=10_000))

    best = min(fits, key=lambda fit: np.sum(fit.fun ** 2))
    startup_ms, ms_per_gflop, ms_per_gb = best.x
    theta = {
        "t0": float(startup_ms / 1000),
        "P": float(1e12 / ms_per_gflop),
        "W": float(1e12 / ms_per_gb),
    }

    compute_dominates = f_g * ms_per_gflop >= d_g * ms_per_gb
    diagnostic = {
        "training_points": len(train),
        "compute_dominated_training_points": int(np.sum(compute_dominates)),
        "memory_dominated_training_points": int(np.sum(~compute_dominates)),
    }
    return theta, diagnostic


def fit_energy(rows):
    train = [r for r in rows if r["status"] == "ok" and not r["is_validation"]
             and r["energy_j"] is not None and r["energy_j"] > 0]
    if len(train) < 3:
        return None, {"status": "unavailable", "reason": "Fewer than three training rows have GPU energy measurements."}

    f_g = np.array([r["flops_formula"] / GIGA for r in train])
    d_g = np.array([r["bytes_formula"] / GIGA for r in train])
    design = np.column_stack([np.ones(len(train)), f_g, d_g])
    measured_j = np.array([r["energy_j"] for r in train])
    fit = lsq_linear(design, measured_j, bounds=(1e-15, np.inf), max_iter=10_000)

    intercept, joules_per_gflop, joules_per_gb = fit.x
    theta = {
        "E0": float(intercept),
        "alpha": float(joules_per_gflop / GIGA),
        "beta": float(joules_per_gb / GIGA),
    }
    return theta, {"status": "fitted", "training_points": len(train)}


def errors(rows, measured_key, predicted_key):
    pairs = [(r[measured_key], r[predicted_key]) for r in rows
             if r["status"] == "ok" and r[measured_key] is not None and r[predicted_key] is not None]
    if not pairs:
        return None

    observed, predicted = np.array(pairs).T
    return {
        "n": len(pairs),
        "mae": float(np.mean(np.abs(predicted - observed))),
        "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "mean_relative_error_percent": float(100 * np.mean(np.abs(predicted - observed) / observed))
        if np.all(observed > 0) else None,
    }


def compare_plot(rows, measured_key, predicted_key, scale, title, unit, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    plotted = []
    for validation, label, color, marker in [
        (False, "Fit set", "steelblue", "o"),
        (True, "Validation set", "darkorange", "s"),
    ]:
        subset = [r for r in rows if r["status"] == "ok" and r["is_validation"] == validation
                  and r[measured_key] is not None and r[predicted_key] is not None]
        if not subset:
            continue
        observed = np.array([r[measured_key] / scale for r in subset])
        predicted = np.array([r[predicted_key] / scale for r in subset])
        ax.scatter(observed, predicted, label=label, color=color, marker=marker, s=46, alpha=0.85)
        plotted.extend(observed.tolist() + predicted.tolist())

    if not plotted:
        plt.close(fig)
        path.unlink(missing_ok=True)
        return
    low, high = min(plotted), max(plotted)
    pad = 0.05 * (high - low if high > low else max(high, 1))

    ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1, label="Perfect agreement")
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel(f"Measured ({unit})")
    ax.set_ylabel(f"Predicted ({unit})")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def overlay_plot(rows, measured_key, predicted_key, scale, title, unit, path):
    available_batches = sorted({r["batch"] for r in rows if r["status"] == "ok"
                                and r[measured_key] is not None and r[predicted_key] is not None})
    if not available_batches:
        path.unlink(missing_ok=True)
        return
    chosen = [b for b in (1, 16, 128) if b in available_batches]
    if not chosen:
        chosen = sorted({available_batches[i] for i in (0, len(available_batches) // 2, -1)})

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = ("steelblue", "darkorange", "seagreen")
    values = []
    for batch, color in zip(chosen, colors):
        sample = sorted((r for r in rows if r["batch"] == batch and r["status"] == "ok"
                         and r[measured_key] is not None and r[predicted_key] is not None),
                        key=lambda r: r["image_size"])
        ax.plot([r["image_size"] for r in sample],
                [r[predicted_key] / scale for r in sample],
                color=color, linewidth=1.8, label=f"B={batch} model")
        values.extend(r[predicted_key] / scale for r in sample)
        for validation, marker in ((False, "o"), (True, "s")):
            subset = [r for r in sample if r["is_validation"] == validation]
            values.extend(r[measured_key] / scale for r in subset)
            ax.scatter([r["image_size"] for r in subset],
                       [r[measured_key] / scale for r in subset],
                       facecolors="white" if validation else color,
                       edgecolors=color, marker=marker, s=48, zorder=3)

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([], [], linestyle="", marker="o", color="black", label="Measured fit"),
        Line2D([], [], linestyle="", marker="s", markerfacecolor="white", color="black",
               label="Measured validation"),
    ])
    ax.set_xlabel("Input width and height S (pixels)")
    ax.set_ylabel(unit)
    ax.set_title(title)
    positive = [value for value in values if value > 0]
    if positive and len(positive) == len(values) and max(positive) / min(positive) > 50:
        ax.set_yscale("log")

    ax.grid(alpha=0.25)
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def memory_capacity_plot(rows, path):
    fig, ax = plt.subplots(figsize=(7.0, 5.1))
    completed = [r for r in rows if r["status"] == "ok" and r["memory_bytes"] is not None]
    if completed:
        ax.scatter([r["image_size"] for r in completed],
                   [r["memory_bytes"] / GIGA for r in completed],
                   color="steelblue", marker="o", s=38, label="Measured peak")
        ax.scatter([r["image_size"] for r in completed],
                   [r["memory_formula_bytes"] / GIGA for r in completed],
                   facecolors="none", edgecolors="seagreen", marker="o", s=38,
                   label="Formula, completed")
    out_of_memory = [r for r in rows if r["status"] == "oom"]
    if out_of_memory:
        ax.scatter([r["image_size"] for r in out_of_memory],
                   [r["memory_formula_bytes"] / GIGA for r in out_of_memory],
                   color="firebrick", marker="x", s=55, label="Formula, OOM")
    capacities = [r["gpu_total_memory_bytes"] for r in rows if r["gpu_total_memory_bytes"]]
    if capacities:
        ax.axhline(np.median(capacities) / GIGA, color="dimgray", linestyle="--",
                   label="GPU physical memory")

    ax.set_xlabel("Input width and height S (pixels)")
    ax.set_ylabel("Peak memory (GB)")
    ax.set_title("Measured peak, tensor formula and OOM runs")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def oom_analysis(rows):
    completed = [r for r in rows if r["status"] == "ok"]
    out_of_memory = [r for r in rows if r["status"] == "oom"]
    capacities = [r["gpu_total_memory_bytes"] for r in rows if r["gpu_total_memory_bytes"]]
    capacity = float(np.median(capacities)) if capacities else None

    return {
        "completed_runs": len(completed),
        "oom_runs": len(out_of_memory),
        "other_errors": sum(r["status"] == "error" for r in rows),
        "largest_completed_formula_bytes": max((r["memory_formula_bytes"] for r in completed), default=None),
        "smallest_oom_formula_bytes": min((r["memory_formula_bytes"] for r in out_of_memory), default=None),
        "gpu_total_memory_bytes": capacity,
        "oom_cases": [{"image_size": r["image_size"], "batch": r["batch"],
                       "memory_formula_bytes": r["memory_formula_bytes"],
                       "below_physical_capacity": r["memory_formula_bytes"] < capacity if capacity else None}
                      for r in out_of_memory],
    }


def save_predictions(rows, path):
    columns = ["image_size", "batch", "status", "is_validation", "flops_formula",
               "memory_formula_bytes", "memory_bytes", "bytes_formula",
               "latency_s", "latency_predicted_s", "energy_j", "energy_predicted_j", "energy_method"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: ("" if row.get(column) is None else row[column]) for column in columns})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "results" / "measurements.csv")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    rows = load_measurements(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)

    latency_theta, latency_diagnostic = fit_latency(rows)
    energy_theta, energy_diagnostic = fit_energy(rows)
    for row in rows:
        if row["status"] == "ok":
            row["latency_predicted_s"] = float(latency(row["image_size"], row["batch"], latency_theta))
            row["energy_predicted_j"] = (float(energy(row["image_size"], row["batch"], energy_theta))
                                          if energy_theta is not None else None)
        else:
            row["latency_predicted_s"] = None
            row["energy_predicted_j"] = None

    metrics = {}
    for name, measured, predicted in [
        ("latency_s", "latency_s", "latency_predicted_s"),
        ("memory_bytes", "memory_bytes", "memory_formula_bytes"),
        ("energy_j", "energy_j", "energy_predicted_j"),
    ]:
        metrics[name] = {
            "fit": errors([r for r in rows if not r["is_validation"]], measured, predicted),
            "validation": errors([r for r in rows if r["is_validation"]], measured, predicted),
        }

    report = {
        "latency": latency_theta,
        "energy": energy_theta,
        "units": {"t0": "s", "P": "FLOP/s", "W": "byte/s", "E0": "J", "alpha": "J/FLOP", "beta": "J/byte"},
        "latency_fit": latency_diagnostic,
        "energy_fit": energy_diagnostic,
        "errors": metrics,
        "oom_analysis": oom_analysis(rows),
    }
    (args.output / "theta.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_predictions(rows, args.output / "predictions.csv")

    compare_plot(rows, "latency_s", "latency_predicted_s", 1e-3,
                 "Latency: measured vs model", "ms", figures / "latency_pred_vs_measured.png")
    overlay_plot(rows, "latency_s", "latency_predicted_s", 1e-3,
                 "Latency across image sizes", "Time per forward (ms)", figures / "latency_by_size.png")
    compare_plot(rows, "memory_bytes", "memory_formula_bytes", GIGA,
                 "Peak memory: measured vs formula", "GB", figures / "memory_pred_vs_measured.png")
    overlay_plot(rows, "memory_bytes", "memory_formula_bytes", GIGA,
                 "Peak memory across image sizes", "Peak allocated memory (GB)", figures / "memory_by_size.png")
    if energy_theta is not None:
        compare_plot(rows, "energy_j", "energy_predicted_j", 1,
                     "GPU energy: measured vs model", "J / forward", figures / "energy_pred_vs_measured.png")
        overlay_plot(rows, "energy_j", "energy_predicted_j", 1,
                     "GPU energy across image sizes", "Whole-GPU energy (J / forward)", figures / "energy_by_size.png")
    else:
        for name in ("energy_pred_vs_measured.png", "energy_by_size.png"):
            (figures / name).unlink(missing_ok=True)
    memory_capacity_plot(rows, figures / "memory_and_oom.png")

    print("done")


if __name__ == "__main__":
    main()
