import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODES = {
    'onecycle': ('Super-convergence', 99.25, 1406),
    'inv': ('Baseline', 99.03, 10000),
    'clr': ('Super-convergence', 90.4, 20000),
    'pc': ('Baseline', 88.6, 80000),
}


def plot_results(directory, destination=None):
    destination = destination or directory / 'plots'
    destination.mkdir(parents=True, exist_ok=True)
    histories = []
    for path in sorted(directory.glob('*_history.csv')):
        with path.open() as file:
            rows = list(csv.DictReader(file))
        if rows:
            histories.append(rows)
    plt.rcParams.update({'font.size': 10, 'figure.figsize': (8, 4.5)})
    for key, label in [('test_accuracy', 'Test accuracy (%)'), ('train_loss', 'Training loss'),
                       ('test_loss', 'Test loss'), ('lr', 'Learning rate')]:
        fig, ax = plt.subplots()
        for rows in histories:
            mode, seed = rows[0]['mode'], rows[0]['seed']
            name, _, expected = MODES[mode]
            partial = int(rows[-1]['step']) < expected
            title = f'{name}, seed {seed}' + (' (in progress)' if partial else '')
            ax.plot([float(row['epoch']) for row in rows],
                    [float(row[key]) for row in rows], label=title)
        ax.set(xlabel='Epochs', ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(destination / f'{key}.png', dpi=180)
        plt.close(fig)
    metrics = []
    for rows in histories:
        row = rows[-1]
        mode = row['mode']
        name, paper, expected = MODES[mode]
        complete = int(row['step']) == expected
        metrics.append(dict(mode=mode, seed=int(row['seed']), complete=complete,
                            step=int(row['step']), test_accuracy=float(row['test_accuracy']),
                            test_loss=float(row['test_loss']), train_seconds=float(row['train_seconds']),
                            paper_accuracy=paper, delta_vs_paper_pp=float(row['test_accuracy']) - paper if complete else None))
    with (destination / 'metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    aggregate = {}
    for mode in sorted({row['mode'] for row in metrics}):
        completed = [row for row in metrics if row['mode'] == mode and row['complete']]
        if not completed:
            continue
        values = [row['test_accuracy'] for row in completed]
        aggregate[mode] = dict(n=len(values), mean_accuracy=float(np.mean(values)),
                               sd_accuracy=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                               paper_accuracy=MODES[mode][1],
                               delta_vs_paper_pp=float(np.mean(values)) - MODES[mode][1])
    summaries = {path.stem.removesuffix('_summary'): json.loads(path.read_text())
                 for path in directory.glob('*_summary.json')}
    paired = []
    for super_mode, baseline in [('onecycle', 'inv'), ('clr', 'pc')]:
        seeds = {row['seed'] for row in metrics if row['mode'] == super_mode and row['complete']}
        for seed in sorted(seeds):
            first = summaries.get(f'{super_mode}_seed{seed}')
            second = summaries.get(f'{baseline}_seed{seed}')
            if first is None or second is None:
                continue
            matched = first['initial_sha256'] == second['initial_sha256']
            paired.append(dict(seed=seed, super_mode=super_mode, baseline_mode=baseline,
                               initial_weights_match=matched,
                               accuracy_difference_pp=first['test_accuracy'] - second['test_accuracy'],
                               train_time_ratio=second['train_seconds'] / first['train_seconds']))
    comparison = dict(modes=aggregate, paired=paired,
                      reference='arXiv:1708.07120v3: Table 2 (MNIST), Figure 4b (CIFAR-10)',
                      partial_runs_excluded=True)
    (destination / 'comparison.json').write_text(json.dumps(comparison, indent=2))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    plot_results(args.results_dir, args.out_dir)
