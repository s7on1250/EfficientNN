import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from data import load_data, prepare_data
from models import make_model
from optim import SGD, schedule
from plots import plot_results

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
EXPERIMENTS = {
    'lenet': dict(dataset='mnist', batch=512, replicas=1, eval_batch=100,
                  interval=117, modes={'super': 'onecycle', 'baseline': 'inv'},
                  steps={'onecycle': 1406, 'inv': 10000}, short_steps=1406),
    'resnet20': dict(dataset='cifar10', batch=1000, replicas=8, eval_batch=125,
                     interval=100, modes={'super': 'clr', 'baseline': 'pc'},
                     steps={'clr': 20000, 'pc': 80000}, short_steps=20000),
}


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


@torch.no_grad()
def evaluate(model, images, labels, batch):
    model.eval()
    correct, loss = 0, 0.0
    for start in range(0, len(labels), batch):
        target = labels[start:start + batch]
        logits = model(images[start:start + batch])
        correct += (logits.argmax(1) == target).sum().item()
        loss += F.cross_entropy(logits, target, reduction='sum').item()
    model.train()
    return 100 * correct / len(labels), loss / len(labels)


def run(name, mode, seed, device, data, output, config):
    prefix = output / f'{mode}_seed{seed}'
    summary_path = prefix.with_name(prefix.name + '_summary.json')
    if summary_path.exists():
        return
    experiment = EXPERIMENTS[name]
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 100000)
    model = make_model(name, mode).to(device)
    optimizer = SGD(model, name)
    initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                     for p in model.parameters())).hexdigest()
    initial_path = output / f'initial_seed{seed}_weights.pt'
    if not initial_path.exists():
        torch.save(model.state_dict(), initial_path)
    images, labels, test_images, test_labels = data
    batch, replicas = experiment['batch'], experiment['replicas']
    microbatch = batch // replicas
    indices = torch.arange(microbatch, device=device)
    checkpoint = prefix.with_name(prefix.name + '_resume.pt')
    history, begin = [], 0
    train_seconds, eval_seconds, wall_seconds = 0.0, 0.0, 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['rng'])
        begin, history = saved['done'], saved['history']
        train_seconds, eval_seconds = saved['train_seconds'], saved['eval_seconds']
        wall_seconds = saved['wall_seconds']
    started = time.perf_counter()
    window_loss, window_count = 0.0, 0
    total = experiment['steps'][mode]
    for step in range(begin, total):
        synchronize(device)
        tick = time.perf_counter()
        model.zero_grad(set_to_none=True)
        lr, momentum = schedule(mode, step)
        step_loss = 0.0
        for rank in range(replicas):
            idx = (indices * replicas + (step * batch) % len(labels) + rank) % len(labels)
            inputs = images[idx]
            if name == 'resnet20':
                flip = (torch.rand(microbatch, generator=generator) < 0.5).to(device)
                inputs = torch.where(flip[:, None, None, None], inputs.flip(-1), inputs)
                model.rank(rank)
            loss = F.cross_entropy(model(inputs), labels[idx])
            step_loss += loss.item() / replicas
            (loss / replicas).backward()
        optimizer.step(lr, momentum)
        synchronize(device)
        train_seconds += time.perf_counter() - tick
        window_loss += step_loss
        window_count += 1
        done = step + 1
        if done % experiment['interval'] and done not in (experiment['short_steps'], total):
            continue
        tick = time.perf_counter()
        accuracy, test_loss = evaluate(model, test_images, test_labels, experiment['eval_batch'])
        synchronize(device)
        eval_seconds += time.perf_counter() - tick
        row = dict(mode=mode, seed=seed, step=done, epoch=done * batch / len(labels),
                   lr=lr, momentum=momentum,
                   train_loss=window_loss / window_count if name == 'lenet' else step_loss,
                   test_accuracy=accuracy, test_loss=test_loss,
                   train_seconds=train_seconds, eval_seconds=eval_seconds,
                   wall_seconds=wall_seconds + time.perf_counter() - started)
        history.append(row)
        with prefix.with_name(prefix.name + '_history.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        saved = dict(config=config, done=done, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), rng=generator.get_state(),
                     history=history, train_seconds=train_seconds,
                     eval_seconds=eval_seconds, wall_seconds=row['wall_seconds'])
        temp = checkpoint.with_suffix('.tmp')
        torch.save(saved, temp)
        temp.replace(checkpoint)
        save_json(output / 'status.json', dict(row, total_steps=total))
        print(f'{mode} seed={seed} step={done}/{total} accuracy={accuracy:.2f}% lr={lr:.5f}', flush=True)
        window_loss, window_count = 0.0, 0
    torch.save(model.state_dict(), prefix.with_name(prefix.name + '_weights.pt'))
    save_json(summary_path, dict(history[-1], initial_sha256=initial,
                                parameters=sum(p.numel() for p in model.parameters())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=EXPERIMENTS, required=True)
    parser.add_argument('--mode', choices=['all', 'super', 'baseline'], default='all')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    experiment = EXPERIMENTS[args.model]
    data_dir = args.data_dir or ROOT / 'data' / experiment['dataset']
    output = args.out_dir or ROOT / 'results' / args.model
    seeds = args.seeds if args.seeds is not None else ([0, 1, 2] if args.model == 'lenet' else [0])
    torch.set_num_threads(args.threads)
    code_hash = hashlib.sha256(b''.join((CODE / file).read_bytes()
                              for file in ['train.py', 'models.py', 'data.py', 'optim.py'])).hexdigest()
    config = dict(model=args.model, experiment=experiment, seeds=seeds, device=args.device,
                  threads=args.threads, python=platform.python_version(), torch=torch.__version__,
                  code_sha256=code_hash, train_loss='window mean' if args.model == 'lenet' else 'last step')
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    existing = output / 'config.json'
    
    save_json(existing, config)
    prepare_data(args.model, data_dir)
    data = tuple(t.to(args.device) for t in load_data(args.model, data_dir))
    modes = experiment['modes'].values() if args.mode == 'all' else [experiment['modes'][args.mode]]
    for mode in modes:
        for seed in seeds:
            run(args.model, mode, seed, args.device, data, output, config)
    plot_results(output)


if __name__ == '__main__':
    main()
