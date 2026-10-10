import gzip
import pickle
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

FILES = (
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
)

def dataset(directory, prefix):
    image_path = directory / (prefix + '-images-idx3-ubyte.gz')
    label_path = directory / (prefix + '-labels-idx1-ubyte.gz')
    images = gzip.decompress(image_path.read_bytes())
    labels = gzip.decompress(label_path.read_bytes())
    magic, count, h, w = struct.unpack('>IIII', images[:16])
    lmagic, lcount = struct.unpack('>II', labels[:8])
    x = torch.from_numpy(np.frombuffer(images, dtype=np.uint8, offset=16).copy()).reshape(count, 1, h, w).float() / 256.0
    y = torch.from_numpy(np.frombuffer(labels, dtype=np.uint8, offset=8).copy()).long()
    return (x, y)

def load_cifar(root):
    path = root / 'cifar-10-python.tar.gz'
    batches = []
    with tarfile.open(path, 'r:gz') as archive:
        for name in [*[f'data_batch_{i}' for i in range(1, 6)], 'test_batch']:
            with archive.extractfile('cifar-10-batches-py/' + name) as f:
                batches.append(pickle.load(f, encoding='bytes'))
    x = np.concatenate([b[b'data'] for b in batches[:5]]).reshape(-1, 3, 32, 32).astype(np.float32)
    y = np.concatenate([b[b'labels'] for b in batches[:5]])
    tx = batches[5][b'data'].reshape(-1, 3, 32, 32).astype(np.float32)
    ty = np.array(batches[5][b'labels'])
    mean = x.mean(axis=0, keepdims=True)
    x -= mean
    tx -= mean
    return (torch.from_numpy(x), torch.from_numpy(y).long(), torch.from_numpy(tx), torch.from_numpy(ty).long(), mean)


def prepare_data(name, root):
    root.mkdir(parents=True, exist_ok=True)
    if name == "lenet":
        urls = {file: "https://ossci-datasets.s3.amazonaws.com/mnist/" + file for file in FILES}
    else:
        file = "cifar-10-python.tar.gz"
        urls = {file: "https://www.cs.toronto.edu/~kriz/" + file}
    for file, url in urls.items():
        path = root / file
        if not path.exists():
            temp = path.with_suffix(".download")
            urllib.request.urlretrieve(url, temp)
            temp.replace(path)


def load_data(name, root):
    if name == "lenet":
        return (*dataset(root, "train"), *dataset(root, "t10k"))
    return load_cifar(root)[:4]
