import numpy as np


CONV_PARAMETERS = (
    3 * 32 * 7**2
    + 32 * 64 * 5**2
    + 64 * 128 * 3**2
    + 128 * 256
    + 256 * 256 * 3**2
    + 256 * 512
)
LINEAR_PARAMETERS = 512 * 256 + 256 * 100
PARAMETER_BYTES = 4 * (CONV_PARAMETERS + LINEAR_PARAMETERS)


def _sizes(image_size):
    s = np.asarray(image_size, dtype=float)
    return {
        "input": 3 * s**2,
        "conv7": 32 * (s / 2) ** 2,
        "pool": 32 * (s / 4) ** 2,
        "conv5": 64 * (s / 4) ** 2,
        "conv3a": 128 * (s / 8) ** 2,
        "conv1a": 256 * (s / 8) ** 2,
        "conv3b": 256 * (s / 16) ** 2,
        "conv1b": 512 * (s / 16) ** 2,
    }


def flops(image_size, batch):
    b = np.asarray(batch, dtype=float)
    n = _sizes(image_size)
    conv_macs = (
        n["conv7"] * (3 * 7**2)
        + n["conv5"] * (32 * 5**2)
        + n["conv3a"] * (64 * 3**2)
        + n["conv1a"] * 128
        + n["conv3b"] * (256 * 3**2)
        + n["conv1b"] * 256
    )

    linear_macs = 512 * 256 + 256 * 100
    return 2 * b * (conv_macs + linear_macs)


def memory(image_size, batch):
    b = np.asarray(batch, dtype=float)
    n = _sizes(image_size)
    live_tensors = [
        n["input"] + n["conv7"],
        n["input"] + n["conv7"] + n["pool"],
        n["input"] + n["pool"] + n["conv5"],
        n["input"] + n["conv5"] + n["conv3a"],
        n["input"] + n["conv3a"] + n["conv1a"],
        n["input"] + n["conv1a"] + n["conv3b"],
        n["input"] + n["conv3b"] + n["conv1b"],
        n["input"] + n["conv1b"] + 512,
        n["input"] + 512 + 256,
        n["input"] + 512 + 256 + 100,
    ]
    peak_elements = np.maximum.reduce(live_tensors)

    return PARAMETER_BYTES + 4 * b * peak_elements


def bytes_moved(image_size, batch):
    b = np.asarray(batch, dtype=float)
    n = _sizes(image_size)
    conv_pairs = (
        ("input", "conv7"),
        ("pool", "conv5"),
        ("conv5", "conv3a"),
        ("conv3a", "conv1a"),
        ("conv1a", "conv3b"),
        ("conv3b", "conv1b"),
    )

    conv_bytes = 4 * b * sum(n[source] + n[result] for source, result in conv_pairs)
    relu_bytes = 4 * b * 2 * sum(
        n[name] for name in ("conv7", "conv5", "conv3a", "conv1a", "conv3b", "conv1b")
    )
    pool_bytes = 4 * b * (n["conv7"] + n["pool"])
    average_bytes = 4 * b * (n["conv1b"] + 512)
    linear_bytes = 4 * b * ((512 + 256) + 2 * 256 + (256 + 100))

    return PARAMETER_BYTES + conv_bytes + relu_bytes + pool_bytes + average_bytes + linear_bytes


def latency(image_size, batch, theta):
    compute_time = flops(image_size, batch) / theta["P"]
    memory_time = bytes_moved(image_size, batch) / theta["W"]

    return theta["t0"] + np.maximum(compute_time, memory_time)


def energy(image_size, batch, theta_energy):
    compute_energy = theta_energy["alpha"] * flops(image_size, batch)
    memory_energy = theta_energy["beta"] * bytes_moved(image_size, batch)

    return theta_energy["E0"] + compute_energy + memory_energy
