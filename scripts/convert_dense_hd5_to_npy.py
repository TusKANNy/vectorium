import argparse
import sys

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: h5py. Install with `pip install h5py`."
    ) from exc


def list_datasets(h5_file):
    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    h5_file.visititems(visitor)
    return datasets


def select_dataset(h5_file, dataset_path):
    if dataset_path:
        if dataset_path in h5_file:
            return h5_file[dataset_path]
        raise ValueError(f"Dataset not found: {dataset_path}")

    datasets = list_datasets(h5_file)
    if not datasets:
        raise ValueError("No datasets found in the .hd5 file.")
    if len(datasets) > 1:
        joined = "\n".join(f"- {name}" for name in datasets)
        raise ValueError(
            "Multiple datasets found; specify one with --dataset:\n" + joined
        )
    return h5_file[datasets[0]]


def copy_chunked(dataset, out_path, chunk_rows):
    shape = dataset.shape
    if len(shape) == 0:
        data = dataset[()]
        np.save(out_path, data)
        return

    mmap = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dataset.dtype, shape=shape
    )

    if len(shape) == 1:
        total = shape[0]
        for start in range(0, total, chunk_rows):
            end = min(start + chunk_rows, total)
            mmap[start:end] = dataset[start:end]
    else:
        total = shape[0]
        for start in range(0, total, chunk_rows):
            end = min(start + chunk_rows, total)
            slicer = (slice(start, end),) + (slice(None),) * (len(shape) - 1)
            mmap[slicer] = dataset[slicer]

    mmap.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Convert a dense .hd5 dataset to a .npy file."
    )
    parser.add_argument("input", help="Input .hd5 file")
    parser.add_argument("output", help="Output .npy file")
    parser.add_argument(
        "--dataset",
        help="Dataset path inside the .hd5 file (required if multiple datasets exist)",
        default=None,
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Write using a memmap and chunked reads (lower peak memory usage)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=8192,
        help="Rows per chunk when using --chunked",
    )

    args = parser.parse_args()

    if args.chunk_rows <= 0:
        raise SystemExit("--chunk-rows must be > 0")

    with h5py.File(args.input, "r") as h5_file:
        dataset = select_dataset(h5_file, args.dataset)

        if args.chunked:
            copy_chunked(dataset, args.output, args.chunk_rows)
        else:
            data = dataset[...]
            np.save(args.output, data)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
