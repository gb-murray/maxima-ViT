# Local datasets

Place the synthetic and experimental HDF5 files referenced by `configs/train_swin.yaml` in this directory, or set `XRD_DATA_DIR` to a different host directory before starting Docker Compose.

Dataset files in this directory are excluded from Git. The directory is mounted read-only at `/app/data` inside the container.
