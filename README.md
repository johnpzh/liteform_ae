# LiteForm Artifact

The is the artifact repo for the paper
[LiteForm: Lightweight and Automatic Format Composition for Sparse Matrix-Matrix Multiplication on GPUs](https://dl.acm.org/doi/10.1145/3731545.3731574). The artifact was test on an NVIDIA V100 GPU 16GB, Driver Version 575.57.08, CUDA Version 12.9.

## Prerequisite
Please set up the environment according to [SparseTir Artifact](https://github.com/uwsampl/sparsetir-artifact).

## Run
Please run the following commands
```bash
cd playground_pipline
bash drive05.spmm.end-to-end.sh
```

## Utilities
The script `scripts/download_suitesparse.v3.curl.py` can be used to download all SuiteSparse matrices. It was tested using Python 3.9.18. The total disk space is about 900GB.