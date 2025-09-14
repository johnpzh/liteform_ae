# import tvm
import os
import sys
import argparse
import torch as th
import math
import numpy as np
import pandas as pd
import time
from format_matrix_market import MTX
from format_algos import (
    build_hyb_format,
    bench_hyb_with_config,
    search_bucket_config,
    CostModelSettings,
    bench_bsrmm,
    bench_naive,
)
from joblib import load


NAMES = ['cora', 'citeseer', 'pubmed', 'ppi',  'arxiv', 'proteins', 'reddit']
# NAMES = ['proteins']


def get_X_for_format_selection(g: MTX):
    mtx_features = g.matrix_features()
    features = [
        "num_rows",
        "num_cols",
        "nnz",
        "avg_nnz_per_row",
        "min_nnz_per_row",
        "max_nnz_per_row",
        "std_dev_nnz_per_row",
    ]
    # # test
    # print(f"mtx_features: {mtx_features}")
    # # end test
    # values = [mtx_features[f] for f in features]
    # # test
    # print(f"values: {values}")
    # # end test
    X = [np.array([mtx_features[f] for f in features])]

    return X


def get_X_for_num_partitions(g: MTX, num_features: int):
    mtx_features = g.matrix_features()
    features = [
        "num_rows",
        "num_cols",
        "nnz",
        "avg_nnz_density_per_row",
        "min_nnz_density_per_row",
        "max_nnz_density_per_row",
        "std_dev_nnz_density_per_row",
    ]
    X = [np.array([mtx_features[f] for f in features] + [num_features])]

    return X


def predict_format_selection(g: MTX,
                             clf):
    # Get the matrix characteristics
    X_selection = get_X_for_format_selection(g)

    # Predict
    time_s_start = time.perf_counter()
    y_pred = clf.predict(X_selection)
    time_s_end = time.perf_counter()
    time_s = time_s_end - time_s_start

    format = "CELL" if y_pred[0] == 1 else "BCSR"
    print(f"name: {g.name} format: {format} time(s): {time_s}")

    return format, time_s


def predict_num_partitions(g: MTX,
                           clf: str,
                           num_features: int):
    # Get the matrix characteristics
    X_selection = get_X_for_num_partitions(g, num_features)

    # Predict
    time_s_start = time.perf_counter()
    y_pred = clf.predict(X_selection)
    time_s_end = time.perf_counter()
    time_s = time_s_end - time_s_start

    num_parts = y_pred[0]
    print(f"name: {g.name} num_parts: {num_parts} time(s): {time_s}")

    return num_parts, time_s


if __name__ == "__main__":
    parser = argparse.ArgumentParser("predict the format selection")
    parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    parser.add_argument("--model-format-selection", "-s", type=str, help="model for format selection")
    parser.add_argument("--model-num-partitions", "-p", type=str, help="model for format selection")
    # parser.add_argument("--feature-csv", "-f", type=str, help="input feature csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    name = args.dataset
    model_selection_name = args.model_format_selection
    model_num_partitions = args.model_num_partitions

    # Load data
    clf_selection = load(model_selection_name)
    clf_partition = load(model_num_partitions)

    names_list = []
    formats_list = []
    num_features_list = []
    num_parts_list = []
    time_selet_list = []
    time_num_parts_list = []
    time_bucket_list = []
    time_exe_list = []

    for name in NAMES:
        print(f"\nGo to name {name} ...")
        # Load the data
        g = MTX(name)

        # Predict format selection
        format, time_selet = predict_format_selection(g, clf=clf_selection)
        # formats_list.append(format)
        # time_selet_list.append(time_selet)

        if format == "CELL":
            # Use CELL format
            # Predict number of partitions
            # num_features = 32
            for num_features in [32, 64, 128, 256, 512]:
                num_parts, time_parts = predict_num_partitions(g, clf=clf_partition, num_features=num_features)

                # test
                if name == "ppi":
                    num_parts = 16
                if name == "proteins":
                    num_parts = 32
                # end test

                # Build buckets
                print("Configuring ...", flush=True)
                time_bucket_start = time.perf_counter()
                cost_model_config = CostModelSettings(feat_size=num_features, num_parts=num_parts)
                bucket_config = search_bucket_config(g, num_parts, cost_model_config)
                time_bucket_end = time.perf_counter()
                time_bucket = time_bucket_end - time_bucket_start

                # Run kernel
                features = g.matrix_features()
                features["K"] = num_features
                x = th.ones((g.num_dst_nodes(), num_features))
                y_ndarray = g.dot(x.numpy())
                hyb_format = build_hyb_format(g, bucket_config)
                # print(f"name: {name} feat_size: {num_features} num_partitions: {num_parts} cost_model_config: {cost_model_config} bucket_config: {bucket_config}", flush=True)
                try:
                    exe_time = bench_hyb_with_config(g,
                                            x,
                                            # y_golden,
                                            y_ndarray,
                                            feat_size=num_features,
                                            bucket_widths=bucket_config,
                                            bucketing_format=hyb_format,
                                            coarsening_factor=2,
                                            num_col_parts=num_parts,
                                            use_implicit_unroll=True)
                    print(f"name: {name} format: {format} feat_size: {num_features} num_partitions: {num_parts} cost_model_config: {cost_model_config} bucket_config: {bucket_config} exe_time: {exe_time}", flush=True)
                except Exception as e:
                    exe_time = math.inf
                    print(e, file=sys.stderr)

                # Record statistics
                names_list.append(name)
                formats_list.append(format)
                num_features_list.append(num_features)
                num_parts_list.append(num_parts)
                time_selet_list.append(time_selet)
                time_num_parts_list.append(time_parts)
                time_bucket_list.append(time_bucket)
                time_exe_list.append(exe_time)
        else:
            # Not use CELL format
            format = "BSR"
            # feat_size = 32
            for num_features in [32, 64, 128, 256, 512]:
                # mtx = MTX(name)
                num_src = g.num_src_nodes()
                num_dst = g.num_dst_nodes()
                # assert num_src == num_dst, f"Error: matrix {name} is not sqaure ({num_src} x {num_dst})."
                if num_src != num_dst:
                    num_src = max(num_src, num_dst)
                    num_dst = num_src

                # Block size
                bsize = 32
                # Padding the size
                print(f"original_size: ({num_src}, {num_dst})")
                num_block_row = (num_src + bsize - 1) // bsize
                num_src = num_block_row * bsize
                num_dst = num_src

                print(f"padded_size: ({num_src}, {num_dst}) bsize: {bsize}")
                try:
                    if name not in ['proteins', 'reddit']:
                        bsr_weight = g.tobsr_with_padding(shape=(num_src, num_dst), blocksize=(bsize, bsize))
                        # print(f"g: {g.coo_mtx.toarray()}")
                        # print(f"bsr_weight: {bsr_weight.toarray()}")
                        print(f"g.shape: ({g.num_src_nodes(), g.num_dst_nodes()}) nnz: {g.num_edges()}")
                        print(f"bsr_weight.shape: ({bsr_weight.shape[0], bsr_weight.shape[1]}) nnz: {bsr_weight.nnz} size: {bsr_weight.size}")

                        x = th.rand(bsr_weight.shape[1], num_features).half()
                        # print(f"mtx_bsr: {mtx_bsr}")
                        # csr_weight = mtx.tocsr()
                        exe_time = bench_bsrmm(bsr_weight, x, block_size=bsize)
                        print(f"name: {name} exe_time: {exe_time}")
                        # print(f"mtx_csr: {mtx_csr}")
                        # formats_list.append("BSR")
                        # format = "BSR"
                        # blocksizes_list.append(bsize)
                        # time_exe_list.append(exe_time)
                    else:
                        # name is in ['proteins', 'reddit']
                        raise Exception(f"name {name} cannot do BSR (scipy.sparse.bsr_matrix() will cause Segmentation Fault).")
                except Exception as e:
                    print(e, file=sys.stderr)
                    print(f"name {name} in BSR is out-of-memory. Do CSR then.")
                    try:
                        # BSR is too large to be handled.
                        # Try CSR instead.
                        format = "CSR"
                        x = th.rand((g.num_dst_nodes(), num_features))
                        # y_golden = dgl.ops.copy_u_sum(g, x)
                        y_ndarray = g.dot(x.numpy())
                        exe_time = bench_naive(g,
                                            x,
                                            y_ndarray,
                                            feat_size=num_features,
                                            coarsening_factor=2)
                        print(f"name: {name} exe_time: {exe_time}")
                        # formats_list.append("CSR")
                        # format = "CSR"
                        # blocksizes_list.append(0)
                        # time_exe_list.append(exe_time)
                    except Exception as e2:
                        print(e2, file=sys.stderr)
                        print(f"name: {name} in CSR is also out-of-memory. Exit.")
                        # formats_list.append("Crashed")
                        exe_time = 0
                        format = "CRASHED"
                        # blocksizes_list.append(0)
                        # time_exe_list.append(0)

                # Record statistics
                names_list.append(name)
                formats_list.append(format)
                num_features_list.append(num_features)
                num_parts_list.append(0)
                time_selet_list.append(time_selet)
                time_num_parts_list.append(0)
                time_bucket_list.append(0)
                time_exe_list.append(exe_time)

    table = {
        "name": names_list,
        "format": formats_list,
        "feat_size": num_features_list,
        "num_parts": num_parts_list,
        "time_select(s)": time_selet_list,
        "time_num_parts(s)": time_num_parts_list,
        "time_bucket(s)": time_bucket_list,
        "time_exe_our(ms)": time_exe_list,
    }

    df = pd.DataFrame(data=table)
    print(df.to_string())
    df.to_csv(f"output.{sys.argv[0]}.csv", index=False)
