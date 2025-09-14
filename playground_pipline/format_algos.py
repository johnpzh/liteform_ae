from format_matrix_market import MTX, Bucket
import numpy as np
import bisect
import tvm
import tvm.testing
import tvm.tir as tir
from tvm.script import tir as T
from tvm.sparse import lower_sparse_buffer, lower_sparse_iter
from tvm.sparse import (
    FormatRewriteRule,
    lower_sparse_buffer,
    lower_sparse_iter,
    column_part_hyb,
    format_decompose,
)
import tvm.sparse
from sparsetir_artifact import profile_tvm_ms
import time
import sys
import math
from typing import List, Callable, Any, Tuple, Union
import torch as th
from scipy.stats import variation
import pandas as pd

__all__ = ["build_hyb_format", 
           "bench_hyb_with_config", 
           "search_bucket_config", 
           "CostModelSettings",
           "bench_bsrmm",
           "bench_naive"]

# ################# #
# ----------------- #
# CELL Format
# ----------------- #
# ################# #

class CostModelSettings:
    # def __init__(self, mem_w: float, bub_w: float):
    #     self.mem_w = mem_w  # weight of memory access overhead in the cost model
    #     self.bub_w = bub_w  # weight of bubble overhead in the cost model
    # def __str__(self) -> np.str:
    #     return F"(mem_w={self.mem_w}, bub_w={self.bub_w})"
    def __init__(self, 
                 feat_size: int, 
                 num_parts: int):
        self.feat_size = feat_size  # weight of memory access overhead in the cost model
        self.num_parts = num_parts  # weight of bubble overhead in the cost model
    def __str__(self) -> np.str:
        return F"(feat_size={self.feat_size}, num_parts={self.num_parts})"

#####################
# SparseTIR settings
@T.prim_func
def ell(
    a: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    O = T.dense_fixed(1)
    I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
    J = T.sparse_fixed(I, (n, nnz_cols), indices_j)
    A = T.match_sparse_buffer(a, (O, I, J), "float32")
    T.evaluate(0)


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    num_tiles: T.int32,
    nnz: T.int32,
    coarsening_factor: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    # I, J, J_detach, K1, K2, K3 are Axes.
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K1 = T.dense_fixed(num_tiles)
    K2 = T.dense_fixed(coarsening_factor)
    K3 = T.dense_fixed(32)
    # A, B, C are Sparse Buffers.
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), "float32")
    C = T.match_sparse_buffer(c, (I, K1, K2, K3), "float32")
    # Sparse Iterations.
    with T.iter([I, J, K1, K2, K3], "SRSSS", "csrmm") as [i, j, k1, k2, k3]:
        with T.init():
            C[i, k1, k2, k3] = 0.0
        C[i, k1, k2, k3] = C[i, k1, k2, k3] + A[i, j] * B[j, k1, k2, k3]


def csr2ell_inv_index_map(o, i, j):
    return i, j


def csr2ell_index_map(i, j):
    return 0, i, j

# End SparseTIR settings
#########################


# def get_max_bucket_width(buckets: dict):
#     return max(buckets.keys)


# def modify_max_bucket_width(buckets: dict,
#                            num_parts: int):
#     curr_max_width = get_max_bucket_width(buckets)
#     # get_bubble_overhead(buckets)
#     while curr_max_width < 1:
#         # Halve
#         curr_max_width /= 2

def print_table(columns: dict):
    # Pandas settings
    pd.set_option("display.width", 800)
    pd.set_option("display.max_columns", None)

    df = pd.DataFrame(data=columns)
    print(df)


def bucket_statistics(bucket_dict: dict):
    max_width = max(bucket_dict.keys())
    total_num_rows_in_bucket = 0
    total_elements_in_bucket = 0
    total_nnz_in_bucket = 0
    for width, val in bucket_dict.items():
        total_nnz_in_bucket += val.nnz
        row_size_sub_matrix = max_width // width
        rows_nnz = val.num_rows
        num_sub_matrix = (rows_nnz + row_size_sub_matrix - 1) // row_size_sub_matrix
        rows_total = num_sub_matrix * row_size_sub_matrix
        total_num_rows_in_bucket += rows_total
        total_elements_in_bucket += rows_total * width
    
    collect = {
        "total_num_rows_in_bucket": total_num_rows_in_bucket,
        "total_elements_in_bucket": total_elements_in_bucket,
        "total_nnz_in_bucket": total_nnz_in_bucket
    }

    return collect


def profile_100_runs_ms(f: tvm.runtime.Module, args: List[Any]) -> float:
    return run_100_times(lambda: f(*args))


def run_100_times(f: Callable[[], None]) -> float:
    # flush_l2 = os.getenv("FLUSH_L2", "OFF") == "ON"
    # n_warmup = 10
    # n_repeat = 100
    n_warmup = 0
    n_repeat = 1
    # if flush_l2:
    cache = th.empty(int(256e6), dtype=th.int8, device="cuda")
    start_event = [
        th.cuda.Event(enable_timing=True) for i in range(n_repeat)
    ]
    end_event = [
        th.cuda.Event(enable_timing=True) for i in range(n_repeat)
    ]

    # Warm-up
    for _ in range(n_warmup):
        f()

    # Benchmark
    # Repeat 
    for i in range(n_repeat):
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        f()
        end_event[i].record()
    # Record clocks
    th.cuda.synchronize()
    times = th.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    dur = th.mean(times).item()
    print(F"#### Run {len(times)} times (warm up {n_warmup}) in ms:")
    for t in times:
        print("#### {:.6}".format(t))
    cv = variation(times)
    print("#### Coefficent_of_Variation: {:.2%}".format(cv))
    
    return dur



def build_list_of_list(bucket_widths):
    """Return an empty list of lists with shape (num_partitions, num_buckets), and the inner element is a list.

    Args:
        bucket_widths (list of lists): All bucket widths with shape (num_partitions, num_buckets). [ [b_0_0, b_0_1, b_0_2, ...], [b_1_0, b_1_1, b_1_2, ...], ... ], b_i_j the bucket width of bucket j in partition i.

    Returns:
        list of list: an empty list of lists with shape (num_partitions, num_buckets), and the inner element is a list.
    """
    # shape is [num_buckets, num_partitions]
    num_parts = len(bucket_widths)
    res = []
    for part_ind in range(num_parts):
        num_buckets = len(bucket_widths[part_ind])
        inner = [ [] for _ in range(num_buckets) ]
        res.append(inner)
    
    return res


def build_hyb_format(g: MTX,
                    #  num_parts,
                     bucket_widths):
    """build the hyb format based on the given bucket widths.

    Args:
        g (MTX): the matrix
        bucket_widths (list of lists): All bucket widths with shape (num_partitions, num_buckets). [ [b_0_0, b_0_1, b_0_2, ...], [b_1_0, b_1_1, b_1_2, ...], ... ], b_i_j the bucket width of bucket j in partition i.

    Returns:
        Tuple: a tuple of (row_indices_nd, col_indices_nd, mask_nd). row_indices_nd is with shape (num_partitions, num_buckets), and the inner element is a numpy ndarray. col_indices_nd and mask_nd are in the same shape.
    """
    num_parts = len(bucket_widths)
    num_rows = g.num_src_nodes()
    num_cols = g.num_dst_nodes()
    partition_size = (num_cols + num_parts - 1) // num_parts

    # # test
    # print(F"\npartition_size: {partition_size}")
    # # end test

    # Count the degree of each row in each partition
    # degree_counter shape is partition_size * num_rows
    degree_counter = [ [0] * num_rows for i in range(num_parts) ]
    for row_ind, col_ind in zip(g.coo_mtx.row, g.coo_mtx.col):
        part_ind = col_ind // partition_size
        degree_counter[part_ind][row_ind] += 1
    
    indptr, indices, _ = g.adj_tensors("csr")

    # num_bucket = len(bucket_widths)
    # 3-dimentional list
    # It is a list of list with shape [num_parts, num_bucket], where the innermost element is a list
    # The inner most element needs to be a numpy ndarray in the end.
    # row_indices = [ [ [] for j in range(num_buckets)] for i in range(num_parts) ]
    # col_indices = [ [ [] for j in range(num_buckets)] for i in range(num_parts) ]
    # mask = [ [ [] for j in range(num_buckets)] for i in range(num_parts) ]
    row_indices = build_list_of_list(bucket_widths)
    col_indices = build_list_of_list(bucket_widths)
    mask = build_list_of_list(bucket_widths)

    # for row_ind, col_ind in zip(g.coo_mtx.row, g.coo_mtx.col):
    for row_ind in range(num_rows):
        for col_loc in range(indptr[row_ind], indptr[row_ind + 1]):
            col_ind = indices[col_loc]
            part_ind = col_ind // partition_size
            b_widths = bucket_widths[part_ind]
            num_buckets = len(b_widths)
            degree = degree_counter[part_ind][row_ind]
            bucket_ind = bisect.bisect_left(b_widths, degree)
            if bucket_ind == num_buckets:
                # The very long row goes to the largest bucket
                bucket_ind -= 1
            bucket_width = b_widths[bucket_ind]
            need_new_row = False
            # # # test
            # print("")
            # print(F"row_ind: {row_ind} col_ind: {col_ind} degree: {degree} part_ind: {part_ind} bucket_ind: {bucket_ind} bucket_width: {bucket_width}")
            # # print(F"len(col_indices[part_ind={part_ind}][bucket_ind={bucket_ind}]): {len(col_indices[part_ind][bucket_ind])} (bucket_width - 1): {(bucket_width - 1)}")
            # # # end test
            remainder = len(col_indices[part_ind][bucket_ind]) & (bucket_width - 1)
            # remainder = len(col_indices[part_ind][bucket_ind]) % bucket_width
            if remainder == 0:
                # Current row is full, so need a new row
                need_new_row = True
            else:
                assert row_indices[part_ind][bucket_ind], F"row_indices[{part_ind}][{bucket_ind}] is empty"
                if row_ind != row_indices[part_ind][bucket_ind][-1]:
                    # row_ind is changed, so need a new row
                    need_new_row = True
                    # Padding current row if not full
                    for _ in range(remainder, bucket_width):
                        col_indices[part_ind][bucket_ind].append(0)
                        mask[part_ind][bucket_ind].append(0)
            
            if need_new_row:
                # Current row is full, or the row_ind changed
                assert len(col_indices[part_ind][bucket_ind]) & (bucket_width - 1) == 0, F"invalid padding for col_indices[{part_ind}][{bucket_ind}] and bucket_width={bucket_width}"
                # assert len(col_indices[part_ind][bucket_ind]) % bucket_width == 0, F"invalid padding for col_indices[{part_ind}][{bucket_ind}] and bucket_width={bucket_width}"
                row_indices[part_ind][bucket_ind].append(row_ind)
                # # test
                # print(F"row_indices[{part_ind}][{bucket_ind}]: {row_indices[part_ind][bucket_ind]}")
                # # end test
            
            col_indices[part_ind][bucket_ind].append(col_ind)
            mask[part_ind][bucket_ind].append(1)
            # # test
            # print(F"len(col_indices[part_ind={part_ind}][bucket_ind={bucket_ind}]): {len(col_indices[part_ind][bucket_ind])} {col_indices[part_ind][bucket_ind]} col_ind: {col_ind}")
            # # end test
    # # test
    # print(F"#207 len(col_indices[0][1]): {len(col_indices[0][1])}")  
    # # end test
            
    # # test
    # for part_ind in range(num_parts):
    #     b_widths = bucket_widths[part_ind]
    #     num_buckets = len(b_widths)
    #     print(F"\npart_ind: {part_ind} num_buckets: {num_buckets}")
    #     for bucket_ind in range(num_buckets):
    #         bucket_width = bucket_widths[part_ind][bucket_ind]
    #         print(F"\nbucket_ind: {bucket_ind} bucket_width: {bucket_width}")
    #         print(F"row_indices[part_ind={part_ind}][bucket_ind={bucket_ind}]: {row_indices[part_ind][bucket_ind]}")
    #         print(F"col_indices[part_ind={part_ind}][bucket_ind={bucket_ind}]: {col_indices[part_ind][bucket_ind]}")
    #         print(F"mask[part_ind={part_ind}][bucket_ind={bucket_ind}]: {mask[part_ind][bucket_ind]}")
    # # end test

    # Padding the last rows and convert to numpy ndarray required by SparseTIR
    row_indices_nd = []
    col_indices_nd = []
    mask_nd = []
    for part_ind in range(num_parts):
        row_indices_part_local = []
        col_indices_part_local = []
        mask_part_local = []
        b_widths = bucket_widths[part_ind]
        num_buckets = len(b_widths)
        for bucket_ind in range(num_buckets):
            bucket_width = b_widths[bucket_ind]
            remainder = len(col_indices[part_ind][bucket_ind]) & (bucket_width - 1)
            # remainder = len(col_indices[part_ind][bucket_ind]) % bucket_width
            # Padding the last row
            if remainder:
                for _ in range(remainder, bucket_width):
                    col_indices[part_ind][bucket_ind].append(0)
                    mask[part_ind][bucket_ind].append(0)

            num_nz_rows = len(row_indices[part_ind][bucket_ind])
            # # test
            # print(F"len(col_indices[{part_ind}][{bucket_ind}]): {len(col_indices[part_ind][bucket_ind])}")
            # print(F"num_nz_rows: {num_nz_rows} bucket_width: {bucket_width} num_nz_rows * bucket_width: {num_nz_rows * bucket_width}")
            # # end test
            assert len(col_indices[part_ind][bucket_ind]) == num_nz_rows * bucket_width, F"invalid padding for len(col_indices[{part_ind}][{bucket_ind}]) is not equal to num_nz_rows * bucket_width ({num_nz_rows} * {bucket_width})"
            assert len(mask[part_ind][bucket_ind]) == num_nz_rows * bucket_width, F"invalid padding for len(mask[{part_ind}][{bucket_ind}]) is not equal to num_nz_rows * bucket_width ({num_nz_rows} * {bucket_width})"
            
            # Convert to numpy ndarray
            if num_nz_rows:
                row_indices_part_local.append(np.array(row_indices[part_ind][bucket_ind]))
                col_indices_part_local.append(np.array(col_indices[part_ind][bucket_ind]).reshape(num_nz_rows, bucket_width))
                mask_part_local.append(np.array(mask[part_ind][bucket_ind]).reshape(num_nz_rows, bucket_width))
            else:
                row_indices_part_local.append(np.empty(num_nz_rows, dtype=int))
                col_indices_part_local.append(np.empty(num_nz_rows * bucket_width, dtype=int).reshape(num_nz_rows, bucket_width))
                mask_part_local.append(np.empty(num_nz_rows * bucket_width, dtype=int).reshape(num_nz_rows, bucket_width))
        row_indices_nd.append(row_indices_part_local)
        col_indices_nd.append(col_indices_part_local)
        mask_nd.append(mask_part_local)

    # # test
    # for part_ind in range(num_parts):
    #     b_widths = bucket_widths[part_ind]
    #     num_buckets = len(b_widths)
    #     for bucket_ind in range(num_buckets):
    #         print(F"\npart_ind: {part_ind} num_buckets: {num_buckets} bucket_ind: {bucket_ind} bucket_width: {b_widths[bucket_ind]}")
    #         bucket_width = bucket_widths[part_ind][bucket_ind]
    #         print(F"row_indices_nd[part_ind={part_ind}][bucket_ind={bucket_ind}]: {row_indices_nd[part_ind][bucket_ind]}")
    #         print(F"col_indices_nd[part_ind={part_ind}][bucket_ind={bucket_ind}]: {col_indices_nd[part_ind][bucket_ind]}")
    #         print(F"mask_nd[part_ind={part_ind}][bucket_ind={bucket_ind}]: {mask_nd[part_ind][bucket_ind]}")
    # # end test
            
    return (row_indices_nd, col_indices_nd, mask_nd)
        

def bench_hyb_with_config(g,
                          x,
                          # y_golden,
                          y_ndarray,
                          feat_size=128,
                          # bucket_sizes=[],
                          bucket_widths=[],
                          bucketing_format=[],
                          coarsening_factor=2,
                          num_col_parts=1,
                          use_implicit_unroll=False,
                          check_correctness=False):
    """Build and benchmark the SpMM kernel using hyb format basedon SparseTIR and TVM.

    """
    tvm_start_time = time.perf_counter()
    # num_buckets = len(bucket_sizes)
    coarsening_factor = min(coarsening_factor, feat_size // 32)
    # indptr, indices, _ = g.adj_tensors("csc")
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    nnz = g.num_edges()
    # Changed by Zhen Peng on 3/21/2024
    # global cached_bucketing_format
    # if cached_bucketing_format is None:
    #     indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
    #     indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
    #     cached_bucketing_format = column_part_hyb(
    #         m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
    #     )

    #
    # indptr_nd = tvm.nd.array(indptr, device=tvm.cpu())
    # indices_nd = tvm.nd.array(indices, device=tvm.cpu())
    # indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
    # indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
    # cached_bucketing_format = column_part_hyb(
    #     m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
    # )
    # row_indices, col_indices, mask = cached_bucketing_format
    row_indices, col_indices, mask = bucketing_format

    # rewrite csrmm
    nnz_cols_symbol = ell.params[-1]
    rewrites = []
    for part_id in range(num_col_parts):
        b_widths = bucket_widths[part_id]
        # for bucket_id, bucket_size in enumerate(bucket_sizes):
        for bucket_id, bucket_size in enumerate(b_widths):
            rewrites.append(
                FormatRewriteRule(
                    str(part_id) + "_" + str(bucket_id),
                    ell.specialize({nnz_cols_symbol: bucket_size}),
                    ["A"],
                    ["I", "J"],
                    ["O", "I", "J"],
                    {"I": ["O", "I"], "J": ["J"]},
                    csr2ell_index_map,
                    csr2ell_inv_index_map,
                )
            )
    mod = tvm.IRModule.from_expr(csrmm)
    mod = format_decompose(mod, rewrites)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    # specialize
    params = mod["main"].params
    param_map = {
        params[5]: m,  # m
        params[6]: n,  # n
        params[7]: feat_size // coarsening_factor // 32,  # num_tiles,
        params[8]: nnz,  # nnz
        params[9]: coarsening_factor,  # coersening_factor
    }
    loc_base = 14
    loc_step = 7
    loc = loc_base
    for part_id in range(num_col_parts):
        b_widths = bucket_widths[part_id]
        num_buckets = len(b_widths)
        for bucket_id in range(num_buckets):
            # param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 4]] = m
            # param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 5]] = n
            # param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 6]] = row_indices[
            #     part_id
            # ][bucket_id].shape[0]
            param_map[params[loc]] = m
            param_map[params[loc + 1]] = n
            param_map[params[loc + 2]] = row_indices[
                part_id
            ][bucket_id].shape[0]
            loc += loc_step

    mod["main"] = mod["main"].specialize(param_map).with_attr("horizontal_fuse", True)

    # schedule
    iter_names = []
    for part_id in range(num_col_parts):
        b_widths = bucket_widths[part_id]
        num_buckets = len(b_widths)
        for bucket_id in range(num_buckets):
            iter_names.append(F"csrmm_{part_id}_{bucket_id}")

    sch = tvm.tir.Schedule(mod)
    # for sp_iter_name in [
    #     "csrmm_{}_{}".format(i, j) for j in range(num_buckets) for i in range(num_col_parts)
    # ]:
    for sp_iter_name in iter_names:
        sp_iteration = sch.get_sparse_iteration(sp_iter_name)
        o, i, j, k1, k2, k3 = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [o, i])

    mod = sch.mod
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    for part_id in range(num_col_parts):
        b_widths = bucket_widths[part_id]
        num_buckets = len(b_widths)
        # for bucket_id, bucket_size in enumerate(bucket_sizes):
        for bucket_id, bucket_size in enumerate(b_widths):
            is_atomic = num_col_parts > 1 or bucket_id + 1 == num_buckets
            blk = sch.get_block("csrmm_{}_{}0".format(part_id, bucket_id))
            i, j, foo, foi, fi = sch.get_loops(blk)
            sch.reorder(foo, fi, j, foi)
            if is_atomic:
                sch.annotate(blk, "atomic", True)
                write_blk = sch.cache_write(blk, 0, "local")
                sch.reverse_compute_at(write_blk, fi, True)
                # sch.unroll(sch.get_loops(write_blk)[-2])
            sch.bind(fi, "threadIdx.x")
            sch.bind(foo, "blockIdx.y")
            sch.unroll(foi)
            if use_implicit_unroll:
                sch.annotate(foi, "pragma_unroll_explicit", 0)
            sch.unroll(j)
            if use_implicit_unroll:
                sch.annotate(j, "pragma_unroll_explicit", 0)
            # # test
            # print(F"part_id: {part_id} bucket_id: {bucket_id} bucket_size: {bucket_size} b_widths: {b_widths}")
            # # end test
            # io, ioi, ii = sch.split(i, [None, bucket_sizes[-1] // bucket_size, 8])
            io, ioi, ii = sch.split(i, [None, b_widths[-1] // bucket_size, 8])
            # io, ioi, ii = sch.split(i, [b_widths[-1] // bucket_size, 8])
            # io, ii = sch.split(i, [1, 8])
            sch.bind(io, "blockIdx.x")
            sch.bind(ii, "threadIdx.y")
            init_blk = sch.decompose_reduction(blk, fi)
            ax0, ax1 = sch.get_loops(init_blk)[-2:]
            sch.bind(ax0, "threadIdx.x")
            sch.unroll(ax1)
            if use_implicit_unroll:
                sch.annotate(ax1, "pragma_unroll_explicit", 0)

    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    f = tvm.build(mod, target="cuda")

    # prepare nd array
    b_nd = tvm.nd.array(
        x.numpy().reshape(-1).astype("float32"),
        device=tvm.cuda(0),
    )
    c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype("float32"), device=tvm.cuda(0))
    # prepare args
    args = [b_nd, c_nd]

    for part_id in range(num_col_parts):
        b_widths = bucket_widths[part_id]
        # for bucket_id, _ in enumerate(bucket_sizes):
        for bucket_id, _ in enumerate(b_widths):
            # weight = tvm.nd.array(
            #     mask[part_id][bucket_id].numpy().reshape(-1).astype("float32"), device=tvm.cuda(0)
            # )
            # rows = tvm.nd.array(
            #     row_indices[part_id][bucket_id].numpy().astype("int32"), device=tvm.cuda(0)
            # )
            # cols = tvm.nd.array(
            #     col_indices[part_id][bucket_id].numpy().reshape(-1).astype("int32"),
            #     device=tvm.cuda(0),
            # )
            weight = tvm.nd.array(
                mask[part_id][bucket_id].reshape(-1).astype("float32"), device=tvm.cuda(0)
            )
            rows = tvm.nd.array(
                row_indices[part_id][bucket_id].astype("int32"), device=tvm.cuda(0)
            )
            cols = tvm.nd.array(
                col_indices[part_id][bucket_id].reshape(-1).astype("int32"),
                device=tvm.cuda(0),
            )
            args += [weight, rows, cols]

    tvm_end_time = time.perf_counter()
    tvm_exe_time = tvm_end_time - tvm_start_time
    print(F"tvm_schedule_time(s): {tvm_exe_time:.6}")

    # Dump code generated
    # test
    # dev_module = func.imported_modules[0]
    # print(dev_module.get_source())

    # dev_module = f.imported_modules[0]
    # print(dev_module.get_source())
    # sys.exit(-1)
    # end test

    # # test accuracy
    f(*args)
    if check_correctness:
        tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_ndarray, rtol=1e-4)
    # # tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)

    # evaluate time
    dur = profile_tvm_ms(f, args)
    # dur = profile_100_runs_ms(f, args)
    print("cell exe time: {:.6f} ms".format(dur))

    return dur


def get_bucket_widths_list(buckets: dict):
    """Return list of bucket widths for a partition.

    Args:
        buckets (dict): A dictionary is for a partition, and every dictionary element is a bucket {width: Bucket()}.

    Returns:
        List: A list of bucket widths for the given partition. [w_0, w_1, ...].
    """
    bucket_pool = list(buckets.keys())
    bucket_pool.sort()

    return bucket_pool


def move_largest_bucket(buckets: dict, 
                        pre_max_width: int,
                        new_max_width: int):
    """Move elements in the largest bucket to the new largest bucket. 
    new_max_width == pre_max_width/2

    Args:
        buckets (dict) *OUTPUT*: A dictionary is for a partition, and every dictionary element is a bucket {width: Bucket()}.
        pre_max_width (int): old largest bucket width
        new_max_width (int): new largest bucket width.
    """
    # # test
    # print(F"pre_max_width: {pre_max_width} new_max_width: {new_max_width}")
    # # end test
    larger_bucket = buckets[pre_max_width]
    ratio = pre_max_width // new_max_width
    new_num_rows = larger_bucket.num_rows * ratio
    if new_max_width not in buckets:
        buckets[new_max_width] = Bucket(new_max_width, new_num_rows, larger_bucket.nnz, larger_bucket.row_indices, larger_bucket.col_indices)
    else:
        buckets[new_max_width].num_rows += new_num_rows
        buckets[new_max_width].nnz += larger_bucket.nnz
        buckets[new_max_width].row_indices += larger_bucket.row_indices
        buckets[new_max_width].col_indices |= larger_bucket.col_indices

    del buckets[pre_max_width]


def get_bubble_overhead(buckets_dict: dict):
    """Get the bubble overhead of all buckets in a partition

    Args:
        buckets_dict (dict): A dictionary is for a partition, and every dictionary element is a bucket {width: Bucket()}.
    """
    origin_max_width = max(buckets_dict.keys())
    overhead = 0

    for width, bucket in buckets_dict.items():
        num_rows = bucket.num_rows
        shape_rows = origin_max_width / width
        remaining = num_rows % shape_rows  # num_rows % shape_rows
        # remaining = num_rows & (shape_rows - 1)  # num_rows % shape_rows
        if remaining != 0:
            bubble_rows = shape_rows - remaining
        else:
            bubble_rows = 0
        # # test
        # print(F"bucket: {bucket} max_width: {origin_max_width} bubble_rows: {bubble_rows}")
        # # end test
        # overhead += bubble_rows
        overhead += bubble_rows * width

    return overhead


def get_memory_overhead(buckets_dict: dict,
                    pre_max_width: int,
                    new_max_width: int):
    """Return the memory access overhead for moving elements from the previous largest bucket to the new largest bucket. 
    new_max_width == pre_max_width/2

    Args:
        buckets_dict (dict): A dictionary is for a partition, and every dictionary element is a bucket {width: Bucket()}.
        pre_max_width (int): old largest bucket width.
        new_max_width (int): new largest bucket width.

    Returns:
        float: memory access overhead
    """
    larger_bucket = buckets_dict[pre_max_width]
    ratio = pre_max_width / new_max_width
    cost = (ratio - 1) * larger_bucket.num_rows

    # # Update the buckets
    # new_num_rows = larger_bucket.num_rows * ratio
    # if new_max_width not in buckets:
    #     buckets[new_max_width] = Bucket(new_max_width, new_num_rows, larger_bucket.nnz)
    # else:
    #     buckets[new_max_width].num_rows += new_num_rows
    #     buckets[new_max_width].nnz += larger_bucket.nnz

    # del buckets[pre_max_width]

    return cost    


def modify_bucket_widths(buckets_dict: dict,
                         cost_model_config: CostModelSettings):
    """Modify initial bucket settings of a partition by using a cost model.

    Args:
        buckets_dict (Dict{bucket_width: Bucket}): A dictionary is for a partition, and every dictionary element is a bucket {width: Bucket()}.
    """
    # # test
    # for width, bk in buckets.items():
    #     print(F"504 width: {width} bucket: {bk}")
    # # end test
    # Get original overhead as baseline
    # base_overhead = get_bubble_overhead(buckets_dict)
    # feat_size = cost_model_config.feat_size
    # num_parts = cost_model_config.num_parts
    # base_overhead, _, _, _ = get_memory_cost(buckets_dict, 
    #                                          feat_size=feat_size, 
    #                                          num_parts=num_parts)
    cost_dict = get_memory_cost(buckets_dict, 
                                cost_model_config)
                                # feat_size=feat_size, 
                                # num_parts=num_parts)
    base_overhead = cost_dict["cost"]
    base_buckets_list = get_bucket_widths_list(buckets_dict)
    min_overhead = base_overhead
    best_setting = base_buckets_list
    # # test
    # print(F"base_overhead: {base_overhead} base_buckets_list: {base_buckets_list}")
    # # end test

    origin_max_width = max(buckets_dict.keys())
    pre_max_width = origin_max_width
    new_max_width = pre_max_width // 2
    # mem_w = cost_model_config.mem_w
    # bub_w = cost_model_config.bub_w
    while new_max_width >= 1:
        # # test
        # print(F"pre_max_width: {pre_max_width} new_max_width: {new_max_width}")
        # # end test
        # Calculate the overhead if change the max bucket width
        # mem_overhead = get_memory_overhead(buckets_dict, pre_max_width, new_max_width)
        move_largest_bucket(buckets_dict, pre_max_width, new_max_width)
        # # test
        # for width, bk in buckets_dict.items():
        #     print(F"539 width: {width} bucket: {bk}")
        # # end test
        # bb_overhead = get_bubble_overhead(buckets_dict)
        # total_overhead = mem_w * mem_overhead + bub_w * bb_overhead
        # # test
        # print(F"mem_overhead: {mem_overhead} bb_overhead: {bb_overhead} total_overhead: {total_overhead}")
        # # end test
        # memory_cost, _, _, _ = get_memory_cost(buckets_dict, 
        #                                        feat_size=feat_size, 
        #                                        num_parts=num_parts)
        cost_dict = get_memory_cost(buckets_dict, 
                                    cost_model_config)
                                    # feat_size=feat_size, 
                                    # num_parts=num_parts)
        memory_cost = cost_dict["cost"]
        if memory_cost < min_overhead:
            best_setting = get_bucket_widths_list(buckets_dict)
            min_overhead = memory_cost
        pre_max_width = new_max_width
        new_max_width //= 2
    
    return best_setting


def get_bucket_config(init_buckets: list,
                      cost_model_config: CostModelSettings):
    """Return bucket configurations.

    Args:
        init_buckets (list): List of dictionary. len(List) == num_parts. Every dictionary is for a partition, and every dictionary element is a bucket {width: Bucket()}.

    Returns:
        List of lists: len(List) == num_parts. Every inner list contains bucket sizes. [[w_0_0, w_0_1, ...], [w_1_0, w_1_1, ...], ...].
    """
    num_parts = len(init_buckets)
    bucket_config = []

    for part_ind in range(num_parts):
        # # test
        # print(F"Go to partition {part_ind}...")
        # # end test
        buckets = init_buckets[part_ind]

        if not buckets:
            # If the bucket is empty, add a empty width list.
            bucket_config.append([])
            continue

        # # test
        # print(F"533 buckets: {buckets}")
        # # end test
        # # test
        # print(F"old buckets setting: {get_bucket_widths_list(buckets)}")
        # # end test
        bucket_widths = modify_bucket_widths(buckets, cost_model_config)
        # # test
        # print(F"537 bucket_widths: {bucket_widths}")
        # # end test
        # # test
        # print(F"new buckets setting: {bucket_widths}")
        # # end test
        bucket_config.append(bucket_widths)

    return bucket_config


def get_init_buckets(g: MTX, 
                     num_parts: int, 
                     width_limit: int=1024):
    """Return the initial buckets of the matrix

    Args:
        num_parts (int): number of partitions
        width_limit (int, optional): The maximum limit of the bucket width. Defaults to 1024.

    Returns:
        List of dictionary: List of dictionary. len(List) == num_parts. Every dictionary is for a partition, and every dictionary element is a bucket {width: Bucket()}.
    """
    # row_indices = [None] * num_parts
    # col_indices = [None] * num_parts

    num_rows = g.num_src_nodes()
    num_cols = g.num_dst_nodes()
    partition_size = (num_cols + num_parts - 1) // num_parts

    # # test
    # print(F"num_rows: {num_rows} num_cols: {num_cols} num_parts: {num_parts} partition_size: {partition_size}")
    # # end test

    # Count the degree of each row in each partition
    # degree_counter shape is partition_size * num_rows
    degree_counter = [ [0] * num_rows for i in range(num_parts) ]
    edgelists = [ [ [] for j in range(num_rows) ] for i in range(num_parts) ]  # size (num_parts x num_rows), and every element is a list
    for row_ind, col_ind in zip(g.coo_mtx.row, g.coo_mtx.col):
        part_ind = col_ind // partition_size
        degree_counter[part_ind][row_ind] += 1
        edgelists[part_ind][row_ind].append(col_ind)

    # # test
    # for part_ind in range(num_parts):
    #     for row_ind in range(num_rows):
    #         print(F"edgelists[part_ind={part_ind}][row_ind={row_ind}]: {edgelists[part_ind][row_ind]}")
    # # end test
    
    # # test
    # for part_ind in range(num_parts):
    #     print(F"degree_counter[{part_ind}]: {degree_counter[part_ind]}")
    # # end test

    # Put rows into its corresponding bucket, a row with length l and
    # 2^{i - 1} < l <= 2^{i} should be in the bucket with width 2^{i}
    buckets = []
    for part_ind in range(num_parts):
        # buckets.append({})
        b_pool = {}
        for row_ind in range(num_rows):
            degree = degree_counter[part_ind][row_ind]
            # # test
            # print(F"degree_counter[{part_ind}][{row_ind}]: {degree}")
            # # end test
            if 0 == degree:
                continue
            pow_ceil = math.ceil(math.log2(degree))
            width = int(2 ** pow_ceil)
            new_rows = 1
            if width > width_limit:
                # Limit the width according to GPU block thread limit (1024 for CUDA)
                ratio = width // width_limit
                width = width_limit
                new_rows *= ratio

            neighbors = edgelists[part_ind][row_ind]
            # # test
            # print(F"neighbors: {neighbors}")
            # # end test
            if width not in b_pool:
                # A new bucket
                b_pool[width] = Bucket(width, new_rows, degree, [row_ind], set(neighbors))
            else:
                # Existing bucket
                b_pool[width].nnz += degree
                b_pool[width].num_rows += new_rows
                b_pool[width].row_indices.append(row_ind)
                b_pool[width].col_indices |= set(neighbors)  # adding to the set of column indices
        b_pool = dict(sorted(b_pool.items())) # sort by keys
        buckets.append(b_pool)
    
    return buckets


def move_width_one_bucket(buckets_dict: dict,
                          widths_list: list):
    """If the bucket width is 1 and number of rows of the bucket is also 1 (i.e., only has 1 element in the bucket), move the elements into a existing larger bucket.

    Args:
        buckets_dict (dict): A dictionary is for a partition, and every dictionary element is a bucket {width: Bucket()}.
        widths_list (list): Bucket widths of this partition, [w_0, w_1, ...]

    Raises:
        ValueError: if the bucket only has one element, raise an exception.
    """
    base_width = 1
    new_width = 2
    max_width = max(widths_list)
    while new_width <= max_width and new_width not in widths_list:
        new_width *= 2
    
    if new_width > max_width:
        raise ValueError("The partition is too small, only having 1 non-zero element.")
    else:
        move_largest_bucket(buckets_dict, base_width, new_width)




def modify_single_row_buckets(buckets: list):
    """If a bucket only had 1 row, move this row to a smaller bucket to that SparseTIR would not crash.

    Args:
        buckets (list) *OUTPUT*: List of dictionary. len(List) == num_parts. Every dictionary is for a partition, and every dictionary element is a bucket {width: Bucket()}.

    Raises:
        ValueError: _description_
    """
    num_parts = len(buckets)
    for part_ind in range(num_parts):
        b_pool = buckets[part_ind]
        widths_list = list(b_pool.keys())
        widths_list.sort(reverse=True)
        # The width from large to small
        for width in widths_list:
            num_rows = b_pool[width].num_rows
            if num_rows == 1:
                # SparseTIR does not support singl-row bucket
                # Move this row to the smaller bucket
                if width == 1:
                    # raise ValueError("The partition is too small, only having 1 non-zero element.")
                    move_width_one_bucket(b_pool, widths_list)
                else:
                    new_width = width // 2
                    move_largest_bucket(b_pool, width, new_width)



def print_buckets(buckets: list):
    num_parts = len(buckets)
    for part_ind in range(num_parts):
        bucket_dict = buckets[part_ind]
        print_a_bucket(bucket_dict, part_ind)
        # max_width = max(bucket_dict.keys())
        # total_rows_in_block = 0
        # for w, b in bucket_dict.items():
        #     num_rows = b.num_rows
        #     block_size = max_width // w
        #     num_blocks = (num_rows + block_size - 1) // block_size
        #     rows_in_block = num_blocks * block_size
        #     total_rows_in_block += rows_in_block
        #     print(F"part_ind: {part_ind} width: {w} Bucket: {b} rows_in_block: {rows_in_block}")
        # print(F"total_rows_in_block: {total_rows_in_block}")


def print_a_bucket(bucket_dict: dict,
                   part_ind: int):
    max_width = max(bucket_dict.keys())
    total_rows_in_block = 0
    for w, b in bucket_dict.items():
        num_rows = b.num_rows
        block_size = max_width // w
        num_blocks = (num_rows + block_size - 1) // block_size
        rows_in_block = num_blocks * block_size
        total_rows_in_block += rows_in_block
        print(F"part_ind: {part_ind} width: {w} Bucket: {b} rows_in_block: {rows_in_block}")
    print(F"total_rows_in_block: {total_rows_in_block}")


def search_bucket_config(g: MTX,
                        # x,
                        # y_ndarray,
                        # feat_size,
                        num_parts: int,
                        cost_model_config: CostModelSettings):
                        # coarsening_factor,
                        # use_implicit_unroll):
    """Searching the bucket sizes.

    Args:
        g (MTX): the matrix
        num_parts (int): number of partitions

    Returns:
        List of lists: len(List) == num_parts. Every inner list contains bucket sizes. [[w_0_0, w_0_1, ...], [w_1_0, w_1_1, ...], ...].
    """
    # init_buckets()
    # Initial buckets are list of dictionary. len(List) == num_parts. Every dictionary is for a partition, and every dictionary element is a bucket {width: Bucket()}.
    # init_buckets = g.init_buckets(num_parts)
    init_buckets = get_init_buckets(g, num_parts)
    # # test
    # print("\ninit_buckets:")
    # print_buckets(init_buckets)
    # # end test
    # # test
    # for part_ind in range(num_partitions):
    #     buckets = init_buckets[part_ind]
    #     print(F"part_id: {part_ind} num_buckets: {len(buckets)}")
    #     for width, bucket in buckets.items():
    #         print(F"bucekt: width={width} num_rows={bucket.num_rows} nnz={bucket.nnz}")
    # # end test

    modify_single_row_buckets(init_buckets)
    # # test
    # print("\nupdated_buckets")
    # print_buckets(init_buckets)
    # # sys.exit(-1)
    # # end test
    
    # Get bucket configuration, a list of lists. len(List) == num_parts. Every inside list contains bucket sizes. [[w_0_0, w_0_1, ...], [w_1_0, w_1_1, ...], ...].
    # bucket_config = get_bucket_config(init_buckets, cost_model_config)
    bucket_config = get_bucket_config(init_buckets, cost_model_config)

    return bucket_config


def get_memory_cost(bucket_dict: dict,
                    cost_model_config: CostModelSettings):
                    # feat_size: int,
                    # # output_num_rows: int,
                    # num_parts: int):
    # cost = 2 * I * K + |set(col_indices(i, k)| * J + I * J
    # cost = 2 * bucket_num_rows * bucket_width + len(bucket_col_indices) * feat_size + output_num_rows * feat_size

    # stats = {
    #     "bucket_width": [],
    #     "bucket_memory_cost": [],
    #     "2IK": [],
    #     "|set(col_ind[i,k])|J": [],
    #     "IJ": [],
    #     "Atomic?": [],
    #     "AtomicWeight": []
    # }
    feat_size = cost_model_config.feat_size
    num_parts = cost_model_config.num_parts

    cost = 0
    cost1 = 0
    cost2 = 0
    cost3 = 0
    max_bucket_width = max(bucket_dict.keys())
    bucket_num_rows_list = []
    bucket_cost_list = []
    bucket_thread_load_list = []
    M = 0
    for width, bucket in bucket_dict.items():
        ## 2 * bucket_num_rows * bucket_width
        # bucket_num_rows = bucket.num_rows
        block_size = max_bucket_width // width
        num_block = (bucket.num_rows + block_size - 1) // block_size
        bucket_num_rows = num_block * block_size
        bucket_num_rows_list.append(bucket_num_rows)
        bucket_cost1 = 2 * bucket_num_rows * width
        # bucket_cost1 = 2 * bucket_num_rows * width + 10 * bucket_num_rows
        # bucket_cost1 = bucket_num_rows * width

        ## bucket_width * feat_size
        # bucket_cost2 = width * feat_size
        # bucket_cost2 = width * bucket_num_rows * feat_size
        bucket_cost2 = len(bucket.col_indices) * feat_size

        ## atomic_weight * (output_num_rows * feat_size)
        is_atomic = (num_parts > 1 or width == max_bucket_width)
        output_num_rows = len(bucket.row_indices)
        M += output_num_rows
        atomic_weight = bucket_num_rows / output_num_rows if is_atomic else 1
        # atomic_weight = 2 if is_atomic else 1
        # atomic_weight = 1
        bucket_cost3 = atomic_weight * output_num_rows * feat_size

        thread_load = 2 * width + width + 1 * atomic_weight
        bucket_thread_load_list.append(thread_load)

        ## Summary
        bucket_cost = bucket_cost1 + bucket_cost2 + bucket_cost3
        bucket_cost_list.append(bucket_cost)
        cost += bucket_cost
        cost1 += bucket_cost1
        cost2 += bucket_cost2
        cost3 += bucket_cost3

    # Load imbalance penalty
    max_thread_load = max(bucket_thread_load_list)
    load_imbalance_penalty = 0
    penalty1 = 0
    tmp_penalty_list = []
    # active_threads = 0
    # for thread_load, bucket_num_rows in zip(bucket_thread_load_list, bucket_num_rows_list):
    #     # threads = bucket_num_rows * feat_size
    #     # tmp_penalty = (max_thread_load - thread_load) * threads
    #     tmp_penalty = (max_thread_load - thread_load) * bucket_num_rows
    #     penalty1 += tmp_penalty
    #     tmp_penalty_list.append(tmp_penalty)
    #     # active_threads += threads
    penalty2 = 0
    # active_threads = sum(bucket_num_rows_list) * feat_size
    # device_threads = 80 * 2048 / num_parts  # NVIDIA V100 has 80 SMs, 2048 threads per SM.
    # if active_threads < device_threads:
    #     penalty2 += (device_threads - active_threads) * max_thread_load
    load_imbalance_penalty += penalty1
    load_imbalance_penalty += penalty2

    # # test
    # print(F"bucket_widths: {list(bucket_dict.keys())}")
    # print(F"max_thread_load: {max_thread_load}")
    # print(F"bucket_thread_load_list: {bucket_thread_load_list}")
    # print(F"bucket_num_rows_list: {bucket_num_rows_list}")
    # print(F"penalty1: {penalty1} {tmp_penalty_list}")
    # print(F"active_thrad: {active_threads} ( - {device_threads} = {device_threads - active_threads}) penalty2: {penalty2}")
    # # end test
    cost += load_imbalance_penalty

        # stats["bucket_width"].append(width)
        # stats["bucket_memory_cost"].append(bucket_cost)
        # stats["2IK"].append(bucket_cost1)
        # stats["|set(col_ind[i,k])|J"].append(bucket_cost2)
        # stats["IJ"].append(bucket_cost3)
        # stats["Atomic?"].append(is_atomic)
        # stats["AtomicWeight"].append(atomic_weight)
    # cost /= sum(bucket_num_rows_list) * feat_size

    # ratio = M * feat_size / num_parts / sum(bucket_num_rows_list)
    # # # test
    # # print(F"M: {M} should be the same with num_rows of C")
    # # # end test
    # max_load = 0
    # for i in range(len(bucket_dict)):
    #     active_threads = bucket_num_rows_list[i] * ratio
    #     thread_load = bucket_cost_list[i] / active_threads
    #     if thread_load > max_load:
    #         max_load = thread_load
    
    # print("")
    # print(F"max_bucket_width: {max_bucket_width}")
    # print_table(columns=stats)
    # print(F"total_memory_cost: {sum(stats['bucket_memory_cost'])} total_2IK: {sum(stats['2IK'])} total_KJ: {sum(stats['KJ'])} total_IJ: {sum(stats['IJ'])}")
    # print(F"total_memory_cost: {sum(stats['bucket_memory_cost'])}")
    return_dict = {
        "cost": cost,
        "cost1": cost1,
        "cost2": cost2,
        "cost3": cost3,
        "load_imbalance_penalty": load_imbalance_penalty
    }

    # return cost, cost1, cost2, cost3
    # return max_load, cost1, cost2, cost3
    return return_dict


# ################# #
# ----------------- #
# End CELL Format
# ----------------- #
# ################# #


# ################# #
# ----------------- #
# BSR Format
# ----------------- #
# ################# #

def bsrmm(mb, nb, nnz, blk, feat_size):
    @T.prim_func
    def func(
        a: T.handle,
        b: T.handle,
        c: T.handle,
        indptr: T.handle,
        indices: T.handle,
    ) -> None:
        T.func_attr(
            {"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2}
        )
        I = T.dense_fixed(mb)
        J = T.sparse_variable(I, (nb, nnz), (indptr, indices), "int32")
        J_detach = T.dense_fixed(nb)
        BI = T.dense_fixed(blk)
        BJ = T.dense_fixed(blk)
        F = T.dense_fixed(feat_size)
        A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float16")
        B = T.match_sparse_buffer(b, (J_detach, BJ, F), "float16")
        C = T.match_sparse_buffer(c, (I, BI, F), "float16")

        with T.iter([I, BI, BJ, F, J], "SSRSR", "bsrmm") as [
            i,
            bi,
            bj,
            f,
            j,
        ]:
            with T.init():
                C[i, bi, f] = T.float16(0.0)
            C[i, bi, f] = C[i, bi, f] + A[i, j, bi, bj] * B[j, bj, f]

    return func


@T.prim_func
def wmma_sync_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )

    with T.block("root"):
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = (
                    C_frag[vii, vjj] + A_frag[vii, vkk] * B_frag[vkk, vjj]
                )


@T.prim_func
def wmma_sync_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )

    with T.block("root"):
        T.reads(
            [
                C_frag[0:16, 0:16],
                A_frag[0:16, 0:16],
                B_frag[0:16, 0:16],
            ]
        )
        T.writes(C_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_mma_sync(
                    C_frag.data,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    A_frag.data,
                    A_frag.elem_offset // 256
                    + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    B_frag.data,
                    B_frag.elem_offset // 256
                    + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    C_frag.data,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    dtype="handle",
                )
            )


def wmma_load_a(scope: str):
    @T.prim_func
    def wmma_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
        A = T.match_buffer(
            a, (16, 16), "float16", align=128, offset_factor=1, scope=scope
        )
        A_frag = T.match_buffer(
            a_frag,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope="wmma.matrix_a",
        )

        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(A_frag[0:16, 0:16])
            for i, j in T.grid(16, 16):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    A_frag[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_load_a_impl(a: T.handle, a_frag: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        A = T.match_buffer(
            a,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope=scope,
            strides=[s0, s1],
        )
        A_frag = T.match_buffer(
            a_frag,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope="wmma.matrix_a",
        )

        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(A_frag[0:16, 0:16])
            for tx in T.thread_binding(0, 32, "threadIdx.x"):
                T.evaluate(
                    T.tvm_load_matrix_sync(
                        A_frag.data,
                        16,
                        16,
                        16,
                        A_frag.elem_offset // 256
                        + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                        A.access_ptr("r"),
                        A.strides[0],
                        "row_major",
                        dtype="handle",
                    )
                )

    return wmma_load_a_desc, wmma_load_a_impl


def wmma_load_b(scope: str):
    @T.prim_func
    def wmma_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
        B = T.match_buffer(
            b, (16, 16), "float16", align=128, offset_factor=1, scope=scope
        )
        B_frag = T.match_buffer(
            b_frag,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope="wmma.matrix_b",
        )
        with T.block("root"):
            for i, j in T.grid(16, 16):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    B_frag[vii, vjj] = B[vii, vjj]

    @T.prim_func
    def wmma_load_b_impl(b: T.handle, b_frag: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        B = T.match_buffer(
            b,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope=scope,
            strides=[s0, s1],
        )
        B_frag = T.match_buffer(
            b_frag,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope="wmma.matrix_b",
        )
        with T.block("root"):
            T.reads(B[0:16, 0:16])
            T.writes(B_frag[0:16, 0:16])
            for tx in T.thread_binding(0, 32, "threadIdx.x"):
                T.evaluate(
                    T.tvm_load_matrix_sync(
                        B_frag.data,
                        16,
                        16,
                        16,
                        B_frag.elem_offset // 256
                        + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                        B.access_ptr("r"),
                        B.strides[0],
                        "row_major",
                        dtype="handle",
                    )
                )

    return wmma_load_b_desc, wmma_load_b_impl


@T.prim_func
def wmma_fill_desc(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float16(0)


@T.prim_func
def wmma_fill_impl(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )
    with T.block("root"):
        T.reads([])
        T.writes(C_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_fill_fragment(
                    C_frag.data,
                    16,
                    16,
                    16,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    T.float16(0),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_store_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=1, scope="global"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_store_impl(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )
    C = T.match_buffer(
        c,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="global",
        strides=[s0, s1],
    )
    with T.block("root"):
        T.reads(C_frag[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_store_matrix_sync(
                    C_frag.data,
                    16,
                    16,
                    16,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


WMMA_SYNC = tir.TensorIntrin.register(
    "wmma_sync",
    wmma_sync_desc,
    wmma_sync_impl,
)

WMMA_LOAD_A_SHARED = tir.TensorIntrin.register(
    "wmma_load_a_shared", *wmma_load_a("shared")
)

WMMA_LOAD_A_GLOBAL = tir.TensorIntrin.register(
    "wmma_load_a_global", *wmma_load_a("global")
)

WMMA_LOAD_B_SHARED = tir.TensorIntrin.register(
    "wmma_load_b_shared", *wmma_load_b("shared")
)

WMMA_LOAD_B_GLOBAL = tir.TensorIntrin.register(
    "wmma_load_b_global", *wmma_load_b("global")
)

WMMA_FILL = tir.TensorIntrin.register(
    "wmma_fill",
    wmma_fill_desc,
    wmma_fill_impl,
)

WMMA_STORE = tir.TensorIntrin.register(
    "wmma_store",
    wmma_store_desc,
    wmma_store_impl,
)


def bench_bsrmm(bsr_mat: Any, x: th.Tensor, block_size: int):
    global bsrmm
    mb = bsr_mat.shape[0] // bsr_mat.blocksize[0]
    nb = bsr_mat.shape[1] // bsr_mat.blocksize[1]
    nnzb = bsr_mat.nnz // (block_size**2)
    feat_size = x.shape[1]
    ind = (bsr_mat.indptr[1:] - bsr_mat.indptr[:-1]).nonzero()[0]
    print(bsr_mat.indptr[ind + 1] - bsr_mat.indptr[ind])

    mod = tvm.IRModule.from_expr(bsrmm(mb, nb, nnzb, block_size, feat_size))
    sch = tvm.tir.Schedule(mod)
    sp_iteration = sch.get_sparse_iteration("bsrmm")
    i, bi, bj, f, j = sch.get_sp_iters(sp_iteration)
    sch.sparse_reorder(sp_iteration, [i, j, bi, f, bj])
    mod = lower_sparse_iter(sch.mod)
    sch = tir.Schedule(mod)
    blk_inner = sch.get_block("bsrmm1")
    blk_outer = sch.get_block("bsrmm0")
    j, bi, f, bj = sch.get_loops(blk_inner)
    bio, bii = sch.split(bi, [block_size // 16, 16])
    bjo, bji = sch.split(bj, [block_size // 16, 16])
    foo, foi, fi = sch.split(f, [None, 2, 16])
    sch.reorder(foo, j, bio, foi, bjo, bii, fi, bji)
    sch.unroll(foi)
    (i,) = sch.get_loops(blk_outer)
    sch.bind(i, "blockIdx.x")
    sch.bind(bio, "threadIdx.y")
    sch.bind(foo, "blockIdx.y")
    C_local = sch.cache_write(blk_inner, 0, "wmma.accumulator")
    sch.reverse_compute_at(C_local, foo, True)
    ax0, ax1 = sch.get_loops(C_local)[-2:]
    ax2, ax3 = sch.split(ax1, [None, 16])
    ax0, ax1 = sch.split(ax0, [None, 16])
    sch.reorder(ax0, ax2, ax1, ax3)
    sch.unroll(ax2)
    sch.bind(ax0, "threadIdx.y")
    init_blk = sch.decompose_reduction(blk_inner, j)
    A_local = sch.cache_read(blk_inner, 1, "wmma.matrix_a")
    sch.compute_at(A_local, bio)
    ax0, ax1 = sch.get_loops(A_local)[-2:]
    ax1, ax2 = sch.split(ax1, [None, 16])
    sch.reorder(ax1, ax0, ax2)
    sch.unroll(ax1)
    B_shared = sch.cache_read(blk_inner, 2, "shared")
    sch.compute_at(B_shared, foi)
    B_local = sch.cache_read(blk_inner, 2, "wmma.matrix_b")
    sch.compute_at(B_local, bjo)
    sch.hide_buffer_access(blk_inner, "read", [3])
    sch.tensorize(sch.get_loops(blk_inner)[-3], "wmma_sync")
    sch.tensorize(sch.get_loops(B_local)[-2], "wmma_load_b_shared")
    sch.tensorize(sch.get_loops(A_local)[-2], "wmma_load_a_global")
    sch.tensorize(sch.get_loops(C_local)[-2], "wmma_store")
    sch.tensorize(sch.get_loops(init_blk)[-2], "wmma_fill")
    # schedule B_shared
    ax0, ax1 = sch.get_loops(B_shared)[-2:]
    fused_ax = sch.fuse(ax0, ax1)
    ax0, ax1, ax2, ax3 = sch.split(fused_ax, [None, 2, 32, 4])
    sch.vectorize(ax3)
    sch.bind(ax2, "threadIdx.x")
    sch.bind(ax1, "threadIdx.y")
    sch.unroll(ax0)

    mod = lower_sparse_buffer(sch.mod)

    f = tvm.build(mod["main"], target="cuda")

    ctx = tvm.cuda(0)
    A_indptr = tvm.nd.array(np.copy(bsr_mat.indptr).astype("int32"), device=ctx)
    A_indices = tvm.nd.array(np.copy(bsr_mat.indices).astype("int32"), device=ctx)
    A_data = tvm.nd.array(
        np.copy(bsr_mat.data).reshape(-1).astype("float16"), device=ctx
    )
    X_nd = tvm.nd.array(np.copy(x.reshape(-1)).astype("float16"), device=ctx)
    Y_nd = tvm.nd.array(
        np.zeros((mb * block_size * feat_size), dtype="float16"), device=ctx
    )
    args = [A_data, X_nd, Y_nd, A_indptr, A_indices]
    f(*args)

    avg_time = profile_tvm_ms(f, args)
    print("bsrmm time: \t{:.5f}ms".format(avg_time))
    return avg_time

# ################# #
# ----------------- #
# End BSR Format
# ----------------- #
# ################# #




# ################# #
# ----------------- #
# CSR Format
# ----------------- #
# ################# #

@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    num_tiles: T.int32,
    nnz: T.int32,
    coarsening_factor: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K1 = T.dense_fixed(num_tiles)
    K2 = T.dense_fixed(coarsening_factor)
    K3 = T.dense_fixed(32)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), "float32")
    C = T.match_sparse_buffer(c, (I, K1, K2, K3), "float32")
    with T.iter([I, J, K1, K2, K3], "SRSSS", "csrmm") as [i, j, k1, k2, k3]:
        with T.init():
            C[i, k1, k2, k3] = T.float32(0)
        C[i, k1, k2, k3] = C[i, k1, k2, k3] + A[i, j] * B[j, k1, k2, k3]


def bench_naive(
    g,
    x,
    # y_golden,
    y_ndarray,
    feat_size=128,
    coarsening_factor=2,
    check_correctness=False
):
    indptr, indices, _ = g.adj_tensors("csc")
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    nnz = g.num_edges()
    if feat_size < 64:
        coarsening_factor = 1
    mod = tvm.IRModule.from_expr(csrmm)
    # specialize
    params = mod["main"].params
    param_map = {
        params[5]: m,  # m
        params[6]: n,  # n
        params[7]: feat_size // coarsening_factor // 32,  # num_tiles,
        params[8]: nnz,  # nnz
        params[9]: coarsening_factor,  # coarsening factor
    }

    mod["main"] = mod["main"].specialize(param_map)

    # schedule
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    outer_blk = sch.get_block("csrmm0")
    inner_blk = sch.get_block("csrmm1")
    (i,) = sch.get_loops(outer_blk)
    j, foo, foi, fi = sch.get_loops(inner_blk)
    sch.reorder(foo, fi, j, foi)
    sch.bind(fi, "threadIdx.x")
    sch.bind(foo, "blockIdx.y")
    sch.unroll(foi)
    io, ii = sch.split(i, [None, 8])
    sch.bind(io, "blockIdx.x")
    sch.bind(ii, "threadIdx.y")
    init_blk = sch.decompose_reduction(inner_blk, fi)
    ax0, ax1 = sch.get_loops(init_blk)[-2:]
    sch.bind(ax0, "threadIdx.x")
    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    f = tvm.build(mod["main"], target="cuda")
    # prepare nd array
    # indptr_nd = tvm.nd.array(indptr.numpy().astype("int32"), device=tvm.cuda(0))
    indptr_nd = tvm.nd.array(indptr.astype("int32"), device=tvm.cuda(0))
    b_nd = tvm.nd.array(
        x.numpy().reshape(-1).astype("float32"),
        device=tvm.cuda(0),
    )
    # indices_nd = tvm.nd.array(indices.numpy().astype("int32"), device=tvm.cuda(0))
    indices_nd = tvm.nd.array(indices.astype("int32"), device=tvm.cuda(0))
    c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype("float32"), device=tvm.cuda(0))
    a_nd = tvm.nd.array(np.ones((nnz,)).astype("float32"), device=tvm.cuda(0))
    args = [a_nd, b_nd, c_nd, indptr_nd, indices_nd]
    f(*args)
    if check_correctness:
        # tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)
        tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_ndarray, rtol=1e-4)
    dur = profile_tvm_ms(f, args)
    print("tir naive time: {:.5f} ms".format(dur))

    return dur
# ################# #
# ----------------- #
# End CSR Format
# ----------------- #
# ################# #