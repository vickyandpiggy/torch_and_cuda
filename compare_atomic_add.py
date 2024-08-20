import triton
import triton.language as tl
import torch
import math
from torch.utils.cpp_extension import load_inline

cuda_source = '''
__global__ void scatter_add_kernel(int* dw, int block_m, int block_n,
                                   int max_seq_len, long long stride_dws,
                                   long long stride_dwzh) {
  int w_len = 2 * max_seq_len - 1;
  int thread_idx = threadIdx.x * block_n + threadIdx.y;
  extern __shared__ int w[];
  if (thread_idx < w_len) {
    w[thread_idx] = 0;
  }
  __syncthreads();
  int start_n = blockIdx.x * block_n;
  int off_n = start_n + threadIdx.y;
  int low = start_n;
  int off_m;
  int idx;
  for (int start_m = low; start_m < max_seq_len; start_m += block_m) {
    off_m = start_m + threadIdx.x;
    idx = off_n - off_m + max_seq_len - 1;
    atomicAdd(w + idx, 1);
  }
  __syncthreads();
  if (thread_idx < w_len) {
    dw[blockIdx.x * stride_dws + blockIdx.y * stride_dwzh + thread_idx] =
        w[thread_idx];
  }
}

torch::Tensor scatter_add(torch::Tensor input, int block_m, int block_n,
                          int max_seq_len, int num_seq_block, int num_z_h,
                          long long stride_dws, long long stride_dwzh) {
  dim3 threads_per_block(block_m, block_n);
  dim3 number_of_blocks(num_seq_block, num_z_h);

  scatter_add_kernel<<<number_of_blocks, threads_per_block,
                       (2 * max_seq_len + 1) * sizeof(int)>>>(
      input.data_ptr<int>(), block_m, block_n, max_seq_len, stride_dws,
      stride_dwzh);

  return input;
}
'''

cpp_source = '''
            torch::Tensor scatter_add(torch::Tensor input, int block_m, int block_n,
                          int max_seq_len, int num_seq_block, int num_z_h,
                          long long stride_dws, long long stride_dwzh);
            '''

test_extension = load_inline(
    name='test_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['scatter_add'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./load_inline_cuda',
)


def test_cuda(N):
  BLOCK_M = 16
  BLOCK_N = 64
  Z = 1024
  H = 2
  num_sequence_block = math.ceil(N / BLOCK_N)
  num_z_h = Z * H
  dw = torch.zeros([num_sequence_block, num_z_h, 2 * N - 1],
                   dtype=torch.int32,
                   device='cuda')
  stride_dws = dw.stride(0)
  stride_dwzh = dw.stride(1)
  test_extension.scatter_add(dw, BLOCK_M, BLOCK_N, N, num_sequence_block,
                             num_z_h, stride_dws, stride_dwzh)
  dw = dw.sum(dim=0)
  dw = dw.sum(dim=0)
  return dw


@triton.jit
def _kernel_one_block(
    start_n,
    DW,
    MAX_SEQ_LEN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
  start_n = tl.multiple_of(start_n, BLOCK_N)
  offs_n = start_n + tl.arange(0, BLOCK_N)
  offs_m_base = tl.arange(0, BLOCK_M)
  low = start_n // BLOCK_M * BLOCK_M
  high = MAX_SEQ_LEN
  ds = tl.full([BLOCK_M, BLOCK_N], 1, dtype=tl.int32)
  for start_m in range(low, high, BLOCK_M):
    start_m = tl.multiple_of(start_m, BLOCK_M)
    offs_m = start_m + offs_m_base
    offs_w = offs_n[None, :] - offs_m[:, None] + MAX_SEQ_LEN - 1
    tl.atomic_add(DW + offs_w, ds)


@triton.jit
def _kernel(
    DW,
    MAX_SEQ_LEN,
    stride_dws,
    stride_dwzh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
  off_s = tl.program_id(0).to(tl.int64)
  start_n = off_s * BLOCK_N
  DW += stride_dws * off_s + stride_dwzh * tl.program_id(1).to(tl.int64)
  _kernel_one_block(
      start_n=start_n,
      DW=DW,
      MAX_SEQ_LEN=MAX_SEQ_LEN,
      BLOCK_M=BLOCK_M,
      BLOCK_N=BLOCK_N,
  )


def test_triton(N):
  num_warps = 8
  num_stages = 1
  Z = 1024
  H = 2
  BLOCK_M = 16
  BLOCK_N = 64
  num_sequence_block = math.ceil(N / BLOCK_N)
  num_z_h = Z * H
  dw = torch.zeros([num_sequence_block, num_z_h, 2 * N - 1],
                   dtype=torch.int32,
                   device='cuda')
  grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), num_z_h)
  _kernel[grid](
      DW=dw,
      MAX_SEQ_LEN=N,
      stride_dws=dw.stride(0),
      stride_dwzh=dw.stride(1),
      BLOCK_M=BLOCK_M,
      BLOCK_N=BLOCK_N,
      num_stages=num_stages,
      num_warps=num_warps,
  )
  dw = dw.sum(dim=0)
  dw = dw.sum(dim=0)
  return dw


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in [1, 2, 4, 8, 16]
               ],  # different possible values for `x_name`
        line_arg=
        'provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'triton',
            'cuda',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Cuda",
        ],  # label name for the lines
        styles=[('blue', '-'), ('red', '--')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name=
        "test performance",  # name for the plot. Used also as a file name for saving the plot.
        args={},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(N, provider):
  quantiles = [0.5, 0.2, 0.8]
  if provider == 'triton':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: test_triton(N),
                                                 quantiles=quantiles)
  elif provider == 'cuda':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: test_cuda(N),
                                                 quantiles=quantiles)
  return ms


benchmark.run(show_plots=True, print_data=True, save_path='.')

triton_dw = test_triton(512)
cuda_dw = test_cuda(512)
print(triton_dw.max())
print(cuda_dw.max())
comp = torch.abs(triton_dw - cuda_dw) < 0.01
print((comp).sum(dim=0))
false_indices = torch.nonzero(comp == False)
print(false_indices)
