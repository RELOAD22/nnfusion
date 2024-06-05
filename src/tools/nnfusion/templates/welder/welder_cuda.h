#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"

namespace cutlass {
namespace gemm {
namespace warp {

template<class MmaWarp, int KSize>
class MMAWarpWrapper {
public:
  typename MmaWarp::FragmentA frag_A[2];
  typename MmaWarp::FragmentB frag_B[2];
  typename MmaWarp::FragmentC accum;
  MmaWarp mma_op;
  typename MmaWarp::IteratorA iter_A;
  typename MmaWarp::IteratorB iter_B;
  const int warp_idx_m_, warp_idx_n_, lane_id_;

  using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
  using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
  static_assert(KSize % MmaWarp::Shape::kK == 0);
  static int constexpr kKgroups = KSize / MmaWarp::Shape::kK;

  CUTLASS_DEVICE
  MMAWarpWrapper(int warp_idx_m, int warp_idx_n, int lane_id)
  : warp_idx_m_(warp_idx_m), warp_idx_n_(warp_idx_n), lane_id_(lane_id), iter_A({nullptr, 0}, 0), iter_B({nullptr, 0}, 0) {
    accum.clear();
  }

  CUTLASS_DEVICE
  void prologue(const TensorRefA &ref_A, const TensorRefB &ref_B) {
    iter_A = typename MmaWarp::IteratorA(ref_A, lane_id_);
    iter_B = typename MmaWarp::IteratorB(ref_B, lane_id_);
    iter_A.add_tile_offset({warp_idx_m_, 0});
    iter_B.add_tile_offset({0, warp_idx_n_});
    iter_A.load(frag_A[0]);
    iter_B.load(frag_B[0]);
    ++iter_A;
    ++iter_B;
  }
  CUTLASS_DEVICE
  void body() {
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups - 1; ++k) {
      iter_A.load(frag_A[(k + 1) % 2]);
      iter_B.load(frag_B[(k + 1) % 2]);
      ++iter_A;
      ++iter_B;
      mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
    }
    __syncthreads();
  }
  CUTLASS_DEVICE
  void epilogue() {
    mma_op(accum, frag_A[(kKgroups - 1) % 2], frag_B[(kKgroups - 1) % 2], accum);
  }
};

template <
  typename Shape,
  typename SMemLayoutA,
  typename SMemLayoutB
>
class GemmTensorOp {
public:
  using InstructionShape = GemmShape<16, 8, 16>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::ColumnMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
    cutlass::half_t,
    SMemLayoutA,
    cutlass::half_t,
    SMemLayoutB,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    Policy
  >;
  using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
  MMA mma;

  CUTLASS_DEVICE
  GemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
  : mma(warp_idx_m, warp_idx_n, lane_id) {}
  CUTLASS_DEVICE
  half& operator[](size_t i) const {
    return ((half*)mma.accum.data())[i];
  }
  CUTLASS_DEVICE
  half* operator+(size_t i) const {
    return (half*)mma.accum.data() + i;
  }
};

template <
  typename Shape,
  typename SMemLayoutA,
  typename LayoutA,
  typename SMemLayoutB,
  typename LayoutB,
  typename LayoutC
>
class VoltaGemmTensorOp {
public:
  using InstructionShape = GemmShape<16, 16, 4>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      cutlass::half_t,
      LayoutA,
      cutlass::half_t,
      LayoutB,
      cutlass::half_t,
      LayoutC,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaVoltaTensorOp<
    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
    cutlass::half_t,
    SMemLayoutA,
    cutlass::half_t,
    SMemLayoutB,
    cutlass::half_t,
    LayoutC,
    Policy
  >;
  using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
  MMA mma;

  CUTLASS_DEVICE
  VoltaGemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
  : mma(warp_idx_m, warp_idx_n, lane_id) {}
  CUTLASS_DEVICE
  half& operator[](size_t i) const {
    return ((half*)mma.accum.data())[i];
  }
  CUTLASS_DEVICE
  half* operator+(size_t i) const {
    return (half*)mma.accum.data() + i;
  }
};

template <
  typename Shape,
  typename SMemLayoutA,
  typename SMemLayoutB
>
class GemmI8TensorOp {
public:
  using InstructionShape = GemmShape<16, 8, 32>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      int8_t,
      cutlass::layout::RowMajor,
      int8_t,
      cutlass::layout::ColumnMajor,
      int,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
    int8_t,
    SMemLayoutA,
    int8_t,
    SMemLayoutB,
    int,
    cutlass::layout::RowMajor,
    Policy
  >;
  using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
  MMA mma;

  CUTLASS_DEVICE
  GemmI8TensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
  : mma(warp_idx_m, warp_idx_n, lane_id) {}
  CUTLASS_DEVICE
  int& operator[](size_t i) const {
    return ((int*)mma.accum.data())[i];
  }
  CUTLASS_DEVICE
  int* operator+(size_t i) const {
    return (int*)mma.accum.data() + i;
  }
};

}}}

template<class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_body(TensorOp& op) {
  op.mma.body();
}

template<class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_epilogue(TensorOp& op) {
  op.mma.epilogue();
}

template<class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_prologue(TensorOp& op, void* pA, void* pB, int sA, int sB) {
  using TensorRefA = typename TensorOp::MMA::TensorRefA;
  using TensorRefB = typename TensorOp::MMA::TensorRefB;
  TensorRefA refA{(typename TensorRefA::Element*)pA, sA};
  TensorRefB refB{(typename TensorRefB::Element*)pB, sB};
  op.mma.prologue(refA, refB);
}

#define ALLOCATE_CUTLASS_OBJECT(var, ...) auto var = __VA_ARGS__;

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

inline __device__ longlong4 make_int8(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7) {
  int2 i0 = make_int2(x0, x1);
  int2 i1 = make_int2(x2, x3);
  int2 i2 = make_int2(x4, x5);
  int2 i3 = make_int2(x6, x7);
  long long l0 = *(long long*)&i0;
  long long l1 = *(long long*)&i1;
  long long l2 = *(long long*)&i2;
  long long l3 = *(long long*)&i3;
  return make_longlong4(l0, l1, l2, l3);
}

inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}
#ifndef __HALF_COMPARE_EX__
#define __HALF_COMPARE_EX__
inline __device__ half max(half x, half y) { return x > y ? x : y; }
inline __device__ half min(half x, half y) { return x < y ? x : y; }
#endif

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x, half y) {                   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x) {                          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

template<int row_size, int col_size, int panel_width>
__device__ int rasterization2DRow(int idx) {
  const int block_size = row_size * col_size;
  const int panel_size = panel_width * col_size;
  const int block_offset = idx % block_size;
  const int block_idx = idx / block_size;
  const int panel_offset = block_offset % panel_size;
  const int panel_idx = block_offset / panel_size;
  const int total_panel = (block_size + panel_size - 1) / panel_size;
  const int stride = panel_idx + 1 < total_panel ? panel_width : (block_size - panel_idx * panel_size) / col_size;
  const int col_idx = (panel_idx & 1) ? col_size - 1 - panel_offset / stride : panel_offset / stride;
  const int row_idx = panel_offset % stride + panel_idx * panel_width;
  return block_idx * block_size + row_idx * col_size + col_idx;
}

template<int row_size, int col_size, int panel_width>
__device__ int rasterization2DColumn(int idx) {
  const int block_size = row_size * col_size;
  const int panel_size = panel_width * row_size;
  const int block_offset = idx % block_size;
  const int block_idx = idx / block_size;
  const int panel_offset = block_offset % panel_size;
  const int panel_idx = block_offset / panel_size;
  const int total_panel = (block_size + panel_size - 1) / panel_size;
  const int stride = panel_idx + 1 < total_panel ? panel_width : (block_size - panel_idx * panel_size) / row_size;
  const int row_idx = (panel_idx & 1) ? row_size - 1 - panel_offset / stride : panel_offset / stride;
  const int col_idx = panel_offset % stride + panel_idx * panel_width;
  return block_idx * block_size + row_idx * col_size + col_idx;
}