package mlx

// gemvF32 computes out[m, n] = sum_k x[m, k] * w[n, k] with float32
// accumulation and output. Each simdgroup owns ROWS output rows and keeps its
// slice of x in registers across them; lanes read eight consecutive weights
// of a row at a time, so a simdgroup streams 256 contiguous weights per row.
const gemvF32MetalSource = `
uint lane = thread_position_in_threadgroup.x;
uint simd = thread_position_in_threadgroup.y;
uint m = threadgroup_position_in_grid.z;
uint row0 = (threadgroup_position_in_grid.y * SIMDS + simd) * ROWS;
size_t xbase = size_t(m) * K;
float acc[ROWS];
for (int r = 0; r < ROWS; ++r) {
  acc[r] = 0.0f;
}
for (uint k = lane * 8; k < uint(K); k += 256) {
  float xv[8];
  for (int j = 0; j < 8; ++j) {
    xv[j] = static_cast<float>(x[xbase + k + j]);
  }
  for (int r = 0; r < ROWS; ++r) {
    uint row = row0 + r;
    if (row < uint(N)) {
      size_t wbase = size_t(row) * K + k;
      for (int j = 0; j < 8; ++j) {
        acc[r] += xv[j] * static_cast<float>(w[wbase + j]);
      }
    }
  }
}
for (int r = 0; r < ROWS; ++r) {
  float s = simd_sum(acc[r]);
  uint row = row0 + r;
  if (lane == 0 && row < uint(N)) {
    out[size_t(m) * N + row] = s;
  }
}
`

// gemvF32MaxRows bounds the rows of x the fused kernel takes: it streams the
// whole weight once per row, which only wins for vector-shaped inputs.
const gemvF32MaxRows = 16

var gemvF32 = &gpuKernel{
	name:    "gemv_f32_out",
	inputs:  []string{"x", "w"},
	outputs: []string{"out"},
	metal:   gpuSource{source: gemvF32MetalSource},
	fallback: func(launch gpuLaunch) []*Array {
		x, w := launch.inputs[0], launch.inputs[1]
		return []*Array{matmulF32OutGraph(x, w)}
	},
}

func matmulF32OutGraph(x, w *Array) *Array {
	return Matmul(x.AsType(DTypeFloat32), Transpose(w.AsType(DTypeFloat32), 1, 0))
}

// MatmulF32Out returns x @ wᵀ for a weight w [N, K], accumulated and stored
// in float32 without materializing a float32 copy of w: a float32 logits head
// or router then reads its weights at their stored width. Up to
// gemvF32MaxRows rows of x with w's dtype run a fused GEMV; anything else
// promotes both operands to float32, which gives the same values.
func MatmulF32Out(x, w *Array) *Array {
	if w == nil || x == nil || w.NumDims() != 2 || x.NumDims() == 0 {
		panic("mlx.MatmulF32Out: need x [..., K] and w [N, K]")
	}
	dims := x.Dims()
	N, K := w.Dim(0), w.Dim(1)
	if dims[len(dims)-1] != K {
		panic("mlx.MatmulF32Out: x and w disagree on K")
	}
	rows := x.Size() / max(K, 1)
	switch {
	case x.DType() != w.DType(),
		x.DType() != DTypeBFloat16 && x.DType() != DTypeFloat16 && x.DType() != DTypeFloat32,
		K%8 != 0, rows == 0, rows > gemvF32MaxRows:
		return matmulF32OutGraph(x, w)
	}

	// Wide outputs give each simdgroup four rows to reuse x; narrow ones,
	// such as a router, spread one row per simdgroup across more cores.
	const simds = 8
	perSimd := 4
	if N < 32768 {
		perSimd = 1
	}
	groups := (N + simds*perSimd - 1) / (simds * perSimd)
	outShape := make([]int32, len(dims))
	for i, d := range dims[:len(dims)-1] {
		outShape[i] = int32(d)
	}
	outShape[len(dims)-1] = int32(N)
	return gemvF32.run(gpuLaunch{
		dtypes:      []gpuDTypeArg{{"InT", x.DType()}},
		ints:        []gpuIntArg{{"K", K}, {"N", N}, {"ROWS", perSimd}, {"SIMDS", simds}},
		outputs:     []gpuOutputSpec{{"GEMV_F32_OUT", outShape, DTypeFloat32}},
		grid:        [3]int{32, groups * simds, rows},
		threadGroup: [3]int{32, simds, 1},
		inputs:      []*Array{x, w},
	})[0]
}
