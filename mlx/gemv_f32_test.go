package mlx

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestMatmulF32OutMatchesFloat64(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		for _, dtype := range []DType{DTypeBFloat16, DTypeFloat16, DTypeFloat32} {
			for _, shape := range []struct {
				lead  []int
				N, K  int
				fused bool
			}{
				{[]int{1, 1}, 100, 64, true},   // N not a multiple of the row tile
				{[]int{1, 1}, 384, 2560, true}, // router-shaped
				{[]int{1, 1}, 40000, 72, true}, // wide path, K not a multiple of 256
				{[]int{3}, 33, 136, true},      // several rows
				{[]int{2, 8}, 50, 64, true},    // gemvF32MaxRows rows
				{[]int{1, 17}, 50, 64, false},  // too many rows: graph path
				{[]int{1, 1}, 40, 30, false},   // K not a multiple of 8: graph path
			} {
				name := fmt.Sprintf("%v_%v_n%d_k%d", dtype, shape.lead, shape.N, shape.K)
				x := patternArray(dtype, append(append([]int(nil), shape.lead...), shape.K), 0.5, 0.03, 13, 97)
				w := patternArray(dtype, []int{shape.N, shape.K}, -0.2, 0.01, 7, 131)
				got := MatmulF32Out(x, w)
				Eval(got)
				if got.DType() != DTypeFloat32 {
					t.Fatalf("%s: dtype %v", name, got.DType())
				}
				xs, ws, gs := x.AsType(DTypeFloat32).Floats(), w.AsType(DTypeFloat32).Floats(), got.Floats()
				rows := len(xs) / shape.K
				if len(gs) != rows*shape.N {
					t.Fatalf("%s: %d outputs, want %d", name, len(gs), rows*shape.N)
				}
				// The fused kernel rounds only its float32 accumulation; the
				// graph path's float32 matmul may run as TF32 on newer GPUs.
				tol := 1e-6
				if !shape.fused {
					tol = 5e-3
				}
				for m := range rows {
					for n := range shape.N {
						var want, mag float64
						for k := range shape.K {
							p := float64(xs[m*shape.K+k]) * float64(ws[n*shape.K+k])
							want += p
							mag += math.Abs(p)
						}
						if d := math.Abs(float64(gs[m*shape.N+n]) - want); d > tol*mag+1e-7 {
							t.Fatalf("%s: [%d,%d] %g want %g (|terms| %g)", name, m, n, gs[m*shape.N+n], want, mag)
						}
					}
				}
			}
		}
		if MetalIsAvailable() && gemvF32.metalDisabled {
			t.Fatal("Metal kernel disabled itself; the fused path never ran")
		}
	})
}
