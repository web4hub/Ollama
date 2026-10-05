package qwen4_exp

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
)

// Quantizes the routed and shared experts the way create writes nvfp4 (block
// scales normalized by the tensor's amax, the multiplier stored as a
// ".global_scale" companion) and checks Forward against the same weights
// dequantized into a dense block, on both the unsorted decode and sorted
// prefill gather paths.
func TestSparseMoENVFP4GlobalScales(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() && !mlx.CUDAIsAvailable() {
			t.Skip("gather_qmm requires a GPU backend")
		}

		const experts, hidden, inter = 4, 64, 32
		cfg := &Config{HiddenSize: hidden, NumExperts: experts, NumExpertsPerTok: 2}
		m := &Model{Config: cfg, quantGroup: 16, quantBits: 4, quantMode: "nvfp4"}
		prefix := "model.language_model.layers.0.mlp"

		quantized, dense := map[string]*mlx.Array{}, map[string]*mlx.Array{}
		put := func(name string, seed uint64, scale float32, quantize bool, dims ...int) {
			n := 1
			for _, dim := range dims {
				n *= dim
			}
			w := mlx.FromValues(moeTestValues(seed, n, scale), dims...)
			if !quantize {
				quantized[name], dense[name] = w, w
				return
			}
			amax := mlx.Flatten(w.Abs()).MaxAxis(0, false).AsType(mlx.DTypeFloat32)
			wq, sc, _ := mlx.QuantizeWithGlobalScale(w, 16, 4, "nvfp4", amax)
			globalScale := mlx.DivScalar(amax, mlx.Nvfp4MaxProduct)
			quantized[name], quantized[name+"_scale"], quantized[name+".global_scale"] = wq, sc, globalScale
			// The companion is a multiplier on the block-scaled weights.
			dense[name] = mlx.Mul(mlx.Dequantize(wq, sc, nil, 16, 4, "nvfp4", nil).AsType(mlx.DTypeFloat32), globalScale)
		}
		// Distinct magnitudes so a scale applied to the wrong projection shows.
		put(prefix+".gate.weight", 1, 0.1, false, experts, hidden)
		put(prefix+".experts.gate_up_proj", 2, 0.25, true, experts, 2*inter, hidden)
		put(prefix+".experts.down_proj", 3, 0.05, true, experts, hidden, inter)
		put(prefix+".shared_expert_gate.weight", 4, 0.1, false, 1, hidden)
		put(prefix+".shared_expert.gate_proj.weight", 5, 0.5, true, inter, hidden)
		put(prefix+".shared_expert.up_proj.weight", 6, 0.1, true, inter, hidden)
		put(prefix+".shared_expert.down_proj.weight", 7, 0.2, true, hidden, inter)

		load := func(tensors map[string]*mlx.Array) *sparseMoE {
			linears := model.NewLinearFactory(tensors, m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant)
			moe, err := m.loadSparseMoE(linears, tensors, prefix)
			if err != nil {
				t.Fatal(err)
			}
			return moe
		}
		moe, ref := load(quantized), load(dense)
		if moe.GateUpScales == nil || moe.DownScales == nil || ref.GateUpScales != nil || ref.DownScales != nil {
			t.Fatal("expert banks did not load onto the quantized and dense gather paths")
		}
		for _, bank := range []*mlx.Array{moe.GateUpGlobalScales, moe.DownGlobalScales} {
			if bank == nil || !slices.Equal(bank.Dims(), []int{experts}) {
				t.Fatal("expert banks did not load one global scale per expert")
			}
		}

		// 64 tokens take the sorted gather path.
		for _, length := range []int{1, 64} {
			x := mlx.FromValues(moeTestValues(8, length*hidden, 1), 1, length, hidden)
			got, want := moe.Forward(x, cfg), ref.Forward(x, cfg)
			mlx.Eval(got, want)
			gotValues, wantValues := got.Floats(), want.Floats()
			var maxDiff, maxWant float64
			for i, v := range gotValues {
				maxDiff = max(maxDiff, math.Abs(float64(v-wantValues[i])))
				maxWant = max(maxWant, math.Abs(float64(wantValues[i])))
			}
			// M5 GPUs run float32 matmuls as TF32 by default, which leaves
			// ~0.3% at prefill; a dropped global scale is off by orders of
			// magnitude.
			if math.IsNaN(maxDiff) || maxDiff > 1e-2*maxWant {
				t.Errorf("length=%d: quantized output differs from dense by %g (max |want| %g)", length, maxDiff, maxWant)
			}
		}
	})
}

func moeTestValues(seed uint64, n int, scale float32) []float32 {
	r := rand.New(rand.NewPCG(seed, 0))
	out := make([]float32, n)
	for i := range out {
		out[i] = scale * (2*r.Float32() - 1)
	}
	return out
}
