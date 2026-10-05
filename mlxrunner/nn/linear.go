package nn

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
)

// LinearLayer is an interface for linear layers (both regular and quantized).
type LinearLayer interface {
	Forward(x *mlx.Array) *mlx.Array
	OutputDim() int32
}

// Linear applies an affine transformation: y = x @ W.T + b
type Linear struct {
	Weight *mlx.Array
	Bias   *mlx.Array
}

func NewLinear(weight *mlx.Array, bias *mlx.Array) *Linear {
	if bias != nil && bias.DType() != weight.DType() {
		bias = bias.AsType(weight.DType())
	}
	return &Linear{Weight: weight, Bias: bias}
}

func (l *Linear) Forward(x *mlx.Array) *mlx.Array {
	w := l.Weight.Transpose(1, 0)
	if l.Bias != nil {
		return l.Bias.Addmm(x, w, 1.0, 1.0)
	}
	return x.Matmul(w)
}

func (l *Linear) OutputDim() int32 {
	return int32(l.Weight.Dim(0))
}

// QuantizedLinear applies an affine transformation using quantized weights.
type QuantizedLinear struct {
	Weight      *mlx.Array // Quantized weight data
	Scales      *mlx.Array // Scale factors for dequantization
	QBiases     *mlx.Array // Quantization biases (nil for nvfp4)
	Bias        *mlx.Array // Layer bias [output_dims] or nil
	GlobalScale *mlx.Array // Per-tensor or per-row global scale for double-scale nvfp4 (nil for standard)
	GroupSize   int
	Bits        int
	Mode        string
}

func NewQuantizedLinear(weight *mlx.Array, bias *mlx.Array, groupSize, bits int, mode string) *QuantizedLinear {
	qw, scales, qbiases := mlx.Quantize(weight, groupSize, bits, mode)
	if qbiases != nil {
		mlx.Eval(qw, scales, qbiases)
	} else {
		mlx.Eval(qw, scales)
	}
	if bias != nil && bias.DType() != weight.DType() {
		bias = bias.AsType(weight.DType())
	}
	return &QuantizedLinear{
		Weight:    qw,
		Scales:    scales,
		QBiases:   qbiases,
		Bias:      bias,
		GroupSize: groupSize,
		Bits:      bits,
		Mode:      mode,
	}
}

func (ql *QuantizedLinear) Forward(x *mlx.Array) *mlx.Array {
	out := ql.matmul(x, ql.GlobalScale)
	if ql.Bias != nil {
		bias := ql.Bias
		if bias.DType() != out.DType() {
			bias = bias.AsType(out.DType())
		}
		out = out.Add(bias)
	}
	return out
}

func (ql *QuantizedLinear) matmul(x, globalScale *mlx.Array) *mlx.Array {
	return mlx.QuantizedMatmul(x, ql.Weight, ql.Scales, ql.QBiases, true,
		ql.GroupSize, ql.Bits, ql.Mode, globalScale)
}

// SwiGLU applies gate and up projections followed by a SwiGLU activation.
// Quantized projections without bias defer their global scales so the scale
// and activation operations can be fused.
func SwiGLU(gate, up LinearLayer, x *mlx.Array) *mlx.Array {
	gateOut, gateScale := forwardDeferScale(gate, x)
	upOut, upScale := forwardDeferScale(up, x)
	return mlx.SwiGLUScaled(gateOut, gateScale, upOut, upScale)
}

func forwardDeferScale(l LinearLayer, x *mlx.Array) (out, pending *mlx.Array) {
	if ql, ok := l.(*QuantizedLinear); ok && ql.GlobalScale != nil && ql.Bias == nil {
		return ql.matmul(x, nil), ql.GlobalScale
	}
	return l.Forward(x), nil
}

func (ql *QuantizedLinear) OutputDim() int32 {
	return int32(ql.Weight.Dim(0))
}

// StackLinears concatenates two linears along the output dimension. Quant
// groups run along the input dimension, so this is exact; per-tensor global
// scales are expanded to per-row so each half keeps its own.
func StackLinears(a, b LinearLayer) (LinearLayer, error) {
	if pa, ok := a.(*Linear); ok {
		pb, ok := b.(*Linear)
		if !ok {
			return nil, fmt.Errorf("stack linears: mixed plain and quantized parts")
		}
		return &Linear{
			Weight: mlx.Concatenate([]*mlx.Array{pa.Weight, pb.Weight}, 0),
			Bias:   concatBias(pa.Bias, int32(pa.Weight.Dim(0)), pb.Bias, int32(pb.Weight.Dim(0))),
		}, nil
	}
	qa, ok := a.(*QuantizedLinear)
	if !ok {
		return nil, fmt.Errorf("stack linears: unsupported layer type %T", a)
	}
	qb, ok := b.(*QuantizedLinear)
	if !ok {
		return nil, fmt.Errorf("stack linears: mixed plain and quantized parts")
	}
	if qa.GroupSize != qb.GroupSize || qa.Bits != qb.Bits || qa.Mode != qb.Mode {
		return nil, fmt.Errorf("stack linears: quant mode mismatch %s/%d/%d vs %s/%d/%d",
			qa.Mode, qa.Bits, qa.GroupSize, qb.Mode, qb.Bits, qb.GroupSize)
	}
	if (qa.QBiases == nil) != (qb.QBiases == nil) {
		return nil, fmt.Errorf("stack linears: quant bias layout mismatch")
	}
	out := &QuantizedLinear{
		Weight:    mlx.Concatenate([]*mlx.Array{qa.Weight, qb.Weight}, 0),
		Scales:    mlx.Concatenate([]*mlx.Array{qa.Scales, qb.Scales}, 0),
		GroupSize: qa.GroupSize,
		Bits:      qa.Bits,
		Mode:      qa.Mode,
	}
	if qa.QBiases != nil {
		out.QBiases = mlx.Concatenate([]*mlx.Array{qa.QBiases, qb.QBiases}, 0)
	}
	out.Bias = concatBias(qa.Bias, int32(qa.Scales.Dim(0)), qb.Bias, int32(qb.Scales.Dim(0)))
	if qa.GlobalScale != nil || qb.GlobalScale != nil {
		out.GlobalScale = mlx.Concatenate([]*mlx.Array{
			perRowGlobal(qa.GlobalScale, int32(qa.Scales.Dim(0))),
			perRowGlobal(qb.GlobalScale, int32(qb.Scales.Dim(0))),
		}, 0)
	}
	return out, nil
}

// perRowGlobal expands a per-tensor global scale to a per-row vector; an
// already per-row scale passes through unchanged. A nil scale fills with the
// identity, which in MLX's representation is Nvfp4MaxProduct rather than 1.
func perRowGlobal(g *mlx.Array, rows int32) *mlx.Array {
	identity := make([]float32, rows)
	for i := range identity {
		identity[i] = mlx.Nvfp4MaxProduct
	}
	v := mlx.FromValues(identity, int(rows))
	if g == nil {
		return v
	}
	return mlx.Mul(mlx.DivScalar(v, mlx.Nvfp4MaxProduct), g)
}

func concatBias(a *mlx.Array, aRows int32, b *mlx.Array, bRows int32) *mlx.Array {
	if a == nil && b == nil {
		return nil
	}
	fill := func(bias *mlx.Array, rows int32, like *mlx.Array) *mlx.Array {
		if bias != nil {
			return bias
		}
		return mlx.ZerosF32([]int32{rows}).AsType(like.DType())
	}
	if a == nil {
		a = fill(nil, aRows, b)
	}
	if b == nil {
		b = fill(nil, bRows, a)
	}
	return mlx.Concatenate([]*mlx.Array{a, b}, 0)
}
