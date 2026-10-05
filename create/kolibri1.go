package create

import (
	"encoding/json"
	"strings"
)

// Keep the router and sandwich norms in source precision. Routed experts hold
// nearly all of this 78B/3.5B-active model and take the requested format.
// Attention, the shared experts and the vocabulary matrices use eight bits to
// limit error. They are read for every token, so in a 4-bit model they move
// more bytes per token than the six active experts.
type kolibri1ImportTransform struct{}

func newKolibri1ImportTransform(json.RawMessage) (quantizePolicy, error) {
	return kolibri1ImportTransform{}, nil
}

func (kolibri1ImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	base := normalizeQuantType(quantize)
	if base == "" || !strings.HasSuffix(name, ".weight") || len(shape) < 2 || isRoutingGate(name) {
		return ""
	}
	if strings.Contains(name, ".mlp.experts.") {
		return sensitiveType(false, shape, base)
	}
	return sensitiveType(true, shape, base)
}
