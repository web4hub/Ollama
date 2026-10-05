package renderers

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// ThinkingForRenderer returns the controls of the selected renderer variant.
// Unknown or invalid descriptors are omitted.
func ThinkingForRenderer(name string) *model.Thinking {
	if name == "harmony" {
		return &model.Thinking{
			Values:  []any{"low", "medium", "high"},
			Default: "medium",
		}
	}
	r := rendererForName(name)
	if r == nil {
		return nil
	}
	thinking := r.Thinking()
	if !thinking.Valid() {
		return nil
	}
	return thinking.Clone()
}

// ResolveThinking preserves explicit booleans and supported names. An
// unsupported "none" turns thinking off when the model supports that, matching
// OpenAI's reasoning_effort. Omission or another unsupported name uses the
// default. Unknown metadata preserves legacy handling.
func ResolveThinking(requestedThink *api.ThinkValue, thinking *model.Thinking) *api.ThinkValue {
	if !thinking.Valid() {
		return requestedThink
	}
	if requestedThink != nil {
		switch value := requestedThink.Value.(type) {
		case bool:
			return requestedThink
		case string:
			if thinking.Supports(value) {
				return requestedThink
			}
			if value == "none" && thinking.Supports(false) {
				return &api.ThinkValue{Value: false}
			}
		}
	}
	return &api.ThinkValue{Value: thinking.Default}
}
