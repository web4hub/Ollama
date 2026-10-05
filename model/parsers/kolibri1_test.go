package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestKolibri1Streaming(t *testing.T) {
	input := "<think>\nCompute.\n</think>\n\n42."
	// Kolibri emits <think> as a single token, so the newline after it
	// arrives in a later chunk.
	chunkings := [][]string{{"<think>", "\n", "Compute", ".", "\n", "</think>", "\n\n", "42", "."}}
	for _, size := range []int{1, 3, 13, 1000} {
		var chunks []string
		for offset := 0; offset < len(input); offset += size {
			chunks = append(chunks, input[offset:min(offset+size, len(input))])
		}
		chunkings = append(chunkings, chunks)
	}
	for _, chunks := range chunkings {
		p := ParserForName("kolibri1")
		p.Init(nil, nil, nil)
		var content, thinking strings.Builder
		for i, chunk := range chunks {
			c, th, _, err := p.Add(chunk, i == len(chunks)-1)
			if err != nil {
				t.Fatal(err)
			}
			content.WriteString(c)
			thinking.WriteString(th)
		}
		if content.String() != "42." || thinking.String() != "Compute." {
			t.Fatalf("chunks %q: content %q thinking %q", chunks, content.String(), thinking.String())
		}
	}
	p := ParserForName("kolibri1")
	p.Init(nil, nil, &api.ThinkValue{Value: false})
	c, th, _, err := p.Add("42.", true)
	if err != nil || c != "42." || th != "" {
		t.Fatalf("thinking off: %q %q %v", c, th, err)
	}
}

func TestKolibri1ToolsAndContinuation(t *testing.T) {
	for _, size := range []int{1, 7, 1000} {
		p := ParserForName("kolibri1")
		p.Init(nil, nil, &api.ThinkValue{Value: false})
		text := `<tool_call>{"name":"weather","arguments":{"city":"Berlin"}}</tool_call>`
		var calls []api.ToolCall
		for i := 0; i < len(text); i += size {
			end := min(i+size, len(text))
			_, _, chunk, err := p.Add(text[i:end], end == len(text))
			if err != nil {
				t.Fatal(err)
			}
			calls = append(calls, chunk...)
		}
		if len(calls) != 1 || calls[0].Function.Name != "weather" || calls[0].Function.Arguments.ToMap()["city"] != "Berlin" {
			t.Fatalf("tools at chunk %d: %+v", size, calls)
		}
	}
	p := ParserForName("kolibri1")
	p.Init(nil, &api.Message{Role: "assistant", Content: "The answer is"}, nil)
	c, th, _, err := p.Add(" 42.", true)
	if err != nil || c != " 42." || th != "" {
		t.Fatalf("continuation: %q %q %v", c, th, err)
	}
}

// The renderer only continues an assistant message with content. Without
// content it gets a fresh generation prompt, so reasoning stays on.
func TestKolibri1EmptyAssistantIsNotPrefill(t *testing.T) {
	call := api.ToolCall{Function: api.ToolCallFunction{Name: "weather"}}
	for _, last := range []*api.Message{
		{Role: "assistant"},
		{Role: "assistant", ToolCalls: []api.ToolCall{call}},
	} {
		p := ParserForName("kolibri1")
		p.Init(nil, last, nil)
		c, th, _, err := p.Add("<think>\nCompute.\n</think>\n\n42.", true)
		if err != nil || c != "42." || th != "Compute." {
			t.Fatalf("%+v: content %q thinking %q err %v", last, c, th, err)
		}
	}
}
