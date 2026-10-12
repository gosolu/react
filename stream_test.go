package react

import (
	"testing"
)

func TestAccumulatorRoundTrip(t *testing.T) {
	resp := &Response{
		Message: Message{
			Role:    RoleAssistant,
			Content: []Part{TextPart("héllo wörld, let me check")},
			ToolCalls: []ToolCall{
				{ID: "a", Name: "add", Arguments: []byte(`{"a":1,"b":2}`)},
				{ID: "b", Name: "echo", Arguments: []byte(`{"s":"x"}`)},
			},
		},
		Reasoning:  "thinking...",
		StopReason: StopToolCall,
		Usage:      Usage{InputTokens: 7, OutputTokens: 3},
	}
	var acc Accumulator
	for ev, err := range StreamResponse(resp, 3) {
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(ev)
	}
	got := acc.Response()
	if got.Message.Text() != resp.Message.Text() {
		t.Fatalf("text = %q", got.Message.Text())
	}
	if got.Reasoning != resp.Reasoning {
		t.Fatalf("reasoning = %q", got.Reasoning)
	}
	if len(got.Message.ToolCalls) != 2 || string(got.Message.ToolCalls[1].Arguments) != `{"s":"x"}` || got.Message.ToolCalls[0].ID != "a" {
		t.Fatalf("calls = %+v", got.Message.ToolCalls)
	}
	if got.StopReason != StopToolCall || got.Usage != resp.Usage {
		t.Fatalf("stop=%v usage=%+v", got.StopReason, got.Usage)
	}
}

func TestAccumulatorDefaults(t *testing.T) {
	var acc Accumulator
	acc.Add(StreamEvent{Type: StreamToolCallStart, Index: 5, ToolCallID: "x", ToolName: "noargs"})
	r := acc.Response()
	if string(r.Message.ToolCalls[0].Arguments) != "{}" || r.StopReason != StopToolCall {
		t.Fatalf("%+v", r)
	}
}
