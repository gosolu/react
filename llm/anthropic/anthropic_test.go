package anthropic_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/llm/anthropic"
)

func TestAnthropicGenerate(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "ant-key" {
			t.Errorf("missing or bad x-api-key: %v", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Errorf("missing anthropic-version header")
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{
			"id": "msg_123",
			"type": "message",
			"role": "assistant",
			"content": [
				{"type": "text", "text": "I will check."},
				{"type": "tool_use", "id": "tu_1", "name": "weather", "input": {"city": "Paris"}}
			],
			"stop_reason": "tool_use",
			"usage": {
				"input_tokens": 15,
				"output_tokens": 8
			}
		}`)
	}))
	defer ts.Close()

	p := anthropic.New(anthropic.Config{
		APIKey:  "ant-key",
		BaseURL: ts.URL,
	})

	resp, err := p.Generate(context.Background(), &llm.Request{
		Model:    "claude-3-5-sonnet-20241022",
		Messages: []llm.Message{llm.UserText("weather in Paris?")},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "I will check." {
		t.Fatalf("text = %q", resp.Message.Text())
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Name != "weather" {
		t.Fatalf("unexpected tool calls: %+v", resp.Message.ToolCalls)
	}
	if resp.StopReason != llm.StopToolUse {
		t.Fatalf("stop reason = %v", resp.StopReason)
	}
	if resp.Usage.InputTokens != 15 || resp.Usage.OutputTokens != 8 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestAnthropicStream(t *testing.T) {
	sseData := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","usage":{"input_tokens":20,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Bonjour"}}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"echo"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"msg\":\"hi\"}"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}

event: message_stop
data: {"type":"message_stop"}

`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := anthropic.New(anthropic.Config{
		APIKey:  "ant-key",
		BaseURL: ts.URL,
	})

	var acc llm.Accumulator
	for ev, err := range p.Stream(context.Background(), &llm.Request{Model: "claude-3-5-sonnet"}) {
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(ev)
	}

	resp := acc.Response()
	if resp.Message.Text() != "Bonjour" {
		t.Fatalf("text = %q, want Bonjour", resp.Message.Text())
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Name != "echo" {
		t.Fatalf("calls = %+v", resp.Message.ToolCalls)
	}
	if !strings.Contains(string(resp.Message.ToolCalls[0].Arguments), `"msg":"hi"`) {
		t.Fatalf("args = %s", string(resp.Message.ToolCalls[0].Arguments))
	}
	if resp.StopReason != llm.StopToolUse {
		t.Fatalf("stop = %v", resp.StopReason)
	}
	if resp.Usage.InputTokens != 20 || resp.Usage.OutputTokens != 12 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}
