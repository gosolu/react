package openai_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/llm/openai"
)

func TestOpenAIGenerate(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing or bad auth header: %v", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{
			"choices": [{
				"message": {
					"content": "Hello world",
					"tool_calls": [{
						"id": "tc1",
						"function": {
							"name": "lookup",
							"arguments": "{\"q\":\"go\"}"
						}
					}]
				},
				"finish_reason": "tool_calls"
			}],
			"usage": {
				"prompt_tokens": 12,
				"completion_tokens": 6
			}
		}`)
	}))
	defer ts.Close()

	p := openai.New(openai.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	resp, err := p.Generate(context.Background(), &llm.Request{
		Model:    "gpt-4o",
		Messages: []llm.Message{llm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "Hello world" {
		t.Fatalf("unexpected text: %q", resp.Message.Text())
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Name != "lookup" {
		t.Fatalf("unexpected tool calls: %+v", resp.Message.ToolCalls)
	}
	if resp.StopReason != llm.StopToolUse {
		t.Fatalf("stop reason = %v", resp.StopReason)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 6 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestOpenAIStream(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"calc","arguments":"{\"a\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}

data: {"choices":[{"finish_reason":"tool_calls"}]}

data: {"usage":{"prompt_tokens":10,"completion_tokens":5}}

data: [DONE]

`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := openai.New(openai.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	var acc llm.Accumulator
	for ev, err := range p.Stream(context.Background(), &llm.Request{Model: "gpt-4o"}) {
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(ev)
	}

	resp := acc.Response()
	if resp.Message.Text() != "Hello" {
		t.Fatalf("text = %q, want Hello", resp.Message.Text())
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Name != "calc" {
		t.Fatalf("calls = %+v", resp.Message.ToolCalls)
	}
	if !strings.Contains(string(resp.Message.ToolCalls[0].Arguments), "1}") {
		t.Fatalf("tool args = %s", string(resp.Message.ToolCalls[0].Arguments))
	}
	if resp.StopReason != llm.StopToolUse {
		t.Fatalf("stop = %v", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}
