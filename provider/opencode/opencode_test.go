package opencode

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider"
)

func TestOpenCodeHeaders(t *testing.T) {
	var capturedHeader http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeader = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{"choices":[{"message":{"role":"assistant","content":"Hello from OpenCode Go"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	p := New(provider.Config{
		APIKey:  "test-opencode-key",
		BaseURL: srv.URL,
	})

	if p.Name() != "opencode-go" {
		t.Fatalf("expected provider name opencode-go, got %s", p.Name())
	}

	req := &react.Request{
		Model:    ModelDeepSeekV4Pro,
		Messages: []react.Message{react.UserText("hi")},
	}

	ctx := react.WithRunContext(context.Background(), &react.RunContext{SessionID: "session-ctx-123"})
	_, err := p.Generate(ctx, req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if got := capturedHeader.Get("User-Agent"); got != DefaultUserAgent {
		t.Errorf("expected User-Agent %q, got %q", DefaultUserAgent, got)
	}
	if got := capturedHeader.Get(HeaderSession); got != "session-ctx-123" {
		t.Errorf("expected x-opencode-session %q, got %q", "session-ctx-123", got)
	}
	if got := capturedHeader.Get("Authorization"); got != "Bearer test-opencode-key" {
		t.Errorf("expected Authorization %q, got %q", "Bearer test-opencode-key", got)
	}
	if got := capturedHeader.Get("Content-Type"); got != "application/json" {
		t.Errorf("expected Content-Type application/json, got %q", got)
	}
}

func TestOpenCodeSessionResolution(t *testing.T) {
	var capturedSession string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedSession = r.Header.Get(HeaderSession)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{"choices":[{"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	p := New(provider.Config{BaseURL: srv.URL})

	// 1. RunContext SessionID should be used when present
	ctx := react.WithRunContext(context.Background(), &react.RunContext{SessionID: "run-ctx-session"})
	_, _ = p.Generate(ctx, &react.Request{Model: ModelKimiK3})
	if capturedSession != "run-ctx-session" {
		t.Errorf("expected run-ctx-session, got %q", capturedSession)
	}

	// 2. Default: generated UUID fallback when RunContext is absent
	_, _ = p.Generate(context.Background(), &react.Request{Model: ModelKimiK3})
	if capturedSession == "" {
		t.Errorf("expected generated UUID session, got empty")
	}
	firstGen := capturedSession
	// Verify it remains stable across requests
	_, _ = p.Generate(context.Background(), &react.Request{Model: ModelKimiK3})
	if capturedSession != firstGen {
		t.Errorf("expected session to remain stable %q, got %q", firstGen, capturedSession)
	}
}

func TestOpenCodeGenerate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"choices": [{
				"message": {
					"role": "assistant",
					"content": "Running tool",
					"tool_calls": [{
						"id": "call_123",
						"function": {
							"name": "read_file",
							"arguments": "{\"path\":\"main.go\"}"
						}
					}]
				},
				"finish_reason": "tool_calls"
			}],
			"usage": {
				"prompt_tokens": 42,
				"completion_tokens": 18
			}
		}`)
	}))
	defer srv.Close()

	p := New(provider.Config{BaseURL: srv.URL})

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    ModelDeepSeekV4Pro,
		Messages: []react.Message{react.UserText("inspect main.go")},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp.Message.Text() != "Running tool" {
		t.Errorf("expected text 'Running tool', got %q", resp.Message.Text())
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Message.ToolCalls))
	}
	if resp.Message.ToolCalls[0].ID != "call_123" {
		t.Errorf("expected tool call ID call_123, got %s", resp.Message.ToolCalls[0].ID)
	}
	if resp.StopReason != react.StopToolCall {
		t.Errorf("expected StopToolCall, got %s", resp.StopReason)
	}
	if resp.Usage.InputTokens != 42 || resp.Usage.OutputTokens != 18 {
		t.Errorf("unexpected usage: %+v", resp.Usage)
	}
}

func TestOpenCodeGenerateReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"choices": [{
				"message": {
					"role": "assistant",
					"content": "Result is 42.",
					"reasoning_content": "Thinking about life, the universe, and everything..."
				},
				"finish_reason": "stop"
			}]
		}`)
	}))
	defer srv.Close()

	p := New(provider.Config{BaseURL: srv.URL})
	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    ModelDeepSeekV4Pro,
		Messages: []react.Message{react.UserText("answer")},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp.Reasoning != "Thinking about life, the universe, and everything..." {
		t.Errorf("unexpected reasoning: %q", resp.Reasoning)
	}
	if resp.Message.Text() != "Result is 42." {
		t.Errorf("unexpected content: %q", resp.Message.Text())
	}
}

func TestOpenCodeGenerateVision(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{"choices":[{"message":{"role":"assistant","content":"I see a chart and a logo."},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	p := New(provider.Config{BaseURL: srv.URL})
	req := &react.Request{
		Model: ModelDeepSeekV4FlashVision,
		Messages: []react.Message{
			{
				Role: react.RoleUser,
				Content: []react.Part{
					react.TextPart("What is in these images?"),
					react.ImageURLPart("https://example.com/logo.png"),
					react.ImageDataPart("image/png", []byte("fake-png-bytes")),
				},
			},
		},
	}

	resp, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "I see a chart and a logo." {
		t.Errorf("unexpected response: %q", resp.Message.Text())
	}

	msgs, ok := capturedBody["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("expected 1 message in payload, got %+v", capturedBody)
	}
	userMsg := msgs[0].(map[string]any)
	parts, ok := userMsg["content"].([]any)
	if !ok || len(parts) != 3 {
		t.Fatalf("expected 3 parts in content, got %+v", userMsg["content"])
	}
}

func TestOpenCodeStream(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"reasoning_content":"Thinking..."}}]}

data: {"choices":[{"delta":{"content":"Result is "}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"calc","arguments":"{\"val\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"42}"}}]}}]}

data: {"choices":[{"finish_reason":"tool_calls"}]}

data: {"usage":{"prompt_tokens":12,"completion_tokens":6}}

data: [DONE]

`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") != "text/event-stream" {
			t.Errorf("expected Accept text/event-stream, got %q", r.Header.Get("Accept"))
		}
		if r.Header.Get("User-Agent") != DefaultUserAgent {
			t.Errorf("expected User-Agent %q, got %q", DefaultUserAgent, r.Header.Get("User-Agent"))
		}
		if r.Header.Get(HeaderSession) == "" {
			t.Errorf("expected non-empty x-opencode-session header")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, sseData)
	}))
	defer srv.Close()

	p := New(provider.Config{BaseURL: srv.URL})
	req := &react.Request{
		Model:    ModelDeepSeekV4Pro,
		Messages: []react.Message{react.UserText("hi")},
	}

	var acc react.Accumulator
	for ev, err := range p.Stream(context.Background(), req) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		acc.Add(ev)
	}

	resp := acc.Response()
	if resp.Message.Text() != "Result is " {
		t.Fatalf("text = %q, want 'Result is '", resp.Message.Text())
	}
	if resp.Reasoning != "Thinking..." {
		t.Fatalf("reasoning = %q, want 'Thinking...'", resp.Reasoning)
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Name != "calc" {
		t.Fatalf("calls = %+v", resp.Message.ToolCalls)
	}
	if !strings.Contains(string(resp.Message.ToolCalls[0].Arguments), "42}") {
		t.Fatalf("tool args = %s", string(resp.Message.ToolCalls[0].Arguments))
	}
	if resp.StopReason != react.StopToolCall {
		t.Fatalf("stop = %v", resp.StopReason)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 6 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestOpenCodeFinishReasons(t *testing.T) {
	tests := []struct {
		in   string
		want react.StopReason
	}{
		{"stop", react.StopEnd},
		{"length", react.StopLength},
		{"tool_calls", react.StopToolCall},
		{"function_call", react.StopToolCall},
		{"content_filter", react.StopContentFilter},
		{"unknown", react.StopOther},
	}

	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			got := provider.MapFinishReason(tt.in)
			if got != tt.want {
				t.Errorf("provider.MapFinishReason(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestOpenCodeResponsesGenerate(t *testing.T) {
	var capturedHeader http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeader = r.Header.Clone()
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q, want /responses", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"id": "resp_oc_test",
			"status": "completed",
			"output": [
				{
					"type": "message",
					"role": "assistant",
					"content": [{"type": "output_text", "text": "OpenCode Responses output"}]
				}
			],
			"usage": {
				"input_tokens": 14,
				"output_tokens": 7
			}
		}`)
	}))
	defer srv.Close()

	p := NewResponses(provider.Config{
		APIKey:  "test-oc-key",
		BaseURL: srv.URL,
	})

	ctx := react.WithRunContext(context.Background(), &react.RunContext{SessionID: "sess-resp-1"})
	resp, err := p.Generate(ctx, &react.Request{
		Model:    ModelDeepSeekV4Pro,
		Messages: []react.Message{react.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "OpenCode Responses output" {
		t.Fatalf("unexpected text: %q", resp.Message.Text())
	}
	if capturedHeader.Get("User-Agent") != DefaultUserAgent {
		t.Errorf("expected User-Agent %q, got %q", DefaultUserAgent, capturedHeader.Get("User-Agent"))
	}
	if capturedHeader.Get(HeaderSession) != "sess-resp-1" {
		t.Errorf("expected x-opencode-session sess-resp-1, got %q", capturedHeader.Get(HeaderSession))
	}
}

func TestOpenCodeResponsesStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q, want /responses", r.URL.Path)
		}
		if r.Header.Get(HeaderSession) != "sess-resp-stream" {
			t.Errorf("expected x-opencode-session sess-resp-stream, got %q", r.Header.Get(HeaderSession))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Streamed response"}

event: response.completed
data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":10,"output_tokens":5}}}

data: [DONE]
`)
	}))
	defer srv.Close()

	p := NewResponses(provider.Config{
		APIKey:  "test-oc-key",
		BaseURL: srv.URL,
	})

	ctx := react.WithRunContext(context.Background(), &react.RunContext{SessionID: "sess-resp-stream"})
	var acc react.Accumulator
	for ev, err := range p.Stream(ctx, &react.Request{Model: ModelKimiK3}) {
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(ev)
	}

	resp := acc.Response()
	if resp.Message.Text() != "Streamed response" {
		t.Fatalf("text = %q, want 'Streamed response'", resp.Message.Text())
	}
	if resp.StopReason != react.StopEnd {
		t.Fatalf("stop = %v, want StopEnd", resp.StopReason)
	}
}
