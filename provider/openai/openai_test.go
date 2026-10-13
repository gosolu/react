package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider/openai"
)

func TestOpenAIGenerate(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing or bad auth header: %v", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
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

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    "gpt-4o",
		Messages: []react.Message{react.UserText("hi")},
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
	if resp.StopReason != react.StopToolCall {
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
		_, _ = fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := openai.New(openai.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	var acc react.Accumulator
	for ev, err := range p.Stream(context.Background(), &react.Request{Model: "gpt-4o"}) {
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
	if resp.StopReason != react.StopToolCall {
		t.Fatalf("stop = %v", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestOpenAIFinishReasons(t *testing.T) {
	cases := []struct {
		finishReason string
		expected     react.StopReason
	}{
		{"stop", react.StopEnd},
		{"length", react.StopLength},
		{"tool_calls", react.StopToolCall},
		{"function_call", react.StopToolCall},
		{"content_filter", react.StopContentFilter},
		{"unknown_custom", react.StopOther},
	}

	for _, tc := range cases {
		t.Run(tc.finishReason, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprintf(w, `{
					"choices": [{
						"message": {"content": "ok"},
						"finish_reason": %q
					}],
					"usage": {"prompt_tokens": 1, "completion_tokens": 1}
				}`, tc.finishReason)
			}))
			defer ts.Close()

			p := openai.New(openai.Config{APIKey: "k", BaseURL: ts.URL})
			resp, err := p.Generate(context.Background(), &react.Request{Model: "gpt-4o", Messages: []react.Message{react.UserText("test")}})
			if err != nil {
				t.Fatal(err)
			}
			if resp.StopReason != tc.expected {
				t.Fatalf("finish_reason %q got %v, want %v", tc.finishReason, resp.StopReason, tc.expected)
			}
		})
	}
}

func TestOpenAIResponsesGenerate(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q, want /responses", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"id": "resp_test",
			"status": "completed",
			"output": [
				{
					"type": "message",
					"role": "assistant",
					"content": [{"type": "output_text", "text": "Hello from Responses API"}]
				},
				{
					"type": "function_call",
					"call_id": "call_1",
					"name": "calc",
					"arguments": "{\"x\":1}"
				}
			],
			"usage": {
				"input_tokens": 10,
				"output_tokens": 20
			}
		}`)
	}))
	defer ts.Close()

	p := openai.NewResponses(openai.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    "gpt-4o",
		Messages: []react.Message{react.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "Hello from Responses API" {
		t.Fatalf("unexpected text: %q", resp.Message.Text())
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Name != "calc" {
		t.Fatalf("unexpected tool calls: %+v", resp.Message.ToolCalls)
	}
	if resp.StopReason != react.StopToolCall {
		t.Fatalf("expected StopToolCall, got %v", resp.StopReason)
	}
}

func TestOpenAIResponsesStream(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q, want /responses", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hi"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"!"}

event: response.completed
data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":5,"output_tokens":2}}}

data: [DONE]
`)
	}))
	defer ts.Close()

	p := openai.NewResponses(openai.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	var acc react.Accumulator
	for ev, err := range p.Stream(context.Background(), &react.Request{Model: "gpt-4o"}) {
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(ev)
	}

	resp := acc.Response()
	if resp.Message.Text() != "Hi!" {
		t.Fatalf("text = %q, want 'Hi!'", resp.Message.Text())
	}
	if resp.StopReason != react.StopEnd {
		t.Fatalf("stop = %v, want StopEnd", resp.StopReason)
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestOpenAICachedTokensAndMultimodalToolResult(t *testing.T) {
	var capturedBody map[string]any

	mockClient := &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			_ = json.NewDecoder(r.Body).Decode(&capturedBody)
			respJSON := `{
				"choices": [{
					"message": {"content": "ok"},
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 120,
					"completion_tokens": 30,
					"prompt_tokens_details": {
						"cached_tokens": 80
					}
				}
			}`
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(respJSON)),
			}, nil
		}),
	}

	p := openai.New(openai.Config{
		APIKey:     "test-key",
		HTTPClient: mockClient,
	})

	req := &react.Request{
		Model: "gpt-4o",
		Messages: []react.Message{
			react.UserText("look at this"),
			{
				Role:      react.RoleAssistant,
				Reasoning: "OpenAI reasoning content",
				ToolCalls: []react.ToolCall{
					{ID: "c1", Name: "browse", Arguments: []byte(`{}`)},
				},
			},
			{
				Role:       react.RoleTool,
				ToolCallID: "c1",
				Content: []react.Part{
					react.TextPart("page loaded:"),
					react.ImageDataPart("image/png", []byte("fake-png-data")),
				},
			},
		},
	}

	resp, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// Verify cached tokens extracted into Usage.CacheReadTokens
	if resp.Usage.CacheReadTokens != 80 {
		t.Errorf("expected 80 cached tokens, got %d", resp.Usage.CacheReadTokens)
	}
	if resp.Usage.InputTokens != 120 || resp.Usage.OutputTokens != 30 {
		t.Errorf("expected 120/30 tokens, got %+v", resp.Usage)
	}

	// Verify request body formatting
	messages, ok := capturedBody["messages"].([]any)
	if !ok || len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %+v", capturedBody["messages"])
	}

	// Assistant message has reasoning_content
	asst := messages[1].(map[string]any)
	if asst["reasoning_content"] != "OpenAI reasoning content" {
		t.Errorf("expected reasoning_content on assistant, got %v", asst["reasoning_content"])
	}

	// Tool message has multimodal parts
	toolMsg := messages[2].(map[string]any)
	if toolMsg["role"] != "tool" {
		t.Errorf("expected role tool, got %v", toolMsg["role"])
	}
	toolParts, ok := toolMsg["content"].([]any)
	if !ok || len(toolParts) != 2 {
		t.Fatalf("expected 2 parts in tool content, got %+v", toolMsg["content"])
	}
	imgPart := toolParts[1].(map[string]any)
	if imgPart["type"] != "image_url" {
		t.Errorf("expected type image_url, got %v", imgPart["type"])
	}
}
