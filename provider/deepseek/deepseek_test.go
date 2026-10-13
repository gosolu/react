package deepseek_test

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
	"github.com/gosolu/react/provider/deepseek"
)

func TestDeepSeekGenerate(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing or bad auth header: %v", r.Header.Get("Authorization"))
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"choices": [{
				"message": {
					"content": "DeepSeek response",
					"tool_calls": [{
						"id": "call_1",
						"function": {
							"name": "search",
							"arguments": "{\"query\":\"react\"}"
						}
					}]
				},
				"finish_reason": "tool_calls"
			}],
			"usage": {
				"prompt_tokens": 15,
				"completion_tokens": 8
			}
		}`)
	}))
	defer ts.Close()

	p := deepseek.New(deepseek.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	if p.Name() != "deepseek" {
		t.Fatalf("unexpected provider name: %s", p.Name())
	}

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    deepseek.ModelChat,
		Messages: []react.Message{react.UserText("hello")},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "DeepSeek response" {
		t.Fatalf("unexpected text: %q", resp.Message.Text())
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Name != "search" {
		t.Fatalf("unexpected tool calls: %+v", resp.Message.ToolCalls)
	}
	if resp.StopReason != react.StopToolCall {
		t.Fatalf("stop reason = %v", resp.StopReason)
	}
	if resp.Usage.InputTokens != 15 || resp.Usage.OutputTokens != 8 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestDeepSeekGenerateVision(t *testing.T) {
	var capturedBody map[string]any

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"choices": [{
				"message": {
					"content": "I see a test diagram."
				},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 100,
				"completion_tokens": 10
			}
		}`)
	}))
	defer ts.Close()

	p := deepseek.New(deepseek.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	userMsg := react.Message{
		Role: react.RoleUser,
		Content: []react.Part{
			react.TextPart("What is in this image?"),
			react.ImageURLPart("https://example.com/chart.png"),
			react.ImageDataPart("image/png", []byte("fake-png-data")),
		},
	}

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    deepseek.ModelFlash,
		Messages: []react.Message{userMsg},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "I see a test diagram." {
		t.Fatalf("unexpected text: %q", resp.Message.Text())
	}

	// Verify request payload format according to DeepSeek Vision documentation
	messages, ok := capturedBody["messages"].([]any)
	if !ok || len(messages) != 1 {
		t.Fatalf("unexpected messages payload: %+v", capturedBody)
	}
	msgMap := messages[0].(map[string]any)
	parts, ok := msgMap["content"].([]any)
	if !ok || len(parts) != 3 {
		t.Fatalf("expected 3 content parts, got: %+v", msgMap["content"])
	}

	p1 := parts[0].(map[string]any)
	if p1["type"] != "text" || p1["text"] != "What is in this image?" {
		t.Fatalf("part 1 mismatch: %+v", p1)
	}

	p2 := parts[1].(map[string]any)
	if p2["type"] != "image_url" {
		t.Fatalf("part 2 type mismatch: %+v", p2)
	}
	imgURL2 := p2["image_url"].(map[string]any)["url"]
	if imgURL2 != "https://example.com/chart.png" {
		t.Fatalf("part 2 image_url mismatch: %v", imgURL2)
	}

	p3 := parts[2].(map[string]any)
	if p3["type"] != "image_url" {
		t.Fatalf("part 3 type mismatch: %+v", p3)
	}
	imgURL3 := p3["image_url"].(map[string]any)["url"].(string)
	if !strings.HasPrefix(imgURL3, "data:image/png;base64,") {
		t.Fatalf("part 3 data URL mismatch: %v", imgURL3)
	}
}

func TestDeepSeekGenerateReasoning(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"choices": [{
				"message": {
					"content": "The answer is 42.",
					"reasoning_content": "Let me think step-by-step: 6 * 7 = 42."
				},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 10,
				"completion_tokens": 20
			}
		}`)
	}))
	defer ts.Close()

	p := deepseek.New(deepseek.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    deepseek.ModelReasoner,
		Messages: []react.Message{react.UserText("solve")},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "The answer is 42." {
		t.Fatalf("unexpected text: %q", resp.Message.Text())
	}
	if resp.Reasoning != "Let me think step-by-step: 6 * 7 = 42." {
		t.Fatalf("unexpected reasoning: %q", resp.Reasoning)
	}
	if resp.StopReason != react.StopEnd {
		t.Fatalf("stop reason = %v", resp.StopReason)
	}
}

func TestDeepSeekStream(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"reasoning_content":"Thinking..."}}]}

data: {"choices":[{"delta":{"content":"Result is "}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"calc","arguments":"{\"val\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"42}"}}]}}]}

data: {"choices":[{"finish_reason":"tool_calls"}]}

data: {"usage":{"prompt_tokens":12,"completion_tokens":6}}

data: [DONE]

`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := deepseek.New(deepseek.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	var acc react.Accumulator
	for ev, err := range p.Stream(context.Background(), &react.Request{Model: deepseek.ModelReasoner}) {
		if err != nil {
			t.Fatal(err)
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

func TestDeepSeekFinishReasons(t *testing.T) {
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
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
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

			p := deepseek.New(deepseek.Config{APIKey: "k", BaseURL: ts.URL})
			resp, err := p.Generate(context.Background(), &react.Request{Model: deepseek.ModelChat, Messages: []react.Message{react.UserText("test")}})
			if err != nil {
				t.Fatal(err)
			}
			if resp.StopReason != tc.expected {
				t.Fatalf("finish_reason %q got %v, want %v", tc.finishReason, resp.StopReason, tc.expected)
			}
		})
	}
}

func TestDeepSeekResponsesGenerate(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q, want /responses", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"id": "resp_ds_test",
			"status": "completed",
			"output": [
				{
					"type": "reasoning",
					"content": [{"type": "reasoning_text", "text": "DeepSeek R1 thinking..."}]
				},
				{
					"type": "message",
					"role": "assistant",
					"content": [{"type": "output_text", "text": "DeepSeek responses result"}]
				}
			],
			"usage": {
				"input_tokens": 8,
				"output_tokens": 16
			}
		}`)
	}))
	defer ts.Close()

	p := deepseek.NewResponses(deepseek.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    deepseek.ModelFlash,
		Messages: []react.Message{react.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "DeepSeek responses result" {
		t.Fatalf("unexpected text: %q", resp.Message.Text())
	}
	if resp.Reasoning != "DeepSeek R1 thinking..." {
		t.Fatalf("unexpected reasoning: %q", resp.Reasoning)
	}
	if resp.StopReason != react.StopEnd {
		t.Fatalf("expected StopEnd, got %v", resp.StopReason)
	}
}

func TestDeepSeekResponsesStream(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q, want /responses", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `event: response.reasoning_text.delta
data: {"type":"response.reasoning_text.delta","delta":"DeepSeek thought"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hello world"}

event: response.completed
data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":6,"output_tokens":10}}}

data: [DONE]
`)
	}))
	defer ts.Close()

	p := deepseek.NewResponses(deepseek.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	var acc react.Accumulator
	for ev, err := range p.Stream(context.Background(), &react.Request{Model: deepseek.ModelReasoner}) {
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(ev)
	}

	resp := acc.Response()
	if resp.Message.Text() != "Hello world" {
		t.Fatalf("text = %q, want 'Hello world'", resp.Message.Text())
	}
	if resp.Reasoning != "DeepSeek thought" {
		t.Fatalf("reasoning = %q, want 'DeepSeek thought'", resp.Reasoning)
	}
	if resp.StopReason != react.StopEnd {
		t.Fatalf("stop = %v, want StopEnd", resp.StopReason)
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestDeepSeekCachedTokensAndReasoningPersistence(t *testing.T) {
	var capturedBody map[string]any

	mockClient := &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			_ = json.NewDecoder(r.Body).Decode(&capturedBody)
			respJSON := `{
				"choices": [{
					"message": {
						"content": "Answer",
						"reasoning_content": "DeepSeek thought process"
					},
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 150,
					"completion_tokens": 40,
					"prompt_cache_hit_tokens": 100,
					"prompt_cache_miss_tokens": 50
				}
			}`
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(respJSON)),
			}, nil
		}),
	}

	p := deepseek.New(deepseek.Config{
		APIKey:     "test-key",
		HTTPClient: mockClient,
	})

	req := &react.Request{
		Model: deepseek.ModelReasoner,
		Messages: []react.Message{
			react.UserText("solve this"),
			{
				Role:      react.RoleAssistant,
				Reasoning: "Prior reasoning step",
				ToolCalls: []react.ToolCall{
					{ID: "c1", Name: "solve", Arguments: []byte(`{}`)},
				},
			},
			{
				Role:       react.RoleTool,
				ToolCallID: "c1",
				Content:    []react.Part{react.TextPart("output 1")},
			},
		},
	}

	resp, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// 1. Verify response captured reasoning and cache metrics
	if resp.Message.Reasoning != "DeepSeek thought process" {
		t.Errorf("expected Reasoning 'DeepSeek thought process', got %q", resp.Message.Reasoning)
	}
	if resp.Reasoning != "DeepSeek thought process" {
		t.Errorf("expected resp.Reasoning 'DeepSeek thought process', got %q", resp.Reasoning)
	}
	if resp.Usage.CacheReadTokens != 100 || resp.Usage.CacheWriteTokens != 50 {
		t.Errorf("expected cache tokens 100/50, got %+v", resp.Usage)
	}

	// 2. Verify request body included reasoning_content for assistant
	messages, ok := capturedBody["messages"].([]any)
	if !ok || len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %+v", capturedBody["messages"])
	}
	asst := messages[1].(map[string]any)
	if asst["reasoning_content"] != "Prior reasoning step" {
		t.Errorf("expected reasoning_content on assistant, got %v", asst["reasoning_content"])
	}
}
