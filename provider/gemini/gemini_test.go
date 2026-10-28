package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider"
)

type stubClient struct {
	postFunc   func(ctx context.Context, url string, headers map[string]string, body io.Reader) (io.ReadCloser, error)
	streamFunc func(ctx context.Context, url string, headers map[string]string, body io.Reader) iter.Seq2[provider.SSEEvent, error]
}

func (s *stubClient) Post(ctx context.Context, url string, headers map[string]string, body io.Reader) (io.ReadCloser, error) {
	return s.postFunc(ctx, url, headers, body)
}

func (s *stubClient) Stream(ctx context.Context, url string, headers map[string]string, body io.Reader) iter.Seq2[provider.SSEEvent, error] {
	return s.streamFunc(ctx, url, headers, body)
}

func TestGeminiChatCompletionsMode(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/interactions" {
			t.Errorf("path = %q, want /interactions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"id": "v1_cc_redirect",
			"status": "completed",
			"steps": [{
				"type": "model_output",
				"content": [{"type": "text", "text": "Hello from Gemini Interactions redirect"}]
			}],
			"usage": {"total_input_tokens": 5, "total_output_tokens": 6}
		}`)
	}))
	defer ts.Close()

	p := NewChatCompletions(provider.Config{
		APIKey:  "gemini-key",
		BaseURL: ts.URL,
	})

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    ModelGemini38Flash,
		Messages: []react.Message{react.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "Hello from Gemini Interactions redirect" {
		t.Fatalf("unexpected text: %q", resp.Message.Text())
	}
}

func TestGeminiInteractions_Stream_Error(t *testing.T) {
	sseData := `data: {"error":{"code":400,"message":"Invalid request argument"}}
`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := New(provider.Config{
		APIKey:  "gemini-key",
		BaseURL: ts.URL,
	})

	var errFound error
	for _, err := range p.Stream(context.Background(), &react.Request{Model: ModelGemini38Flash}) {
		if err != nil {
			errFound = err
			break
		}
	}

	if errFound == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(errFound.Error(), "Invalid request argument") {
		t.Fatalf("expected 'Invalid request argument' in error, got %v", errFound)
	}
}

func TestGeminiInteractions_Generate_Text(t *testing.T) {
	var capturedBody map[string]any

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("missing or bad x-goog-api-key: %v", r.Header.Get("x-goog-api-key"))
		}
		if r.URL.Path != "/interactions" {
			t.Errorf("unexpected endpoint path: %s", r.URL.Path)
		}
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"id": "v1_test_inter_text",
			"model": "gemini-3.8-flash",
			"status": "completed",
			"steps": [
				{
					"type": "model_output",
					"content": [
						{"type": "text", "text": "Hello from Interactions API!"}
					]
				}
			],
			"usage": {
				"total_input_tokens": 12,
				"total_output_tokens": 6,
				"total_thought_tokens": 4,
				"total_cached_tokens": 2
			}
		}`)
	}))
	defer ts.Close()

	// New(cfg) defaults to Interactions API
	p := New(provider.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	temp := 0.7
	resp, err := p.Generate(context.Background(), &react.Request{
		Model:       ModelGemini38Flash,
		System:      "Be helpful and concise",
		Messages:    []react.Message{react.UserText("Hello Gemini")},
		MaxTokens:   500,
		Temperature: &temp,
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp.Message.Text() != "Hello from Interactions API!" {
		t.Fatalf("text = %q, want 'Hello from Interactions API!'", resp.Message.Text())
	}
	if resp.StopReason != react.StopEnd {
		t.Fatalf("stop = %v, want StopEnd", resp.StopReason)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 10 || resp.Usage.CacheReadTokens != 2 {
		t.Fatalf("unexpected usage: %+v", resp.Usage)
	}

	// Verify captured request body
	if capturedBody["model"] != "gemini-3.8-flash" {
		t.Errorf("captured model = %v, want gemini-3.8-flash", capturedBody["model"])
	}
	if capturedBody["system_instruction"] != "Be helpful and concise" {
		t.Errorf("captured system_instruction = %v, want 'Be helpful and concise'", capturedBody["system_instruction"])
	}
	if capturedBody["input"] != "Hello Gemini" {
		t.Errorf("captured input = %v, want 'Hello Gemini'", capturedBody["input"])
	}
	genCfg, ok := capturedBody["generation_config"].(map[string]any)
	if !ok || genCfg["max_output_tokens"].(float64) != 500 {
		t.Errorf("captured max_output_tokens = %v, want 500", genCfg)
	}
}

func TestGeminiInteractions_Generate_ToolCall(t *testing.T) {
	var capturedBody map[string]any

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/interactions" {
			t.Errorf("unexpected endpoint path: %s", r.URL.Path)
		}
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"id": "v1_test_tool",
			"status": "requires_action",
			"steps": [
				{
					"type": "thought",
					"text": "I will check the weather in Paris",
					"signature": "sig_th1"
				},
				{
					"type": "function_call",
					"id": "fc_001",
					"name": "get_weather",
					"arguments": {"city": "Paris"},
					"signature": "sig_fc1"
				}
			],
			"usage": {
				"total_input_tokens": 20,
				"total_output_tokens": 15
			}
		}`)
	}))
	defer ts.Close()

	p := NewInteractions(provider.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:      ModelGemini38Flash,
		Messages:   []react.Message{react.UserText("weather in Paris?")},
		ToolChoice: react.ToolChoiceRequired,
		Tools: []react.ToolSpec{
			{
				Name:        "get_weather",
				Description: "Get weather for city",
				InputSchema: []byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
			},
		},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp.StopReason != react.StopToolCall {
		t.Fatalf("stopReason = %v, want StopToolCall", resp.StopReason)
	}
	if resp.Reasoning != "I will check the weather in Paris" {
		t.Fatalf("reasoning = %q", resp.Reasoning)
	}
	if resp.Message.ReasoningSignature != "sig_th1" {
		t.Fatalf("reasoningSignature = %q, want 'sig_th1'", resp.Message.ReasoningSignature)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "fc_001" || tc.Name != "get_weather" {
		t.Fatalf("unexpected tool call: %+v", tc)
	}
	if tc.ThoughtSignature != "sig_fc1" {
		t.Fatalf("tc thoughtSignature = %q, want 'sig_fc1'", tc.ThoughtSignature)
	}
	if !strings.Contains(string(tc.Arguments), `"Paris"`) {
		t.Fatalf("tc args = %s", string(tc.Arguments))
	}

	// Verify captured tool definition adheres to Interactions API flat polymorphic schema
	tools, ok := capturedBody["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("captured tools = %+v, want 1 tool", capturedBody["tools"])
	}
	t0 := tools[0].(map[string]any)
	if t0["type"] != "function" || t0["name"] != "get_weather" {
		t.Fatalf("tool[0] = %+v, want type: function, name: get_weather", t0)
	}
	genCfg := capturedBody["generation_config"].(map[string]any)
	if genCfg["tool_choice"] != "any" {
		t.Errorf("tool_choice = %v, want 'any'", genCfg["tool_choice"])
	}
}

func TestGeminiInteractions_MultiTurn_Stateless_Input(t *testing.T) {
	var capturedBody map[string]any

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"id": "v1_multiturn",
			"status": "completed",
			"steps": [
				{
					"type": "model_output",
					"content": [{"type": "text", "text": "The weather in Tokyo is 18C."}]
				}
			]
		}`)
	}))
	defer ts.Close()

	p := New(provider.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	req := &react.Request{
		Model: ModelGemini38Flash,
		Messages: []react.Message{
			react.UserText("What is the weather in Tokyo?"),
			{
				Role:               react.RoleAssistant,
				Reasoning:          "Need to check Tokyo weather first.",
				ReasoningSignature: "sig_thought_tokyo",
				ToolCalls: []react.ToolCall{
					{
						ID:               "call_tokyo",
						Name:             "get_weather",
						Arguments:        []byte(`{"city":"Tokyo"}`),
						ThoughtSignature: "sig_fc_tokyo",
					},
				},
			},
			{
				Role:       react.RoleTool,
				ToolCallID: "call_tokyo",
				Name:       "get_weather",
				Content:    []react.Part{react.TextPart(`{"temp": "18C"}`)},
			},
		},
	}

	resp, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "The weather in Tokyo is 18C." {
		t.Fatalf("text = %q", resp.Message.Text())
	}

	// Verify input contains chronological Steps: user_input, thought, function_call, function_result
	inputSteps, ok := capturedBody["input"].([]any)
	if !ok || len(inputSteps) != 4 {
		t.Fatalf("captured input = %+v, want 4 steps", capturedBody["input"])
	}

	s0 := inputSteps[0].(map[string]any)
	if s0["type"] != "user_input" {
		t.Errorf("step 0 type = %v, want user_input", s0["type"])
	}

	s1 := inputSteps[1].(map[string]any)
	if s1["type"] != "thought" || s1["text"] != "Need to check Tokyo weather first." || s1["signature"] != "sig_thought_tokyo" {
		t.Errorf("step 1 = %+v, want thought step with text and signature", s1)
	}

	s2 := inputSteps[2].(map[string]any)
	if s2["type"] != "function_call" || s2["id"] != "call_tokyo" || s2["name"] != "get_weather" || s2["signature"] != "sig_fc_tokyo" {
		t.Errorf("step 2 = %+v, want function_call for call_tokyo with signature", s2)
	}

	s3 := inputSteps[3].(map[string]any)
	if s3["type"] != "function_result" || s3["call_id"] != "call_tokyo" || s3["name"] != "get_weather" {
		t.Errorf("step 3 = %+v, want function_result for call_tokyo", s3)
	}
}

func TestGeminiInteractions_Stream_UnexpectedEOF(t *testing.T) {
	sseData := `event: step.start
data: {"type": "step.start", "index": 0, "step": {"type": "model_output"}}

event: step.delta
data: {"type": "step.delta", "index": 0, "delta": {"type": "text", "text": "Partial stream without completion"}}
`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := New(provider.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	var errFound error
	for _, err := range p.Stream(context.Background(), &react.Request{Model: ModelGemini38Flash, Messages: []react.Message{react.UserText("hi")}}) {
		if err != nil {
			errFound = err
			break
		}
	}

	if !errors.Is(errFound, io.ErrUnexpectedEOF) {
		t.Fatalf("expected io.ErrUnexpectedEOF on truncated stream, got %v", errFound)
	}
}

func TestGeminiInteractions_Stateful_PreviousInteractionID(t *testing.T) {
	var capturedBody map[string]any

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"id": "v1_stateful_turn2",
			"status": "completed",
			"steps": [
				{"type": "model_output", "content": [{"type": "text", "text": "4"}]}
			]
		}`)
	}))
	defer ts.Close()

	p := New(provider.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	req := &react.Request{
		Model: ModelGemini38Flash,
		Messages: []react.Message{
			react.UserText("Hi"),
			react.AssistantText("Hello"),
			react.UserText("What is 2+2?"),
		},
		Extra: map[string]any{
			"previous_interaction_id": "int_prev_999",
		},
	}

	resp, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "4" {
		t.Fatalf("text = %q", resp.Message.Text())
	}

	if capturedBody["previous_interaction_id"] != "int_prev_999" {
		t.Errorf("previous_interaction_id = %v, want int_prev_999", capturedBody["previous_interaction_id"])
	}

	// In stateful mode with previous_interaction_id, only the latest turn is sent!
	inputSteps, ok := capturedBody["input"].([]any)
	if !ok || len(inputSteps) != 1 {
		t.Fatalf("expected 1 step in input, got %d: %+v", len(inputSteps), capturedBody["input"])
	}
	step0 := inputSteps[0].(map[string]any)
	if step0["type"] != "user_input" {
		t.Errorf("step0 type = %v, want user_input", step0["type"])
	}
}

func TestGeminiInteractions_Stream_Text(t *testing.T) {
	sseData := `event: step.start
data: {"type": "step.start", "index": 0, "step": {"type": "thought"}}

event: step.delta
data: {"type": "step.delta", "index": 0, "delta": {"type": "thought", "text": "Thinking..."}}

event: step.start
data: {"type": "step.start", "index": 1, "step": {"type": "model_output"}}

event: step.delta
data: {"type": "step.delta", "index": 1, "delta": {"type": "text", "text": "Hello "}}

event: step.delta
data: {"type": "step.delta", "index": 1, "delta": {"type": "text", "text": "World!"}}

event: interaction.completed
data: {"type": "interaction.completed", "interaction": {"status": "completed", "usage": {"total_input_tokens": 10, "total_output_tokens": 5, "total_thought_tokens": 3}}}
`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/interactions" {
			t.Errorf("path = %q, want /interactions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := New(provider.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	var acc react.Accumulator
	for ev, err := range p.Stream(context.Background(), &react.Request{Model: ModelGemini38Flash, Messages: []react.Message{react.UserText("hi")}}) {
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(ev)
	}

	resp := acc.Response()
	if resp.Message.Text() != "Hello World!" {
		t.Fatalf("text = %q, want 'Hello World!'", resp.Message.Text())
	}
	if resp.Reasoning != "Thinking..." {
		t.Fatalf("reasoning = %q, want 'Thinking...'", resp.Reasoning)
	}
	if resp.StopReason != react.StopEnd {
		t.Fatalf("stopReason = %v, want StopEnd", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 8 {
		t.Fatalf("usage = %+v, want input: 10, output: 8", resp.Usage)
	}
}

func TestGeminiInteractions_Stream_FunctionCall(t *testing.T) {
	sseData := `event: step.start
data: {"type": "step.start", "index": 0, "step": {"type": "function_call", "id": "fc_stream", "name": "calc", "signature": "sig_fc"}}

event: step.delta
data: {"type": "step.delta", "index": 0, "delta": {"type": "arguments_delta", "arguments": "{\"expr\":"}}

event: step.delta
data: {"type": "step.delta", "index": 0, "delta": {"type": "arguments_delta", "arguments": " \"1+1\"}"}}

event: step.stop
data: {"type": "step.stop", "index": 0, "status": "waiting"}

event: interaction.requires_action
data: {"type": "interaction.requires_action", "interaction": {"status": "requires_action"}}
`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := New(provider.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	var acc react.Accumulator
	for ev, err := range p.Stream(context.Background(), &react.Request{Model: ModelGemini38Flash}) {
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(ev)
	}

	resp := acc.Response()
	if resp.StopReason != react.StopToolCall {
		t.Fatalf("stopReason = %v, want StopToolCall", resp.StopReason)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "fc_stream" || tc.Name != "calc" {
		t.Fatalf("unexpected tc: %+v", tc)
	}
	if string(tc.Arguments) != `{"expr": "1+1"}` {
		t.Fatalf("args = %q, want '{\"expr\": \"1+1\"}'", string(tc.Arguments))
	}
	if tc.ThoughtSignature != "sig_fc" {
		t.Fatalf("tc signature = %q, want 'sig_fc'", tc.ThoughtSignature)
	}
}

func TestGeminiInteractions_Multimodal(t *testing.T) {
	var capturedBody map[string]any

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"status": "completed",
			"steps": [
				{"type": "model_output", "content": [{"type": "text", "text": "A lovely image"}]}
			]
		}`)
	}))
	defer ts.Close()

	p := New(provider.Config{
		APIKey:  "test-key",
		BaseURL: ts.URL,
	})

	rawImage := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	req := &react.Request{
		Model: ModelGeminiFlashLatest,
		Messages: []react.Message{
			{
				Role: react.RoleUser,
				Content: []react.Part{
					react.TextPart("Analyze this image:"),
					react.ImageDataPart("image/png", rawImage),
					react.ImageURLPart("https://example.com/test.jpg"),
				},
			},
		},
	}

	resp, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "A lovely image" {
		t.Fatalf("unexpected text: %q", resp.Message.Text())
	}

	contentItems, ok := capturedBody["input"].([]any)
	if !ok || len(contentItems) != 3 {
		t.Fatalf("expected 3 content items in input, got %v", capturedBody["input"])
	}

	// Item 0: text
	c0 := contentItems[0].(map[string]any)
	if c0["type"] != "text" || c0["text"] != "Analyze this image:" {
		t.Errorf("item 0 = %+v", c0)
	}

	// Item 1: inline image
	c1 := contentItems[1].(map[string]any)
	if c1["type"] != "image" || c1["mime_type"] != "image/png" || c1["data"] == "" {
		t.Errorf("item 1 = %+v", c1)
	}

	// Item 2: URL image
	c2 := contentItems[2].(map[string]any)
	if c2["type"] != "image" || c2["uri"] != "https://example.com/test.jpg" {
		t.Errorf("item 2 = %+v", c2)
	}
}
