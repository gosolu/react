package anthropic

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
		_, _ = fmt.Fprintln(w, `{
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

	p := New(Config{
		APIKey:  "ant-key",
		BaseURL: ts.URL,
	})

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    "claude-3-5-sonnet-20241022",
		Messages: []react.Message{react.UserText("weather in Paris?")},
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
	if resp.StopReason != react.StopToolCall {
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
		_, _ = fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := New(Config{
		APIKey:  "ant-key",
		BaseURL: ts.URL,
	})

	var acc react.Accumulator
	for ev, err := range p.Stream(context.Background(), &react.Request{Model: "claude-3-5-sonnet"}) {
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
	if resp.StopReason != react.StopToolCall {
		t.Fatalf("stop = %v", resp.StopReason)
	}
	if resp.Usage.InputTokens != 20 || resp.Usage.OutputTokens != 12 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestAnthropicThinkingAndTurnCollapse(t *testing.T) {
	var capturedBody map[string]any

	mockClient := &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			_ = json.NewDecoder(r.Body).Decode(&capturedBody)
			respJSON := `{
				"id": "msg_456",
				"type": "message",
				"role": "assistant",
				"content": [
					{"type": "thinking", "thinking": "Thinking step 1", "signature": "sig_abc"},
					{"type": "text", "text": "Done."}
				],
				"stop_reason": "end_turn",
				"usage": {
					"input_tokens": 100,
					"output_tokens": 50,
					"cache_creation_input_tokens": 40,
					"cache_read_input_tokens": 60
				}
			}`
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(respJSON)),
			}, nil
		}),
	}

	p := New(Config{
		APIKey:     "ant-key",
		HTTPClient: mockClient,
	})

	// Send an assistant message with thinking + 2 tool calls, followed by 2 tool results (one with an image)
	req := &react.Request{
		Model: "claude-3-7-sonnet",
		Messages: []react.Message{
			react.UserText("look at this"),
			{
				Role:               react.RoleAssistant,
				Reasoning:          "Prior thought",
				ReasoningSignature: "sig_123",
				ToolCalls: []react.ToolCall{
					{ID: "c1", Name: "t1", Arguments: []byte(`{}`)},
					{ID: "c2", Name: "t2", Arguments: []byte(`{}`)},
				},
			},
			{
				Role:       react.RoleTool,
				ToolCallID: "c1",
				Content:    []react.Part{react.TextPart("result 1")},
			},
			{
				Role:       react.RoleTool,
				ToolCallID: "c2",
				Content: []react.Part{
					react.TextPart("screenshot:"),
					react.ImageDataPart("image/png", []byte("fake-png-data")),
				},
			},
		},
	}

	resp, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// 1. Verify response captured thinking and cache tokens
	if resp.Message.Reasoning != "Thinking step 1" {
		t.Errorf("expected Reasoning 'Thinking step 1', got %q", resp.Message.Reasoning)
	}
	if resp.Message.ReasoningSignature != "sig_abc" {
		t.Errorf("expected Signature 'sig_abc', got %q", resp.Message.ReasoningSignature)
	}
	if resp.Usage.CacheWriteTokens != 40 || resp.Usage.CacheReadTokens != 60 {
		t.Errorf("expected cache tokens 40/60, got %+v", resp.Usage)
	}

	// 2. Verify request body formatting
	messages, ok := capturedBody["messages"].([]any)
	if !ok {
		t.Fatalf("messages not a slice: %T", capturedBody["messages"])
	}

	// Should have exactly 3 messages: user, assistant, user (the 2 tool results collapsed into 1 user message!)
	if len(messages) != 3 {
		t.Fatalf("expected 3 collapsed messages, got %d", len(messages))
	}

	// Check assistant turn: thinking block must precede tool_use
	asstMsg := messages[1].(map[string]any)
	if asstMsg["role"] != "assistant" {
		t.Errorf("expected assistant role, got %v", asstMsg["role"])
	}
	asstContent := asstMsg["content"].([]any)
	if len(asstContent) < 3 {
		t.Fatalf("expected at least 3 blocks (thinking + 2 tool_use), got %d", len(asstContent))
	}
	firstBlock := asstContent[0].(map[string]any)
	if firstBlock["type"] != "thinking" || firstBlock["thinking"] != "Prior thought" || firstBlock["signature"] != "sig_123" {
		t.Errorf("expected first block to be thinking block with signature, got %+v", firstBlock)
	}

	// Check collapsed user turn: should have both tool results in one message
	userToolMsg := messages[2].(map[string]any)
	if userToolMsg["role"] != "user" {
		t.Errorf("expected user role for tool results, got %v", userToolMsg["role"])
	}
	toolResults := userToolMsg["content"].([]any)
	if len(toolResults) != 2 {
		t.Fatalf("expected 2 tool_results in single user turn, got %d", len(toolResults))
	}

	// Check second tool result has image block
	secondRes := toolResults[1].(map[string]any)
	if secondRes["tool_use_id"] != "c2" {
		t.Errorf("expected tool_use_id c2, got %v", secondRes["tool_use_id"])
	}
	secondContent, ok := secondRes["content"].([]any)
	if !ok || len(secondContent) != 2 {
		t.Fatalf("expected multimodal content array with 2 parts, got %+v", secondRes["content"])
	}
	imgBlock := secondContent[1].(map[string]any)
	if imgBlock["type"] != "image" {
		t.Errorf("expected image block, got %v", imgBlock["type"])
	}
}

func TestAnthropicPromptCache(t *testing.T) {
	var capturedBody map[string]any

	mockClient := &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			_ = json.NewDecoder(r.Body).Decode(&capturedBody)
			respJSON := `{"id":"m1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}`
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(respJSON)),
			}, nil
		}),
	}

	p := New(Config{
		APIKey:      "ant-key",
		PromptCache: true,
		HTTPClient:  mockClient,
	})

	req := &react.Request{
		Model:    "claude-3-5-sonnet",
		System:   "You are an assistant.",
		Messages: []react.Message{react.UserText("hello")},
		Tools: []react.ToolSpec{
			{Name: "tool1", Description: "d1"},
			{Name: "tool2", Description: "d2"},
		},
	}

	_, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}

	// System should have cache_control
	sysBlocks, ok := capturedBody["system"].([]any)
	if !ok || len(sysBlocks) == 0 {
		t.Fatalf("expected system array with cache_control, got %+v", capturedBody["system"])
	}
	sysFirst := sysBlocks[0].(map[string]any)
	if sysFirst["cache_control"] == nil {
		t.Errorf("expected system cache_control, got nil")
	}

	// Last tool should have cache_control
	tools, ok := capturedBody["tools"].([]any)
	if !ok || len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %+v", capturedBody["tools"])
	}
	tool2 := tools[1].(map[string]any)
	if tool2["cache_control"] == nil {
		t.Errorf("expected last tool to have cache_control, got nil")
	}
}

func TestAnthropicImageURL(t *testing.T) {
	var capturedBody map[string]any
	mockClient := &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			b, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(b, &capturedBody)
			resp := messagesResponse{
				ID:         "msg_img",
				Type:       "message",
				Role:       "assistant",
				Content:    []contentBlock{{Type: "text", Text: "image seen"}},
				StopReason: "end_turn",
			}
			data, _ := json.Marshal(resp)
			return &http.Response{
				StatusCode: 200,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(string(data))),
			}, nil
		}),
	}

	p := New(Config{
		APIKey:     "ant-key",
		HTTPClient: mockClient,
	})

	userMsg := react.Message{
		Role: react.RoleUser,
		Content: []react.Part{
			react.TextPart("look at this:"),
			react.ImageURLPart("https://example.com/diagram.png"),
		},
	}
	toolMsg := react.ToolResultMessage("tu_1", "camera", []react.Part{
		react.ImageURLPart("https://example.com/photo.png"),
	}, false)

	req := &react.Request{
		Model:    "claude-3-5-sonnet",
		Messages: []react.Message{userMsg, toolMsg},
	}

	_, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}

	msgs, ok := capturedBody["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("expected 1 collapsed user message, got %+v", capturedBody["messages"])
	}

	// Verify user message image
	userParts := msgs[0].(map[string]any)["content"].([]any)
	if len(userParts) < 3 {
		t.Fatalf("expected at least 3 content blocks, got %+v", userParts)
	}
	imgPart := userParts[1].(map[string]any)
	if imgPart["type"] != "image" {
		t.Fatalf("expected type image, got %v", imgPart["type"])
	}
	source := imgPart["source"].(map[string]any)
	if source["type"] != "url" || source["url"] != "https://example.com/diagram.png" {
		t.Fatalf("expected URL source, got %+v", source)
	}

	// Verify tool result image
	toolRes := userParts[2].(map[string]any)
	toolContentList := toolRes["content"].([]any)
	toolImgPart := toolContentList[0].(map[string]any)
	toolImgSource := toolImgPart["source"].(map[string]any)
	if toolImgSource["type"] != "url" || toolImgSource["url"] != "https://example.com/photo.png" {
		t.Fatalf("expected tool URL source, got %+v", toolImgSource)
	}
}
