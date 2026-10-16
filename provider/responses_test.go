package provider

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/gosolu/react"
)

func TestBuildResponsesRequestBody(t *testing.T) {
	req := &react.Request{
		Model:  "gpt-4o",
		System: "You are a coding assistant.",
		Messages: []react.Message{
			react.UserText("hello"),
			{
				Role: react.RoleAssistant,
				Content: []react.Part{
					react.TextPart("I will run a command"),
				},
				ToolCalls: []react.ToolCall{
					{
						ID:        "call_1",
						Name:      "shell",
						Arguments: []byte(`{"command":"ls"}`),
					},
				},
			},
			{
				Role:       react.RoleTool,
				ToolCallID: "call_1",
				Content:    []react.Part{react.TextPart("file1.go\nfile2.go")},
			},
		},
		Tools: []react.ToolSpec{
			{
				Name:        "shell",
				Description: "Run shell command",
				InputSchema: []byte(`{"type": "object"}`),
			},
		},
		ToolChoice: react.ToolChoiceAuto,
		MaxTokens:  500,
	}

	body := BuildResponsesRequestBody(req, true)

	if body["model"] != "gpt-4o" {
		t.Errorf("model = %v, want gpt-4o", body["model"])
	}
	if body["instructions"] != "You are a coding assistant." {
		t.Errorf("instructions = %v, want 'You are a coding assistant.'", body["instructions"])
	}
	if body["max_output_tokens"] != 500 {
		t.Errorf("max_output_tokens = %v, want 500", body["max_output_tokens"])
	}
	if body["stream"] != true {
		t.Errorf("stream = %v, want true", body["stream"])
	}
	if body["tool_choice"] != "auto" {
		t.Errorf("tool_choice = %v, want auto", body["tool_choice"])
	}

	input, ok := body["input"].([]any)
	if !ok || len(input) != 4 {
		t.Fatalf("input length = %d, want 4 (user, assistant, tool_call, tool_output)", len(input))
	}

	userMsg := input[0].(map[string]any)
	if userMsg["role"] != "user" || userMsg["content"] != "hello" {
		t.Errorf("unexpected user msg: %+v", userMsg)
	}

	asstMsg := input[1].(map[string]any)
	if asstMsg["role"] != "assistant" || asstMsg["content"] != "I will run a command" {
		t.Errorf("unexpected assistant msg: %+v", asstMsg)
	}

	tcItem := input[2].(map[string]any)
	if tcItem["type"] != "function_call" || tcItem["call_id"] != "call_1" || tcItem["name"] != "shell" {
		t.Errorf("unexpected tool call item: %+v", tcItem)
	}

	toolOut := input[3].(map[string]any)
	if toolOut["type"] != "function_call_output" || toolOut["call_id"] != "call_1" || toolOut["output"] != "file1.go\nfile2.go" {
		t.Errorf("unexpected tool output item: %+v", toolOut)
	}
}

func TestParseResponsesResponse(t *testing.T) {
	jsonPayload := `{
		"id": "resp_123",
		"status": "completed",
		"output": [
			{
				"type": "reasoning",
				"content": [{"type": "reasoning_text", "text": "Analyzing the request..."}]
			},
			{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "output_text", "text": "Here is the response."}]
			},
			{
				"type": "function_call",
				"call_id": "call_abc",
				"name": "lookup",
				"arguments": "{\"key\":\"val\"}"
			}
		],
		"usage": {
			"input_tokens": 15,
			"output_tokens": 30
		}
	}`

	resp, err := ParseResponsesResponse(strings.NewReader(jsonPayload))
	if err != nil {
		t.Fatalf("ParseResponsesResponse failed: %v", err)
	}

	if resp.Message.Text() != "Here is the response." {
		t.Errorf("text = %q, want 'Here is the response.'", resp.Message.Text())
	}
	if resp.Reasoning != "Analyzing the request..." {
		t.Errorf("reasoning = %q, want 'Analyzing the request...'", resp.Reasoning)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls count = %d, want 1", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "call_abc" || tc.Name != "lookup" || string(tc.Arguments) != `{"key":"val"}` {
		t.Errorf("unexpected tool call: %+v", tc)
	}
	if resp.StopReason != react.StopToolCall {
		t.Errorf("stop reason = %v, want StopToolCall", resp.StopReason)
	}
	if resp.Usage.InputTokens != 15 || resp.Usage.OutputTokens != 30 {
		t.Errorf("unexpected usage: %+v", resp.Usage)
	}
}

func TestStreamResponses(t *testing.T) {
	sseData := `event: response.reasoning_text.delta
data: {"type":"response.reasoning_text.delta","delta":"Thinking..."}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","role":"assistant"}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":0,"delta":"Hello"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":0,"delta":" world!"}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"call_99","name":"read_file"}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"path\":\"a.txt\"}"}

event: response.completed
data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":5,"output_tokens":12}}}

data: [DONE]
`

	var acc react.Accumulator
	for ev, err := range StreamResponses(context.Background(), strings.NewReader(sseData)) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		acc.Add(ev)
	}

	resp := acc.Response()
	if resp.Message.Text() != "Hello world!" {
		t.Errorf("text = %q, want 'Hello world!'", resp.Message.Text())
	}
	if resp.Reasoning != "Thinking..." {
		t.Errorf("reasoning = %q, want 'Thinking...'", resp.Reasoning)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls count = %d, want 1", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "call_99" || tc.Name != "read_file" || string(tc.Arguments) != `{"path":"a.txt"}` {
		t.Errorf("unexpected tool call: %+v", tc)
	}
	if resp.StopReason != react.StopToolCall {
		t.Errorf("stop reason = %v, want StopToolCall", resp.StopReason)
	}
	if resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 12 {
		t.Errorf("unexpected usage: %+v", resp.Usage)
	}
}

func TestStreamResponses_Error(t *testing.T) {
	sseData := `event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hello"}

event: error
data: {"error":{"message":"Model failure"}}
`
	var errFound error
	for _, err := range StreamResponses(context.Background(), strings.NewReader(sseData)) {
		if err != nil {
			errFound = err
			break
		}
	}
	if errFound == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(errFound.Error(), "Model failure") {
		t.Fatalf("expected 'Model failure' in error, got %v", errFound)
	}
}

func TestStreamResponses_UnexpectedEOF(t *testing.T) {
	sseData := `event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Half finished"}
`
	var errFound error
	for _, err := range StreamResponses(context.Background(), strings.NewReader(sseData)) {
		if err != nil {
			errFound = err
			break
		}
	}
	if errFound == nil {
		t.Fatal("expected io.ErrUnexpectedEOF, got nil")
	}
	if !errors.Is(errFound, io.ErrUnexpectedEOF) {
		t.Fatalf("expected io.ErrUnexpectedEOF, got %v", errFound)
	}
}
