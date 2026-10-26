package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gosolu/react"
)

func TestGeminiGenerate_Text(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-goog-api-key") != "gemini-key" {
			t.Errorf("missing or bad x-goog-api-key: %v", r.Header.Get("x-goog-api-key"))
		}
		if !strings.Contains(r.URL.Path, "/models/gemini-3.8-flash:generateContent") {
			t.Errorf("unexpected endpoint path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"candidates": [{
				"content": {
					"role": "model",
					"parts": [
						{"text": "Hello from Gemini!"}
					]
				},
				"finishReason": "STOP"
			}],
			"usageMetadata": {
				"promptTokenCount": 10,
				"candidatesTokenCount": 5,
				"cachedContentTokenCount": 2
			}
		}`)
	}))
	defer ts.Close()

	p := New(Config{
		APIKey:  "gemini-key",
		BaseURL: ts.URL,
	})

	if p.Name() != "gemini" {
		t.Fatalf("name = %s, want gemini", p.Name())
	}

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    ModelGemini38Flash,
		Messages: []react.Message{react.UserText("hello")},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp.Message.Text() != "Hello from Gemini!" {
		t.Fatalf("text = %q, want 'Hello from Gemini!'", resp.Message.Text())
	}
	if resp.StopReason != react.StopEnd {
		t.Fatalf("stop = %v, want StopEnd", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 || resp.Usage.CacheReadTokens != 2 {
		t.Fatalf("unexpected usage: %+v", resp.Usage)
	}
}

func TestGeminiGenerate_ToolCall_StopReason(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Notice: Gemini returns finishReason: "STOP" even with tool calls!
		_, _ = fmt.Fprintln(w, `{
			"candidates": [{
				"content": {
					"role": "model",
					"parts": [
						{
							"functionCall": {
								"id": "call_123",
								"name": "get_weather",
								"args": {"city": "Berlin"}
							},
							"thoughtSignature": "sig_abc123"
						}
					]
				},
				"finishReason": "STOP"
			}],
			"usageMetadata": {
				"promptTokenCount": 18,
				"candidatesTokenCount": 8
			}
		}`)
	}))
	defer ts.Close()

	p := New(Config{
		APIKey:  "gemini-key",
		BaseURL: ts.URL,
	})

	resp, err := p.Generate(context.Background(), &react.Request{
		Model:    ModelGemini4Argon,
		Messages: []react.Message{react.UserText("weather in Berlin?")},
		Tools: []react.ToolSpec{
			{Name: "get_weather", Description: "Get current weather"},
		},
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// Must override STOP to react.StopToolCall when ToolCalls are present!
	if resp.StopReason != react.StopToolCall {
		t.Fatalf("stopReason = %v, want react.StopToolCall", resp.StopReason)
	}

	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "call_123" || tc.Name != "get_weather" {
		t.Fatalf("unexpected tool call: %+v", tc)
	}
	if tc.ThoughtSignature != "sig_abc123" {
		t.Fatalf("thoughtSignature = %q, want 'sig_abc123'", tc.ThoughtSignature)
	}
	if !strings.Contains(string(tc.Arguments), `"Berlin"`) {
		t.Fatalf("args = %s, want 'Berlin'", string(tc.Arguments))
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestGeminiMultiTurn_ThoughtSignature_And_Collapsing(t *testing.T) {
	var capturedBody map[string]any

	mockClient := &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			_ = json.NewDecoder(r.Body).Decode(&capturedBody)
			respJSON := `{
				"candidates": [{
					"content": {
						"role": "model",
						"parts": [
							{"text": "The weather is 20C and stock is 150."}
						]
					},
					"finishReason": "STOP"
				}],
				"usageMetadata": {
					"promptTokenCount": 50,
					"candidatesTokenCount": 15
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
		APIKey:     "gemini-key",
		HTTPClient: mockClient,
	})

	req := &react.Request{
		Model:  ModelGemini4Argon,
		System: "You are a test assistant.",
		Messages: []react.Message{
			react.UserText("Check weather and stock"),
			{
				Role:               react.RoleAssistant,
				Reasoning:          "Thinking through both requests...",
				ReasoningSignature: "sig_thought",
				ToolCalls: []react.ToolCall{
					{
						ID:               "c1",
						Name:             "get_weather",
						Arguments:        []byte(`{"city":"Tokyo"}`),
						ThoughtSignature: "sig_fc1",
					},
					{
						ID:               "c2",
						Name:             "get_stock",
						Arguments:        []byte(`{"symbol":"GOOG"}`),
						ThoughtSignature: "sig_fc2",
					},
				},
			},
			{
				Role:       react.RoleTool,
				ToolCallID: "c1",
				// Name omitted intentionally to test fallback lookup by ToolCallID!
				Content: []react.Part{react.TextPart(`{"temp": "20C"}`)},
			},
			{
				Role:       react.RoleTool,
				ToolCallID: "c2",
				Name:       "get_stock",
				Content:    []react.Part{react.TextPart("150 USD")}, // plain text: should wrap as {"result": "150 USD"}
			},
		},
	}

	resp, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "The weather is 20C and stock is 150." {
		t.Fatalf("text = %q", resp.Message.Text())
	}

	// 1. Verify systemInstruction
	sysInst, ok := capturedBody["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("missing systemInstruction in body: %+v", capturedBody)
	}
	sysParts := sysInst["parts"].([]any)
	if sysParts[0].(map[string]any)["text"] != "You are a test assistant." {
		t.Fatalf("system text mismatch: %+v", sysParts)
	}

	// 2. Verify contents array: user, model, user (the two tool results MUST be collapsed into 1 user turn!)
	contents, ok := capturedBody["contents"].([]any)
	if !ok || len(contents) != 3 {
		t.Fatalf("expected 3 contents turns (user, model, user), got %d: %+v", len(contents), contents)
	}

	// Model turn: must contain thought part + 2 functionCall parts with exact thoughtSignatures
	modelTurn := contents[1].(map[string]any)
	if modelTurn["role"] != "model" {
		t.Fatalf("expected role 'model', got %v", modelTurn["role"])
	}
	modelParts := modelTurn["parts"].([]any)
	if len(modelParts) != 3 {
		t.Fatalf("expected 3 model parts (1 thought + 2 functionCalls), got %d: %+v", len(modelParts), modelParts)
	}
	thoughtPart := modelParts[0].(map[string]any)
	if thoughtPart["thought"] != true || thoughtPart["text"] != "Thinking through both requests..." || thoughtPart["thoughtSignature"] != "sig_thought" {
		t.Fatalf("thought part mismatch: %+v", thoughtPart)
	}

	fcPart1 := modelParts[1].(map[string]any)
	if fcPart1["thoughtSignature"] != "sig_fc1" {
		t.Fatalf("fc1 thoughtSignature mismatch: %+v", fcPart1)
	}
	fc1 := fcPart1["functionCall"].(map[string]any)
	if fc1["id"] != "c1" || fc1["name"] != "get_weather" {
		t.Fatalf("fc1 content mismatch: %+v", fc1)
	}

	fcPart2 := modelParts[2].(map[string]any)
	if fcPart2["thoughtSignature"] != "sig_fc2" {
		t.Fatalf("fc2 thoughtSignature mismatch: %+v", fcPart2)
	}

	// Collapsed User turn: both functionResponse parts must be present
	toolTurn := contents[2].(map[string]any)
	if toolTurn["role"] != "user" {
		t.Fatalf("expected collapsed role 'user', got %v", toolTurn["role"])
	}
	toolParts := toolTurn["parts"].([]any)
	if len(toolParts) != 2 {
		t.Fatalf("expected 2 collapsed tool parts in single user turn, got %d", len(toolParts))
	}

	// First functionResponse: JSON object passed as object
	frPart1 := toolParts[0].(map[string]any)["functionResponse"].(map[string]any)
	if frPart1["name"] != "get_weather" || frPart1["id"] != "c1" {
		t.Fatalf("fr1 name/id mismatch: %+v", frPart1)
	}
	respMap1, ok := frPart1["response"].(map[string]any)
	if !ok || respMap1["temp"] != "20C" {
		t.Fatalf("fr1 response should be map with temp=20C: %+v", frPart1["response"])
	}

	// Second functionResponse: plain text wrapped as {"result": "150 USD"}
	frPart2 := toolParts[1].(map[string]any)["functionResponse"].(map[string]any)
	if frPart2["name"] != "get_stock" || frPart2["id"] != "c2" {
		t.Fatalf("fr2 name/id mismatch: %+v", frPart2)
	}
	respMap2, ok := frPart2["response"].(map[string]any)
	if !ok || respMap2["result"] != "150 USD" {
		t.Fatalf("fr2 response should be wrapped as result: %+v", frPart2["response"])
	}
}

func TestGeminiToolChoice_And_Schema(t *testing.T) {
	var capturedBody map[string]any

	mockClient := &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			_ = json.NewDecoder(r.Body).Decode(&capturedBody)
			respJSON := `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(respJSON)),
			}, nil
		}),
	}

	p := New(Config{
		APIKey:     "gemini-key",
		HTTPClient: mockClient,
	})

	// Schema with $schema and additionalProperties that Gemini rejects
	rawSchema := []byte(`{
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"type": "object",
		"properties": {
			"query": {
				"type": "string",
				"description": "search query"
			}
		},
		"required": ["query"],
		"additionalProperties": false
	}`)

	req := &react.Request{
		Model:      ModelGemini4Argon,
		ToolChoice: "search_tool",
		Tools: []react.ToolSpec{
			{
				Name:        "search_tool",
				Description: "Search internet",
				InputSchema: rawSchema,
			},
		},
		Messages: []react.Message{react.UserText("search for Go")},
	}

	_, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// Check toolConfig
	toolConfig, ok := capturedBody["toolConfig"].(map[string]any)
	if !ok {
		t.Fatalf("toolConfig missing: %+v", capturedBody)
	}
	fcc := toolConfig["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" {
		t.Errorf("expected mode ANY, got %v", fcc["mode"])
	}
	allowed := fcc["allowedFunctionNames"].([]any)
	if len(allowed) != 1 || allowed[0] != "search_tool" {
		t.Errorf("allowedFunctionNames mismatch: %+v", allowed)
	}

	// Check cleaned parameters schema: $schema and additionalProperties must be removed!
	toolsList := capturedBody["tools"].([]any)
	decls := toolsList[0].(map[string]any)["functionDeclarations"].([]any)
	decl := decls[0].(map[string]any)
	params := decl["parameters"].(map[string]any)
	if _, exists := params["$schema"]; exists {
		t.Errorf("$schema should have been stripped from parameters: %+v", params)
	}
	if _, exists := params["additionalProperties"]; exists {
		t.Errorf("additionalProperties false should have been stripped: %+v", params)
	}
}

func TestGeminiStream(t *testing.T) {
	sseData := `data: {"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":"Thinking step 1...","thoughtSignature":"sig_stream_t"}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Result: "}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"call_s1","name":"calc","args":{"a":10,"b":20}},"thoughtSignature":"sig_stream_fc"}]}}]}

data: {"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":12,"thoughtsTokenCount":8}}

`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := New(Config{
		APIKey:  "gemini-key",
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
	if resp.Message.Text() != "Result: " {
		t.Fatalf("text = %q, want 'Result: '", resp.Message.Text())
	}
	if resp.Reasoning != "Thinking step 1..." {
		t.Fatalf("reasoning = %q, want 'Thinking step 1...'", resp.Reasoning)
	}
	if resp.Message.ReasoningSignature != "sig_stream_t" {
		t.Fatalf("reasoningSignature = %q, want 'sig_stream_t'", resp.Message.ReasoningSignature)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "call_s1" || tc.Name != "calc" {
		t.Fatalf("tc mismatch: %+v", tc)
	}
	if tc.ThoughtSignature != "sig_stream_fc" {
		t.Fatalf("thoughtSignature = %q, want 'sig_stream_fc'", tc.ThoughtSignature)
	}
	// FinishReason STOP with tool calls must be mapped to StopToolCall!
	if resp.StopReason != react.StopToolCall {
		t.Fatalf("stop = %v, want StopToolCall", resp.StopReason)
	}
	if resp.Usage.InputTokens != 25 || resp.Usage.OutputTokens != 20 { // 12 + 8
		t.Fatalf("usage = %+v, want input=25, output=20", resp.Usage)
	}
}

func TestGeminiFinishReasons(t *testing.T) {
	cases := []struct {
		finishReason string
		expected     react.StopReason
	}{
		{"STOP", react.StopEnd},
		{"MAX_TOKENS", react.StopLength},
		{"SAFETY", react.StopContentFilter},
		{"RECITATION", react.StopContentFilter},
		{"BLOCKLIST", react.StopContentFilter},
		{"UNKNOWN_OTHER", react.StopOther},
	}

	for _, tc := range cases {
		t.Run(tc.finishReason, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprintf(w, `{
					"candidates": [{
						"content": {"role": "model", "parts": [{"text": "done"}]},
						"finishReason": %q
					}]
				}`, tc.finishReason)
			}))
			defer ts.Close()

			p := New(Config{APIKey: "k", BaseURL: ts.URL})
			resp, err := p.Generate(context.Background(), &react.Request{Model: ModelGemini38Flash, Messages: []react.Message{react.UserText("test")}})
			if err != nil {
				t.Fatal(err)
			}
			if resp.StopReason != tc.expected {
				t.Fatalf("finishReason %q got %v, want %v", tc.finishReason, resp.StopReason, tc.expected)
			}
		})
	}
}

func TestGeminiMultimodalInput(t *testing.T) {
	var capturedBody map[string]any

	mockClient := &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			_ = json.NewDecoder(r.Body).Decode(&capturedBody)
			respJSON := `{"candidates":[{"content":{"role":"model","parts":[{"text":"saw images"}]},"finishReason":"STOP"}]}`
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(respJSON)),
			}, nil
		}),
	}

	p := New(Config{
		APIKey:     "gemini-key",
		HTTPClient: mockClient,
	})

	req := &react.Request{
		Model: ModelGemini4Flash,
		Messages: []react.Message{
			{
				Role: react.RoleUser,
				Content: []react.Part{
					react.TextPart("Analyze this:"),
					react.ImageDataPart("image/png", []byte("raw-png-bytes")),
					react.ImageURLPart("https://example.com/chart.jpg"),
				},
			},
		},
	}

	resp, err := p.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if resp.Message.Text() != "saw images" {
		t.Fatalf("text = %q", resp.Message.Text())
	}

	contents := capturedBody["contents"].([]any)
	parts := contents[0].(map[string]any)["parts"].([]any)
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts, got %d: %+v", len(parts), parts)
	}

	// Part 2: inlineData
	inlinePart := parts[1].(map[string]any)["inlineData"].(map[string]any)
	if inlinePart["mimeType"] != "image/png" {
		t.Errorf("mimeType = %v, want image/png", inlinePart["mimeType"])
	}

	// Part 3: fileData
	filePart := parts[2].(map[string]any)["fileData"].(map[string]any)
	if filePart["fileUri"] != "https://example.com/chart.jpg" {
		t.Errorf("fileUri = %v, want https://example.com/chart.jpg", filePart["fileUri"])
	}
}

func TestGeminiChatCompletionsMode(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/openai/chat/completions" {
			t.Errorf("path = %q, want /openai/chat/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintln(w, `{
			"choices": [{
				"message": {"content": "Hello from Gemini Chat Completions"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 6}
		}`)
	}))
	defer ts.Close()

	p := NewChatCompletions(Config{
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
	if resp.Message.Text() != "Hello from Gemini Chat Completions" {
		t.Fatalf("unexpected text: %q", resp.Message.Text())
	}
}

func TestGeminiStream_Error(t *testing.T) {
	sseData := `data: {"error":{"code":400,"message":"Invalid request argument"}}
`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := New(Config{
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

func TestGeminiStream_UnexpectedEOF(t *testing.T) {
	sseData := `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Incomplete response"}]}}]}
`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, sseData)
	}))
	defer ts.Close()

	p := New(Config{
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
		t.Fatal("expected io.ErrUnexpectedEOF, got nil")
	}
	if !errors.Is(errFound, io.ErrUnexpectedEOF) {
		t.Fatalf("expected io.ErrUnexpectedEOF, got %v", errFound)
	}
}
