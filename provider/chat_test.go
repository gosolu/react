package provider

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/gosolu/react"
)

func TestStreamChatCompletions_Success(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" world!"}}]}

data: {"choices":[{"finish_reason":"stop"}]}

data: [DONE]
`
	var acc react.Accumulator
	for ev, err := range StreamChatCompletions(context.Background(), strings.NewReader(sseData)) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		acc.Add(ev)
	}
	resp := acc.Response()
	if resp.Message.Text() != "Hello world!" {
		t.Fatalf("got text %q, want 'Hello world!'", resp.Message.Text())
	}
	if resp.StopReason != react.StopEnd {
		t.Fatalf("got stop reason %q, want %q", resp.StopReason, react.StopEnd)
	}
}

func TestStreamChatCompletions_MidStreamErrorPayload(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"content":"Partial"}}]}

data: {"error":{"message":"Server overloaded","type":"server_error","code":503}}
`
	var errFound error
	var textParts []string
	for ev, err := range StreamChatCompletions(context.Background(), strings.NewReader(sseData)) {
		if err != nil {
			errFound = err
			break
		}
		if ev.Type == react.StreamTextDelta {
			textParts = append(textParts, ev.Text)
		}
	}

	if errFound == nil {
		t.Fatal("expected mid-stream error, got nil")
	}
	if !strings.Contains(errFound.Error(), "Server overloaded") {
		t.Fatalf("expected 'Server overloaded' in error, got %v", errFound)
	}
	if strings.Join(textParts, "") != "Partial" {
		t.Fatalf("expected 'Partial' before error, got %v", textParts)
	}
}

func TestStreamChatCompletions_EventError(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"content":"Hello"}}]}

event: error
data: {"message":"Rate limit exceeded"}
`
	var errFound error
	for _, err := range StreamChatCompletions(context.Background(), strings.NewReader(sseData)) {
		if err != nil {
			errFound = err
			break
		}
	}

	if errFound == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(errFound.Error(), "Rate limit exceeded") {
		t.Fatalf("expected rate limit error, got %v", errFound)
	}
}

func TestStreamChatCompletions_UnexpectedEOF(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"content":"Unfinished sentence"}}]}
`
	var errFound error
	for _, err := range StreamChatCompletions(context.Background(), strings.NewReader(sseData)) {
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

func TestStreamChatCompletions_InvalidJSON(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"content":"Hello"}}]}

data: <html>502 Bad Gateway</html>
`
	var errFound error
	for _, err := range StreamChatCompletions(context.Background(), strings.NewReader(sseData)) {
		if err != nil {
			errFound = err
			break
		}
	}

	if errFound == nil {
		t.Fatal("expected invalid json error, got nil")
	}
	if !strings.Contains(errFound.Error(), "invalid json") {
		t.Fatalf("expected 'invalid json' in error, got %v", errFound)
	}
}
