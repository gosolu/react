package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
)

// ToolSpec describes a tool to the model.
type ToolSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// StopReason explains why the model stopped generating.
type StopReason string

const (
	StopEndTurn   StopReason = "end_turn"
	StopToolUse   StopReason = "tool_use"
	StopMaxTokens StopReason = "max_tokens"
	// StopMaxTurns is set by the agent (not providers) when the turn limit is hit.
	StopMaxTurns StopReason = "max_turns"
	// StopOther covers provider-specific reasons (content filter, stop sequence, ...).
	StopOther StopReason = "other"
)

// Tool choice values for Request.ToolChoice. Any other value is treated as
// the name of a specific tool that must be called.
const (
	ToolChoiceAuto     = "auto"
	ToolChoiceNone     = "none"
	ToolChoiceRequired = "required"
)

// Usage reports token consumption.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Add accumulates o into u.
func (u *Usage) Add(o Usage) {
	u.InputTokens += o.InputTokens
	u.OutputTokens += o.OutputTokens
}

// Total returns input + output tokens.
func (u Usage) Total() int { return u.InputTokens + u.OutputTokens }

// Request is a provider-agnostic model request.
type Request struct {
	Model       string
	System      string
	Messages    []Message
	Tools       []ToolSpec
	ToolChoice  string // "", auto, none, required, or a tool name
	MaxTokens   int
	Temperature *float64
	// Extra is merged into the provider's request body (provider-specific knobs).
	Extra map[string]any
}

// Response is a complete (non-streamed or accumulated) model response.
type Response struct {
	Message    Message
	Reasoning  string // optional reasoning/thinking text, if the provider exposes it
	StopReason StopReason
	Usage      Usage
}

// Provider is implemented by every model backend.
//
// Generate performs a blocking request. Stream returns an iterator of
// incremental events; breaking out of the iteration must release all
// resources (e.g. close the HTTP body).
type Provider interface {
	Name() string
	Generate(ctx context.Context, req *Request) (*Response, error)
	Stream(ctx context.Context, req *Request) iter.Seq2[StreamEvent, error]
}

// APIError is returned when a provider responds with a non-success status.
type APIError struct {
	Provider   string
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	body := e.Body
	if len(body) > 500 {
		body = body[:500] + "..."
	}
	return fmt.Sprintf("%s: HTTP %d: %s", e.Provider, e.StatusCode, body)
}

// Retryable reports whether the request may succeed if retried.
func (e *APIError) Retryable() bool {
	return e.StatusCode == 429 || e.StatusCode >= 500
}
