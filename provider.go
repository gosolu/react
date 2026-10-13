package react

import (
	"context"
	"encoding/json"
	"iter"
)

// StopReason explains why the model or agent stopped generating.
type StopReason string

const (
	StopEnd           StopReason = "stop"           // Model finished naturally or hit a stop sequence
	StopToolCall      StopReason = "tool_call"      // Model requested tool or function execution
	StopLength        StopReason = "length"         // Generation hit token limit or maximum length
	StopContentFilter StopReason = "content_filter" // Content filter omitted or blocked generation
	StopMaxTurns      StopReason = "max_turns"      // Agent stopped because max turn limit was reached
	StopOther         StopReason = "other"          // Other provider-specific reason
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
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
}

// Add accumulates o into u.
func (u *Usage) Add(o Usage) {
	u.InputTokens += o.InputTokens
	u.OutputTokens += o.OutputTokens
	u.CacheReadTokens += o.CacheReadTokens
	u.CacheWriteTokens += o.CacheWriteTokens
}

// Total returns input + output tokens.
func (u Usage) Total() int { return u.InputTokens + u.OutputTokens }

// ResponseFormatType defines the desired output format of the model.
type ResponseFormatType string

const (
	ResponseFormatText       ResponseFormatType = "text"
	ResponseFormatJSONObject ResponseFormatType = "json_object"
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

// ResponseFormat configures structured output from the model.
type ResponseFormat struct {
	Type       ResponseFormatType `json:"type"`
	JSONSchema json.RawMessage    `json:"json_schema,omitempty"`
	Name       string             `json:"name,omitempty"`
	Strict     bool               `json:"strict,omitempty"`
}

// Request is a provider-agnostic model request.
type Request struct {
	Model          string
	System         string
	Messages       []Message
	Tools          []ToolSpec
	ToolChoice     string // "", auto, none, required, or a tool name
	MaxTokens      int
	Temperature    *float64
	ResponseFormat *ResponseFormat
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
