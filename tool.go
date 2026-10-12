// Tool defines the tool abstraction, typed function tools with automatic JSON
// Schema generation, a registry, middleware, and a parallel executor.
package react

import (
	"context"
	"encoding/json"
	"strings"
)

// Tool is something the model can call.
type Tool interface {
	Spec() ToolSpec
	Call(ctx context.Context, args json.RawMessage) (ToolResult, error)
}

// ToolSpec describes a tool to the model and to the agent.
type ToolSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
	Annotations ToolAnnotations `json:"annotations"`
}

// ToolAnnotations are hints used by the agent's permission policy and executor.
type ToolAnnotations struct {
	// ReadOnly tools have no side effects and may be auto-approved.
	ReadOnly bool
	// Destructive tools modify state and require approval under the default policy.
	Destructive bool
	// Sequential tools never run concurrently with other tool calls.
	Sequential bool
}

// ToolResult is the outcome of a tool call.
type ToolResult struct {
	Content []Part
	IsError bool
	// Meta carries data for hooks/observers; it is never sent to the model.
	Meta map[string]any
}

// Text concatenates the text parts of the result.
func (r ToolResult) Text() string {
	var b strings.Builder
	for _, p := range r.Content {
		if p.Type == PartText {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// TextResult returns a successful text result.
func TextResult(s string) ToolResult { return ToolResult{Content: []Part{TextPart(s)}} }

// ErrorResult returns a failed text result.
func ErrorResult(s string) ToolResult {
	return ToolResult{Content: []Part{TextPart(s)}, IsError: true}
}

// funcTool is a Tool backed by a raw function.
type funcTool struct {
	spec ToolSpec
	fn   func(context.Context, json.RawMessage) (ToolResult, error)
}

func (t *funcTool) Spec() ToolSpec { return t.spec }
func (t *funcTool) Call(ctx context.Context, args json.RawMessage) (ToolResult, error) {
	return t.fn(ctx, args)
}

// NewRawTool builds a tool from an explicit spec and a raw-JSON handler. If
// spec.InputSchema is empty, an empty object schema is used.
func NewRawTool(spec ToolSpec, fn func(ctx context.Context, args json.RawMessage) (ToolResult, error)) Tool {
	if len(spec.InputSchema) == 0 {
		spec.InputSchema = json.RawMessage(`{"type":"object","properties":{}}`)
	}
	return &funcTool{spec: spec, fn: fn}
}
