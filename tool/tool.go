// Package tool defines the Tool abstraction, typed function tools with
// automatic JSON Schema generation, a registry, middleware, and a parallel
// executor.
package tool

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/gosolu/react/llm"
)

// Tool is something the model can call.
type Tool interface {
	Spec() Spec
	Call(ctx context.Context, args json.RawMessage) (Result, error)
}

// Spec describes a tool.
type Spec struct {
	Name        string
	Description string
	InputSchema json.RawMessage
	Annotations Annotations
}

// LLM converts the spec into the provider-facing form.
func (s Spec) LLM() llm.ToolSpec {
	return llm.ToolSpec{Name: s.Name, Description: s.Description, InputSchema: s.InputSchema}
}

// Annotations are hints used by the agent's permission policy and executor.
type Annotations struct {
	// ReadOnly tools have no side effects and may be auto-approved.
	ReadOnly bool
	// Destructive tools modify state and require approval under the default policy.
	Destructive bool
	// Sequential tools never run concurrently with other tool calls.
	Sequential bool
}

// Result is the outcome of a tool call.
type Result struct {
	Content []llm.Part
	IsError bool
	// Meta carries data for hooks/observers; it is never sent to the model.
	Meta map[string]any
}

// Text concatenates the text parts of the result.
func (r Result) Text() string {
	var b strings.Builder
	for _, p := range r.Content {
		if p.Type == llm.PartText {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// TextResult returns a successful text result.
func TextResult(s string) Result { return Result{Content: []llm.Part{llm.TextPart(s)}} }

// ErrorResult returns a failed text result.
func ErrorResult(s string) Result {
	return Result{Content: []llm.Part{llm.TextPart(s)}, IsError: true}
}

// funcTool is a Tool backed by a raw function.
type funcTool struct {
	spec Spec
	fn   func(context.Context, json.RawMessage) (Result, error)
}

func (t *funcTool) Spec() Spec { return t.spec }
func (t *funcTool) Call(ctx context.Context, args json.RawMessage) (Result, error) {
	return t.fn(ctx, args)
}

// NewRaw builds a tool from an explicit spec and a raw-JSON handler. If
// spec.InputSchema is empty, an empty object schema is used.
func NewRaw(spec Spec, fn func(ctx context.Context, args json.RawMessage) (Result, error)) Tool {
	if len(spec.InputSchema) == 0 {
		spec.InputSchema = json.RawMessage(`{"type":"object","properties":{}}`)
	}
	return &funcTool{spec: spec, fn: fn}
}
