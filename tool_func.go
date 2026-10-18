package react

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// ToolOption customises a typed tool's spec.
type ToolOption func(*ToolSpec)

// WithToolAnnotations sets the tool's annotations.
func WithToolAnnotations(a ToolAnnotations) ToolOption {
	return func(s *ToolSpec) { s.Annotations = a }
}

// ReadOnly marks the tool as side-effect free.
func ReadOnly() ToolOption { return func(s *ToolSpec) { s.Annotations.ReadOnly = true } }

// Destructive marks the tool as requiring approval.
func Destructive() ToolOption { return func(s *ToolSpec) { s.Annotations.Destructive = true } }

// Sequential prevents the tool from running concurrently with others.
func Sequential() ToolOption { return func(s *ToolSpec) { s.Annotations.Sequential = true } }

// WithToolSchema overrides the generated input schema.
func WithToolSchema(schema json.RawMessage) ToolOption {
	return func(s *ToolSpec) { s.InputSchema = schema }
}

// WithToolMaxOutput overrides the global tool output limit for this tool. The
// bounded copy is used for the event and the message history. A value <= 0
// inherits WithMaxToolOutput.
func WithToolMaxOutput(n int) ToolOption {
	return func(s *ToolSpec) { s.MaxOutput = n }
}

// New builds a tool from a typed Go function. The input schema is derived
// from In (see SchemaFor). Out is converted to the result as follows:
// Result is used as-is, string becomes text, []Part becomes content, and
// anything else is JSON-encoded.
//
// Errors returned by fn are surfaced to the model as error results by the
// agent's executor.
func NewTool[In, Out any](name, description string, fn func(ctx context.Context, in In) (Out, error), opts ...ToolOption) Tool {
	spec := ToolSpec{Name: name, Description: description, InputSchema: SchemaFor[In]()}
	for _, o := range opts {
		o(&spec)
	}
	return &funcTool{spec: spec, fn: func(ctx context.Context, args json.RawMessage) (ToolResult, error) {
		var in In
		if s := strings.TrimSpace(string(args)); s != "" && s != "null" {
			if err := json.Unmarshal(args, &in); err != nil {
				return ErrorResult(fmt.Sprintf("invalid arguments for %s: %v", name, err)), nil
			}
		}
		out, err := fn(ctx, in)
		if err != nil {
			return ToolResult{}, err
		}
		return toResult(out)
	}}
}

func toResult(v any) (ToolResult, error) {
	switch o := v.(type) {
	case ToolResult:
		return o, nil
	case *ToolResult:
		if o == nil {
			return ToolResult{}, nil
		}
		return *o, nil
	case string:
		return TextResult(o), nil
	case []Part:
		return ToolResult{Content: o}, nil
	case nil:
		return ToolResult{}, nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return ToolResult{}, fmt.Errorf("encode tool output: %w", err)
	}
	return TextResult(string(b)), nil
}
