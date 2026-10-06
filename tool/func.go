package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/gosolu/react/llm"
)

// Option customises a typed tool's spec.
type Option func(*Spec)

// WithAnnotations sets the tool's annotations.
func WithAnnotations(a Annotations) Option { return func(s *Spec) { s.Annotations = a } }

// ReadOnly marks the tool as side-effect free.
func ReadOnly() Option { return func(s *Spec) { s.Annotations.ReadOnly = true } }

// Destructive marks the tool as requiring approval.
func Destructive() Option { return func(s *Spec) { s.Annotations.Destructive = true } }

// Sequential prevents the tool from running concurrently with others.
func Sequential() Option { return func(s *Spec) { s.Annotations.Sequential = true } }

// WithSchema overrides the generated input schema.
func WithSchema(schema json.RawMessage) Option { return func(s *Spec) { s.InputSchema = schema } }

// New builds a tool from a typed Go function. The input schema is derived
// from In (see SchemaFor). Out is converted to the result as follows:
// Result is used as-is, string becomes text, []llm.Part becomes content, and
// anything else is JSON-encoded.
//
// Errors returned by fn are surfaced to the model as error results by the
// agent's executor.
func New[In, Out any](name, description string, fn func(ctx context.Context, in In) (Out, error), opts ...Option) Tool {
	spec := Spec{Name: name, Description: description, InputSchema: SchemaFor[In]()}
	for _, o := range opts {
		o(&spec)
	}
	return &funcTool{spec: spec, fn: func(ctx context.Context, args json.RawMessage) (Result, error) {
		var in In
		if s := strings.TrimSpace(string(args)); s != "" && s != "null" {
			if err := json.Unmarshal(args, &in); err != nil {
				return ErrorResult(fmt.Sprintf("invalid arguments for %s: %v", name, err)), nil
			}
		}
		out, err := fn(ctx, in)
		if err != nil {
			return Result{}, err
		}
		return toResult(out)
	}}
}

func toResult(v any) (Result, error) {
	switch o := v.(type) {
	case Result:
		return o, nil
	case *Result:
		if o == nil {
			return Result{}, nil
		}
		return *o, nil
	case string:
		return TextResult(o), nil
	case []llm.Part:
		return Result{Content: o}, nil
	case nil:
		return Result{}, nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return Result{}, fmt.Errorf("encode tool output: %w", err)
	}
	return TextResult(string(b)), nil
}
