package react

import (
	"context"
	"encoding/json"
	"fmt"
)

// RunStructured executes a run constrained by a JSON schema derived from T,
// and unmarshals the final model output into an instance of T.
func RunStructured[T any](ctx context.Context, a *Agent, sessionID, input string, opts ...RunOption) (T, *Result, error) {
	var zero T
	schema := SchemaFor[T]()

	runOpts := make([]RunOption, 0, len(opts)+1)
	runOpts = append(runOpts, opts...)
	runOpts = append(runOpts, WithResponseFormat(ResponseFormat{
		Type:       ResponseFormatJSONSchema,
		JSONSchema: schema,
		Strict:     true,
	}))

	res, err := a.Run(ctx, sessionID, input, runOpts...)
	if err != nil {
		return zero, res, err
	}

	var val T
	if err := json.Unmarshal([]byte(res.Output), &val); err != nil {
		return zero, res, fmt.Errorf("react: failed to unmarshal structured output into %T: %w (output: %q)", zero, err, res.Output)
	}
	return val, res, nil
}
