package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/gosolu/react/tool"
)

type depthKey struct{}
type emitterKey struct{}

func depthFrom(ctx context.Context) int {
	d, _ := ctx.Value(depthKey{}).(int)
	return d
}

func emitterFrom(ctx context.Context) emitFunc {
	e, _ := ctx.Value(emitterKey{}).(emitFunc)
	return e
}

type subAgentInput struct {
	Task string `json:"task" desc:"A complete, self-contained description of the task to delegate, including all needed context."`
}

// AsTool exposes the agent as a tool that another agent can call to delegate
// work. Each call runs in a fresh, ephemeral session; the tool result is the
// sub-agent's final answer. The sub-agent's events are forwarded to the
// parent's event stream with Depth incremented, and nesting is bounded by
// WithMaxDepth.
func (a *Agent) AsTool(name, description string) tool.Tool {
	return tool.New(name, description, func(ctx context.Context, in subAgentInput) (string, error) {
		if strings.TrimSpace(in.Task) == "" {
			return "", errors.New("task is required")
		}
		cctx := context.WithValue(ctx, depthKey{}, depthFrom(ctx)+1)
		var opts []RunOption
		if parent := emitterFrom(ctx); parent != nil {
			opts = append(opts, WithEventHandler(func(ev Event) { parent(ev) }))
		}
		res, err := a.Run(cctx, "", in.Task, opts...)
		if err != nil {
			if res != nil && res.Output != "" {
				return fmt.Sprintf("%s\n\n(sub-agent stopped early: %v)", res.Output, err), nil
			}
			return "", fmt.Errorf("sub-agent %s: %w", a.cfg.name, err)
		}
		return res.Output, nil
	})
}
