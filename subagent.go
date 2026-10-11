package react

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

type depthKey struct{}
type emitterKey struct{}
type hooksKey struct{}

func depthFrom(ctx context.Context) int {
	d, _ := ctx.Value(depthKey{}).(int)
	return d
}

func emitterFrom(ctx context.Context) emitFunc {
	e, _ := ctx.Value(emitterKey{}).(emitFunc)
	return e
}

func withHooks(ctx context.Context, h HookChain) context.Context {
	return context.WithValue(ctx, hooksKey{}, h)
}

func hooksFrom(ctx context.Context) HookChain {
	h, _ := ctx.Value(hooksKey{}).(HookChain)
	return h
}

type subAgentInput struct {
	Task string `json:"task" desc:"A complete, self-contained description of the task to delegate, including all needed context."`
}

// AsTool exposes the agent as a tool that another agent can call to delegate
// work. Each call runs in a fresh, ephemeral session; the tool result is the
// sub-agent's final answer. The sub-agent's events are forwarded to the
// parent's event stream with Depth incremented, and nesting is bounded by
// WithMaxDepth.
func (a *Agent) AsTool(name, description string) Tool {
	return NewTool(name, description, func(ctx context.Context, in subAgentInput) (string, error) {
		if strings.TrimSpace(in.Task) == "" {
			return "", errors.New("task is required")
		}
		rc := RunContextFrom(ctx)
		parentHooks := hooksFrom(ctx)
		parentHooks.SubagentStart(ctx, rc, name, in.Task)
		a.hooks.SubagentStart(ctx, rc, name, in.Task)
		var subErr error
		var output string
		defer func() {
			parentHooks.SubagentStop(ctx, rc, name, in.Task, output, subErr)
			a.hooks.SubagentStop(ctx, rc, name, in.Task, output, subErr)
		}()

		cctx := context.WithValue(ctx, depthKey{}, depthFrom(ctx)+1)
		var opts []RunOption
		if parent := emitterFrom(ctx); parent != nil {
			opts = append(opts, WithEventHandler(func(ev Event) { parent(ev) }))
		}
		res, err := a.Run(cctx, "", in.Task, opts...)
		if err != nil {
			subErr = err
			if res != nil && res.Output != "" {
				output = res.Output
				return fmt.Sprintf("%s\n\n(sub-agent stopped early: %v)", res.Output, err), nil
			}
			return "", fmt.Errorf("sub-agent %s: %w", a.cfg.name, err)
		}
		output = res.Output
		return res.Output, nil
	})
}
