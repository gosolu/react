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
//
// The child runs with the agent's full toolset. For a scoped delegation,
// use AsToolWithAllowedTools.
func (a *Agent) AsTool(name, description string) Tool {
	return a.asTool(name, description, nil, false)
}

// AsToolWithAllowedTools is AsTool with a subtractive per-delegation
// allowlist: the child run sees only the named tools, and calls to any
// other tool are rejected. With no names, the child has no tools at all.
// Names that the agent does not have are ignored. Unlike WithRunTools,
// which is additive only, this can only narrow the child's toolset.
func (a *Agent) AsToolWithAllowedTools(name, description string, allowed ...string) Tool {
	return a.asTool(name, description, allowed, true)
}

func (a *Agent) asTool(name, description string, allowed []string, restrict bool) Tool {
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
		if restrict {
			opts = append(opts, WithAllowedTools(allowed...))
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
