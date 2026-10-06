// Package hook defines lifecycle hooks and the per-run context shared between
// the agent, hooks, plugins and tools.
package hook

import (
	"context"
	"sync"

	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/tool"
)

// RunContext describes the current agent run. Fields are set by the agent;
// hooks should treat them as read-only and use Set/Get for custom state.
type RunContext struct {
	AgentName string
	SessionID string
	RunID     string
	Turn      int
	Depth     int // 0 for the top-level agent, +1 per sub-agent level

	mu     sync.Mutex
	values map[string]any
}

// Set stores run-scoped state (concurrency-safe).
func (rc *RunContext) Set(key string, v any) {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	if rc.values == nil {
		rc.values = map[string]any{}
	}
	rc.values[key] = v
}

// Get loads run-scoped state.
func (rc *RunContext) Get(key string) (any, bool) {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	v, ok := rc.values[key]
	return v, ok
}

type rcKey struct{}

// WithRunContext attaches rc to ctx.
func WithRunContext(ctx context.Context, rc *RunContext) context.Context {
	return context.WithValue(ctx, rcKey{}, rc)
}

// RunContextFrom returns the RunContext attached to ctx, or nil.
func RunContextFrom(ctx context.Context) *RunContext {
	rc, _ := ctx.Value(rcKey{}).(*RunContext)
	return rc
}

// Action is what a BeforeTool hook wants to happen.
type Action int

const (
	// Continue lets the call proceed.
	Continue Action = iota
	// Deny blocks the call; Reason is reported to the model as an error.
	Deny
	// Override skips execution and uses Result instead.
	Override
)

// ToolDecision is returned by BeforeTool.
type ToolDecision struct {
	Action Action
	Reason string
	Result tool.Result
}

// Hooks is a set of optional lifecycle callbacks. Any field may be nil.
//
// Tool-related hooks may be invoked concurrently when tools run in parallel.
// OnEvent calls are serialized by the agent. Returning an error from
// OnRunStart, BeforeModel or AfterModel aborts the run.
type Hooks struct {
	OnRunStart  func(ctx context.Context, rc *RunContext) error
	BeforeModel func(ctx context.Context, rc *RunContext, req *llm.Request) error
	AfterModel  func(ctx context.Context, rc *RunContext, resp *llm.Response) error
	BeforeTool  func(ctx context.Context, rc *RunContext, call *llm.ToolCall) (ToolDecision, error)
	AfterTool   func(ctx context.Context, rc *RunContext, call llm.ToolCall, res *tool.Result) error
	// OnEvent receives every agent.Event (typed as any to avoid an import cycle).
	OnEvent  func(ctx context.Context, ev any)
	OnRunEnd func(ctx context.Context, rc *RunContext, err error)
}

// Chain runs multiple Hooks in order.
type Chain []Hooks

func (c Chain) RunStart(ctx context.Context, rc *RunContext) error {
	for _, h := range c {
		if h.OnRunStart != nil {
			if err := h.OnRunStart(ctx, rc); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c Chain) BeforeModel(ctx context.Context, rc *RunContext, req *llm.Request) error {
	for _, h := range c {
		if h.BeforeModel != nil {
			if err := h.BeforeModel(ctx, rc, req); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c Chain) AfterModel(ctx context.Context, rc *RunContext, resp *llm.Response) error {
	for _, h := range c {
		if h.AfterModel != nil {
			if err := h.AfterModel(ctx, rc, resp); err != nil {
				return err
			}
		}
	}
	return nil
}

// BeforeTool returns the first non-Continue decision.
func (c Chain) BeforeTool(ctx context.Context, rc *RunContext, call *llm.ToolCall) (ToolDecision, error) {
	for _, h := range c {
		if h.BeforeTool != nil {
			d, err := h.BeforeTool(ctx, rc, call)
			if err != nil || d.Action != Continue {
				return d, err
			}
		}
	}
	return ToolDecision{}, nil
}

func (c Chain) AfterTool(ctx context.Context, rc *RunContext, call llm.ToolCall, res *tool.Result) error {
	for _, h := range c {
		if h.AfterTool != nil {
			if err := h.AfterTool(ctx, rc, call, res); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c Chain) Event(ctx context.Context, ev any) {
	for _, h := range c {
		if h.OnEvent != nil {
			h.OnEvent(ctx, ev)
		}
	}
}

func (c Chain) RunEnd(ctx context.Context, rc *RunContext, err error) {
	for _, h := range c {
		if h.OnRunEnd != nil {
			h.OnRunEnd(ctx, rc, err)
		}
	}
}
