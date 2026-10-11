package react

import (
	"context"
	"sync"

	"github.com/gosolu/react/llm"
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

// HookAction is what a BeforeTool hook wants to happen.
type HookAction int

const (
	// HookContinue lets the call proceed.
	HookContinue HookAction = iota
	// HookDeny blocks the call; Reason is reported to the model as an error.
	HookDeny
	// HookOverride skips execution and uses Result instead.
	HookOverride
)

// HookDecision is returned by BeforeTool.
type HookDecision struct {
	Action HookAction
	Reason string
	Result ToolResult
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
	BeforeTool  func(ctx context.Context, rc *RunContext, call *llm.ToolCall) (HookDecision, error)
	AfterTool   func(ctx context.Context, rc *RunContext, call llm.ToolCall, res *ToolResult) error
	OnEvent     func(ctx context.Context, ev Event)
	OnRunEnd    func(ctx context.Context, rc *RunContext, err error)
}

// HookChain runs multiple Hooks in order.
type HookChain []Hooks

func (c HookChain) RunStart(ctx context.Context, rc *RunContext) error {
	for _, h := range c {
		if h.OnRunStart != nil {
			if err := h.OnRunStart(ctx, rc); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c HookChain) BeforeModel(ctx context.Context, rc *RunContext, req *llm.Request) error {
	for _, h := range c {
		if h.BeforeModel != nil {
			if err := h.BeforeModel(ctx, rc, req); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c HookChain) AfterModel(ctx context.Context, rc *RunContext, resp *llm.Response) error {
	for _, h := range c {
		if h.AfterModel != nil {
			if err := h.AfterModel(ctx, rc, resp); err != nil {
				return err
			}
		}
	}
	return nil
}

// BeforeTool returns the first non-HookContinue decision.
func (c HookChain) BeforeTool(ctx context.Context, rc *RunContext, call *llm.ToolCall) (HookDecision, error) {
	for _, h := range c {
		if h.BeforeTool != nil {
			d, err := h.BeforeTool(ctx, rc, call)
			if err != nil || d.Action != HookContinue {
				return d, err
			}
		}
	}
	return HookDecision{}, nil
}

func (c HookChain) AfterTool(ctx context.Context, rc *RunContext, call llm.ToolCall, res *ToolResult) error {
	for _, h := range c {
		if h.AfterTool != nil {
			if err := h.AfterTool(ctx, rc, call, res); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c HookChain) Event(ctx context.Context, ev Event) {
	for _, h := range c {
		if h.OnEvent != nil {
			h.OnEvent(ctx, ev)
		}
	}
}

func (c HookChain) RunEnd(ctx context.Context, rc *RunContext, err error) {
	for _, h := range c {
		if h.OnRunEnd != nil {
			h.OnRunEnd(ctx, rc, err)
		}
	}
}
