package react

import (
	"context"
	"encoding/json"
	"path"
	"strings"
	"sync"

)

// RunContext describes the current agent run. Fields are set by the agent;
// hooks should treat them as read-only and use Set/Get for custom state.
type RunContext struct {
	AgentName string
	SessionID string
	RunID     string
	Turn      int
	Depth     int // 0 for the top-level agent, +1 per sub-agent level

	mu      sync.Mutex
	values  map[string]any
	context []string // pending additional context for the model (Claude-style system reminders)
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

// AddContext queues additional context (e.g. system instructions, reminders, environment state)
// that will be injected into the next model request.
func (rc *RunContext) AddContext(text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	rc.mu.Lock()
	defer rc.mu.Unlock()
	rc.context = append(rc.context, text)
}

// ConsumeContext extracts and clears all pending additional context strings.
func (rc *RunContext) ConsumeContext() []string {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	if len(rc.context) == 0 {
		return nil
	}
	c := rc.context
	rc.context = nil
	return c
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

// HookAction represents the action a hook decision requests.
type HookAction int

const (
	// HookContinue lets the action proceed normally.
	HookContinue HookAction = iota
	// HookDeny blocks the action; Reason is reported to the caller or model as an error.
	HookDeny
	// HookOverride skips execution and uses Result instead (for BeforeTool).
	HookOverride
	// HookContinueWorking prevents the agent from stopping and requests continuation (for BeforeStop).
	HookContinueWorking
)

// HookDecision is returned by BeforeTool and BeforeStop.
type HookDecision struct {
	Action            HookAction
	Reason            string
	Result            ToolResult      // used with HookOverride in BeforeTool
	UpdatedArguments  json.RawMessage // optional updated arguments for BeforeTool
	AdditionalContext string          // context injected into conversation for model
}

// HookPromptDecision is returned by BeforePrompt.
type HookPromptDecision struct {
	Action            HookAction
	Reason            string
	UpdatedPrompt     string
	AdditionalContext string
}

// HookApprovalDecision is returned by OnPermissionRequest.
type HookApprovalDecision struct {
	Handled          bool
	Approved         bool
	Reason           string
	UpdatedArguments json.RawMessage
}

// Hooks is a comprehensive set of optional lifecycle callbacks matching the Claude Code
// hook lifecycle model. Any field may be nil.
//
// Lifecycle points:
//   - Session/Run: OnRunStart, OnRunEnd
//   - Prompt & Stop: BeforePrompt, BeforeStop, OnStopFailure
//   - Model: BeforeModel, AfterModel
//   - Tool Execution: BeforeTool, AfterTool, OnToolFailure, AfterToolBatch
//   - Permissions: OnPermissionRequest, OnPermissionDenied
//   - Subagents: SubagentStart, SubagentStop
//   - Compaction: PreCompact, PostCompact
//   - Observability: OnEvent
type Hooks struct {
	// Session / Run lifecycle
	OnRunStart func(ctx context.Context, rc *RunContext) error
	OnRunEnd   func(ctx context.Context, rc *RunContext, err error)

	// User prompt & stop lifecycle
	BeforePrompt  func(ctx context.Context, rc *RunContext, prompt string) (HookPromptDecision, error)
	BeforeStop    func(ctx context.Context, rc *RunContext, output string) (HookDecision, error)
	OnStopFailure func(ctx context.Context, rc *RunContext, err error)

	// Model lifecycle
	BeforeModel func(ctx context.Context, rc *RunContext, req *Request) error
	AfterModel  func(ctx context.Context, rc *RunContext, resp *Response) error

	// Tool lifecycle
	BeforeTool     func(ctx context.Context, rc *RunContext, call *ToolCall) (HookDecision, error)
	AfterTool      func(ctx context.Context, rc *RunContext, call ToolCall, res *ToolResult) error
	OnToolFailure  func(ctx context.Context, rc *RunContext, call ToolCall, err error) error
	AfterToolBatch func(ctx context.Context, rc *RunContext, results []ToolCallResult) error

	// Permission lifecycle
	OnPermissionRequest func(ctx context.Context, rc *RunContext, req ApprovalRequest) (HookApprovalDecision, error)
	OnPermissionDenied  func(ctx context.Context, rc *RunContext, call ToolCall, reason string)

	// Subagent lifecycle
	SubagentStart func(ctx context.Context, rc *RunContext, agentName string, task string)
	SubagentStop  func(ctx context.Context, rc *RunContext, agentName string, task string, output string, err error)

	// Context compaction lifecycle
	PreCompact  func(ctx context.Context, rc *RunContext, messageCount int)
	PostCompact func(ctx context.Context, rc *RunContext, beforeCount, afterCount int)

	// General event stream
	OnEvent func(ctx context.Context, ev Event)
}

// HookChain runs multiple Hooks in order.
type HookChain []Hooks

// RunStart runs all OnRunStart hooks in order.
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

// RunEnd runs all OnRunEnd hooks in order.
func (c HookChain) RunEnd(ctx context.Context, rc *RunContext, err error) {
	for _, h := range c {
		if h.OnRunEnd != nil {
			h.OnRunEnd(ctx, rc, err)
		}
	}
}

// BeforePrompt runs all BeforePrompt hooks in order. If any hook denies the prompt,
// execution stops immediately.
func (c HookChain) BeforePrompt(ctx context.Context, rc *RunContext, prompt string) (HookPromptDecision, error) {
	currentPrompt := prompt
	for _, h := range c {
		if h.BeforePrompt != nil {
			d, err := h.BeforePrompt(ctx, rc, currentPrompt)
			if err != nil {
				return HookPromptDecision{}, err
			}
			if d.Action == HookDeny {
				return d, nil
			}
			if d.UpdatedPrompt != "" {
				currentPrompt = d.UpdatedPrompt
			}
			if d.AdditionalContext != "" && rc != nil {
				rc.AddContext(d.AdditionalContext)
			}
		}
	}
	return HookPromptDecision{Action: HookContinue, UpdatedPrompt: currentPrompt}, nil
}

// BeforeStop runs all BeforeStop hooks in order. Returns the first decision that blocks
// stopping (HookDeny or HookContinueWorking).
func (c HookChain) BeforeStop(ctx context.Context, rc *RunContext, output string) (HookDecision, error) {
	for _, h := range c {
		if h.BeforeStop != nil {
			d, err := h.BeforeStop(ctx, rc, output)
			if err != nil {
				return HookDecision{}, err
			}
			if d.AdditionalContext != "" && rc != nil {
				rc.AddContext(d.AdditionalContext)
			}
			if d.Action == HookDeny || d.Action == HookContinueWorking {
				return d, nil
			}
		}
	}
	return HookDecision{Action: HookContinue}, nil
}

// StopFailure runs all OnStopFailure hooks in order.
func (c HookChain) StopFailure(ctx context.Context, rc *RunContext, err error) {
	for _, h := range c {
		if h.OnStopFailure != nil {
			h.OnStopFailure(ctx, rc, err)
		}
	}
}

// BeforeModel runs all BeforeModel hooks in order.
func (c HookChain) BeforeModel(ctx context.Context, rc *RunContext, req *Request) error {
	for _, h := range c {
		if h.BeforeModel != nil {
			if err := h.BeforeModel(ctx, rc, req); err != nil {
				return err
			}
		}
	}
	return nil
}

// AfterModel runs all AfterModel hooks in order.
func (c HookChain) AfterModel(ctx context.Context, rc *RunContext, resp *Response) error {
	for _, h := range c {
		if h.AfterModel != nil {
			if err := h.AfterModel(ctx, rc, resp); err != nil {
				return err
			}
		}
	}
	return nil
}

// BeforeTool runs all BeforeTool hooks in order. Intermediate hooks may update
// arguments or queue additional context. Returns the first non-HookContinue decision.
func (c HookChain) BeforeTool(ctx context.Context, rc *RunContext, call *ToolCall) (HookDecision, error) {
	for _, h := range c {
		if h.BeforeTool != nil {
			d, err := h.BeforeTool(ctx, rc, call)
			if err != nil {
				return HookDecision{}, err
			}
			if len(d.UpdatedArguments) > 0 {
				call.Arguments = d.UpdatedArguments
			}
			if d.AdditionalContext != "" && rc != nil {
				rc.AddContext(d.AdditionalContext)
			}
			if d.Action != HookContinue {
				return d, nil
			}
		}
	}
	return HookDecision{Action: HookContinue}, nil
}

// AfterTool runs all AfterTool hooks in order.
func (c HookChain) AfterTool(ctx context.Context, rc *RunContext, call ToolCall, res *ToolResult) error {
	for _, h := range c {
		if h.AfterTool != nil {
			if err := h.AfterTool(ctx, rc, call, res); err != nil {
				return err
			}
		}
	}
	return nil
}

// ToolFailure runs all OnToolFailure hooks in order.
func (c HookChain) ToolFailure(ctx context.Context, rc *RunContext, call ToolCall, err error) error {
	for _, h := range c {
		if h.OnToolFailure != nil {
			if ferr := h.OnToolFailure(ctx, rc, call, err); ferr != nil {
				return ferr
			}
		}
	}
	return nil
}

// AfterToolBatch runs all AfterToolBatch hooks in order.
func (c HookChain) AfterToolBatch(ctx context.Context, rc *RunContext, results []ToolCallResult) error {
	for _, h := range c {
		if h.AfterToolBatch != nil {
			if err := h.AfterToolBatch(ctx, rc, results); err != nil {
				return err
			}
		}
	}
	return nil
}

// PermissionRequest runs all OnPermissionRequest hooks. Returns the first handled decision.
func (c HookChain) PermissionRequest(ctx context.Context, rc *RunContext, req ApprovalRequest) (HookApprovalDecision, error) {
	for _, h := range c {
		if h.OnPermissionRequest != nil {
			dec, err := h.OnPermissionRequest(ctx, rc, req)
			if err != nil {
				return HookApprovalDecision{}, err
			}
			if dec.Handled {
				return dec, nil
			}
		}
	}
	return HookApprovalDecision{}, nil
}

// PermissionDenied runs all OnPermissionDenied hooks in order.
func (c HookChain) PermissionDenied(ctx context.Context, rc *RunContext, call ToolCall, reason string) {
	for _, h := range c {
		if h.OnPermissionDenied != nil {
			h.OnPermissionDenied(ctx, rc, call, reason)
		}
	}
}

// SubagentStart runs all SubagentStart hooks in order.
func (c HookChain) SubagentStart(ctx context.Context, rc *RunContext, agentName, task string) {
	for _, h := range c {
		if h.SubagentStart != nil {
			h.SubagentStart(ctx, rc, agentName, task)
		}
	}
}

// SubagentStop runs all SubagentStop hooks in order.
func (c HookChain) SubagentStop(ctx context.Context, rc *RunContext, agentName, task, output string, err error) {
	for _, h := range c {
		if h.SubagentStop != nil {
			h.SubagentStop(ctx, rc, agentName, task, output, err)
		}
	}
}

// PreCompact runs all PreCompact hooks in order.
func (c HookChain) PreCompact(ctx context.Context, rc *RunContext, messageCount int) {
	for _, h := range c {
		if h.PreCompact != nil {
			h.PreCompact(ctx, rc, messageCount)
		}
	}
}

// PostCompact runs all PostCompact hooks in order.
func (c HookChain) PostCompact(ctx context.Context, rc *RunContext, beforeCount, afterCount int) {
	for _, h := range c {
		if h.PostCompact != nil {
			h.PostCompact(ctx, rc, beforeCount, afterCount)
		}
	}
}

// Event delivers an event to all OnEvent handlers in order.
func (c HookChain) Event(ctx context.Context, ev Event) {
	for _, h := range c {
		if h.OnEvent != nil {
			h.OnEvent(ctx, ev)
		}
	}
}

// MatchTool creates a BeforeTool hook filter that only executes fn if the tool name
// matches the given pattern (exact name or glob pattern like "mcp__*").
func MatchTool(pattern string, fn func(ctx context.Context, rc *RunContext, call *ToolCall) (HookDecision, error)) func(ctx context.Context, rc *RunContext, call *ToolCall) (HookDecision, error) {
	return func(ctx context.Context, rc *RunContext, call *ToolCall) (HookDecision, error) {
		if matchToolPattern(pattern, call.Name) {
			return fn(ctx, rc, call)
		}
		return HookDecision{Action: HookContinue}, nil
	}
}

// MatchToolAfter creates an AfterTool hook filter that only executes fn if the tool name
// matches the given pattern.
func MatchToolAfter(pattern string, fn func(ctx context.Context, rc *RunContext, call ToolCall, res *ToolResult) error) func(ctx context.Context, rc *RunContext, call ToolCall, res *ToolResult) error {
	return func(ctx context.Context, rc *RunContext, call ToolCall, res *ToolResult) error {
		if matchToolPattern(pattern, call.Name) {
			return fn(ctx, rc, call, res)
		}
		return nil
	}
}

func matchToolPattern(pattern, name string) bool {
	if pattern == "" || pattern == "*" || pattern == name {
		return true
	}
	matched, err := path.Match(pattern, name)
	return err == nil && matched
}
