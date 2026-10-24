package react

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// BudgetLimit identifies one dimension of a run budget.
type BudgetLimit string

const (
	// BudgetModelCalls bounds provider calls across the run and its sub-agents.
	BudgetModelCalls BudgetLimit = "model_calls"
	// BudgetToolCalls bounds tool invocations across the run and its sub-agents.
	BudgetToolCalls BudgetLimit = "tool_calls"
	// BudgetRequestTokens bounds the estimated size of one model request.
	BudgetRequestTokens BudgetLimit = "request_tokens"
	// BudgetTotalTokens bounds input plus output tokens across the run tree.
	BudgetTotalTokens BudgetLimit = "total_tokens"
	// BudgetWallTime bounds the wall-clock time of the run tree.
	BudgetWallTime BudgetLimit = "wall_time"
)

// BudgetLimits bounds a run. Zero fields are unlimited.
//
// Sub-agents share the parent's counters and limits; a sub-agent's own
// WithBudget is applied as an additional frame, so both the parent and the
// child limits are enforced.
type BudgetLimits struct {
	MaxModelCalls    int           // total provider calls, including sub-agents
	MaxToolCalls     int           // total tool invocations, including sub-agents
	MaxRequestTokens int           // estimated input + max output for one request
	MaxTotalTokens   int           // input + output across the whole run tree
	MaxWallTime      time.Duration // wall-clock across the whole run tree
}

// BudgetUsage reports what a run consumed. For a sub-agent it reports only the
// sub-agent's own share; for the top-level run it reports the whole tree.
type BudgetUsage struct {
	ModelCalls   int
	ToolCalls    int
	InputTokens  int
	OutputTokens int
	Wall         time.Duration
	Exceeded     BudgetLimit // empty when no limit was hit
}

// ErrBudgetExceeded is returned when a BudgetLimits limit is hit. The partial
// Result is still returned.
var ErrBudgetExceeded = errors.New("agent: budget exceeded")

// ErrModelTimeout is returned when a per-call model timeout expires.
var ErrModelTimeout = errors.New("agent: model call timed out")

type budgetCounters struct {
	modelCalls int
	toolCalls  int
	usage      Usage
}

// budgetFrame is one run's view of the budget hierarchy. Frames form a chain so
// nested sub-agent limits and the parent limits are both enforced while keeping
// sibling sub-agent counters and exceeded states isolated.
type budgetFrame struct {
	limits    BudgetLimits
	counters  budgetCounters
	startedAt time.Time
	deadline  time.Time
	parent    *budgetFrame
	exceeded  BudgetLimit
}

type budgetState struct {
	mu sync.Mutex
}

type runBudget struct {
	state *budgetState
	frame *budgetFrame
}

type budgetContextKey struct{}

func withRunBudget(ctx context.Context, b *runBudget) context.Context {
	return context.WithValue(ctx, budgetContextKey{}, b)
}

func runBudgetFrom(ctx context.Context) *runBudget {
	b, _ := ctx.Value(budgetContextKey{}).(*runBudget)
	return b
}

// newRunBudget attaches a budget to ctx. It returns the derived context (with a
// deadline when a wall-time limit applies), the run's budget view, and a cancel
// function that is non-nil only when a deadline was installed.
func newRunBudget(ctx context.Context, limits BudgetLimits, now time.Time) (context.Context, *runBudget, context.CancelFunc) {
	parent := runBudgetFrom(ctx)
	var (
		state *budgetState
		frame *budgetFrame
	)
	if parent == nil {
		state = &budgetState{}
		frame = &budgetFrame{limits: limits, startedAt: now}
	} else {
		state = parent.state
		frame = &budgetFrame{
			limits:    limits,
			startedAt: now,
			parent:    parent.frame,
		}
	}
	if limits.MaxWallTime > 0 {
		frame.deadline = now.Add(limits.MaxWallTime)
	}
	if parent != nil {
		if parentDeadline := parent.frame.deadline; !parentDeadline.IsZero() &&
			(frame.deadline.IsZero() || parentDeadline.Before(frame.deadline)) {
			frame.deadline = parentDeadline
		}
	}

	var cancel context.CancelFunc
	if !frame.deadline.IsZero() {
		ctx, cancel = context.WithDeadline(ctx, frame.deadline)
	}
	b := &runBudget{state: state, frame: frame}
	return withRunBudget(ctx, b), b, cancel
}

func (f *budgetFrame) noteLocked(limit BudgetLimit) {
	if f.exceeded == "" {
		f.exceeded = limit
	}
}

// noteChainLocked records limit on leaf and all ancestors up to and including target.
func noteChainLocked(leaf, target *budgetFrame, limit BudgetLimit) {
	for f := leaf; f != nil; f = f.parent {
		f.noteLocked(limit)
		if f == target {
			break
		}
	}
}

// checkFrameLocked reports the first limit that prospective counters violate on f.
func checkFrameLocked(f *budgetFrame, next budgetCounters, now time.Time) BudgetLimit {
	l := f.limits
	if l.MaxModelCalls > 0 && next.modelCalls > l.MaxModelCalls {
		return BudgetModelCalls
	}
	if l.MaxToolCalls > 0 && next.toolCalls > l.MaxToolCalls {
		return BudgetToolCalls
	}
	if l.MaxTotalTokens > 0 {
		used := next.usage.InputTokens + next.usage.OutputTokens
		if used > l.MaxTotalTokens {
			return BudgetTotalTokens
		}
	}
	if !f.deadline.IsZero() && !now.Before(f.deadline) {
		return BudgetWallTime
	}
	return ""
}

// checkRequestFrameLocked reports whether a request of est tokens would violate
// f's request or total-token limit.
func checkRequestFrameLocked(f *budgetFrame, est int, now time.Time) BudgetLimit {
	l := f.limits
	if l.MaxRequestTokens > 0 && est > l.MaxRequestTokens {
		return BudgetRequestTokens
	}
	if l.MaxTotalTokens > 0 {
		used := f.counters.usage.InputTokens + f.counters.usage.OutputTokens
		if used+est > l.MaxTotalTokens {
			return BudgetTotalTokens
		}
	}
	if !f.deadline.IsZero() && !now.Before(f.deadline) {
		return BudgetWallTime
	}
	return ""
}

func (b *runBudget) exceeded() BudgetLimit {
	b.state.mu.Lock()
	defer b.state.mu.Unlock()
	for f := b.frame; f != nil; f = f.parent {
		if f.exceeded != "" {
			return f.exceeded
		}
	}
	return ""
}

// wallExpired reports whether any frame's deadline has passed, recording the
// wall-time limit on the expired frame(s) when it has.
func (b *runBudget) wallExpired() bool {
	now := time.Now()
	b.state.mu.Lock()
	defer b.state.mu.Unlock()
	expired := false
	for f := b.frame; f != nil; f = f.parent {
		if !f.deadline.IsZero() && !now.Before(f.deadline) {
			noteChainLocked(b.frame, f, BudgetWallTime)
			expired = true
		}
	}
	return expired
}

// precheck reports a limit that is already exceeded before work starts.
func (b *runBudget) precheck() BudgetLimit {
	if b.wallExpired() {
		return BudgetWallTime
	}
	return b.exceeded()
}

// reserveModelCall reserves one provider call. It returns the violated limit,
// or "" when the call may proceed.
func (b *runBudget) reserveModelCall() BudgetLimit {
	s := b.state
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	for f := b.frame; f != nil; f = f.parent {
		next := f.counters
		next.modelCalls++
		if limit := checkFrameLocked(f, next, now); limit != "" {
			noteChainLocked(b.frame, f, limit)
			return limit
		}
	}
	for f := b.frame; f != nil; f = f.parent {
		f.counters.modelCalls++
	}
	return ""
}

// reserveToolCalls reserves n tool invocations.
func (b *runBudget) reserveToolCalls(n int) BudgetLimit {
	if n <= 0 {
		return ""
	}
	s := b.state
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	for f := b.frame; f != nil; f = f.parent {
		next := f.counters
		next.toolCalls += n
		if limit := checkFrameLocked(f, next, now); limit != "" {
			noteChainLocked(b.frame, f, limit)
			return limit
		}
	}
	for f := b.frame; f != nil; f = f.parent {
		f.counters.toolCalls += n
	}
	return ""
}

// checkRequestTokens checks an estimated request size without reserving
// anything.
func (b *runBudget) checkRequestTokens(est int) BudgetLimit {
	s := b.state
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	for f := b.frame; f != nil; f = f.parent {
		if limit := checkRequestFrameLocked(f, est, now); limit != "" {
			noteChainLocked(b.frame, f, limit)
			return limit
		}
	}
	return ""
}

// addUsage records tokens spent by a completed model call.
func (b *runBudget) addUsage(u Usage) {
	s := b.state
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	for f := b.frame; f != nil; f = f.parent {
		f.counters.usage.Add(u)
		if limit := checkFrameLocked(f, f.counters, now); limit != "" {
			noteChainLocked(b.frame, f, limit)
		}
	}
}

// usage returns this run's share of token usage.
func (b *runBudget) usage() Usage {
	b.state.mu.Lock()
	defer b.state.mu.Unlock()
	return b.frame.counters.usage
}

// budgetUsage returns this run's share of calls, tokens, and wall time.
func (b *runBudget) budgetUsage() BudgetUsage {
	b.state.mu.Lock()
	defer b.state.mu.Unlock()
	c := b.frame.counters
	exceeded := b.frame.exceeded
	if exceeded == "" {
		for f := b.frame.parent; f != nil; f = f.parent {
			if f.exceeded != "" {
				exceeded = f.exceeded
				break
			}
		}
	}
	return BudgetUsage{
		ModelCalls:   c.modelCalls,
		ToolCalls:    c.toolCalls,
		InputTokens:  c.usage.InputTokens,
		OutputTokens: c.usage.OutputTokens,
		Wall:         time.Since(b.frame.startedAt),
		Exceeded:     exceeded,
	}
}

func budgetError(limit BudgetLimit) error {
	if limit == "" {
		limit = "unspecified"
	}
	return fmt.Errorf("%w: %s", ErrBudgetExceeded, limit)
}

// estimateRequestTokens is the same cheap heuristic used by the context
// manager, extended to the system prompt, tool schemas, and the per-response
// output cap.
func estimateRequestTokens(req *Request) int {
	n := EstimateText(req.System)
	for _, m := range req.Messages {
		n += EstimateTokens(m)
	}
	for _, t := range req.Tools {
		n += EstimateText(t.Name) + EstimateText(t.Description) + EstimateText(string(t.InputSchema))
	}
	if req.MaxTokens > 0 {
		n += req.MaxTokens
	}
	return n
}
