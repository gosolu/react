// Package permission decides whether tool calls may run, and how to ask a
// human for approval.
package permission

import (
	"context"
	"path"

	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/tool"
)

// Decision is a policy verdict.
type Decision int

const (
	Allow Decision = iota
	Deny
	Ask
)

func (d Decision) String() string {
	switch d {
	case Allow:
		return "allow"
	case Deny:
		return "deny"
	case Ask:
		return "ask"
	}
	return "unknown"
}

// Policy evaluates a tool call.
type Policy interface {
	Check(ctx context.Context, spec tool.Spec, call llm.ToolCall) Decision
}

// PolicyFunc adapts a function to Policy.
type PolicyFunc func(ctx context.Context, spec tool.Spec, call llm.ToolCall) Decision

func (f PolicyFunc) Check(ctx context.Context, spec tool.Spec, call llm.ToolCall) Decision {
	return f(ctx, spec, call)
}

// AllowAll permits every call.
func AllowAll() Policy {
	return PolicyFunc(func(context.Context, tool.Spec, llm.ToolCall) Decision { return Allow })
}

// Default allows read-only and unannotated tools and asks for Destructive ones.
func Default() Policy {
	return PolicyFunc(func(_ context.Context, s tool.Spec, _ llm.ToolCall) Decision {
		if s.Annotations.Destructive && !s.Annotations.ReadOnly {
			return Ask
		}
		return Allow
	})
}

// Rules matches tool names against glob patterns (path.Match syntax).
// Precedence: Deny, then Ask, then Allow, then Fallback (Default() if nil).
type Rules struct {
	Allow    []string
	Ask      []string
	Deny     []string
	Fallback Policy
}

func (r Rules) Check(ctx context.Context, s tool.Spec, c llm.ToolCall) Decision {
	switch {
	case match(r.Deny, s.Name):
		return Deny
	case match(r.Ask, s.Name):
		return Ask
	case match(r.Allow, s.Name):
		return Allow
	}
	if r.Fallback != nil {
		return r.Fallback.Check(ctx, s, c)
	}
	return Default().Check(ctx, s, c)
}

func match(patterns []string, name string) bool {
	for _, p := range patterns {
		if ok, _ := path.Match(p, name); ok {
			return true
		}
	}
	return false
}

// Request is passed to an Approver.
type Request struct {
	AgentName string
	Spec      tool.Spec
	Call      llm.ToolCall
}

// Approver asks a human (or system) to approve a call. Approve calls are
// serialized by the agent.
type Approver interface {
	Approve(ctx context.Context, req Request) (bool, error)
}

// ApproverFunc adapts a function to Approver.
type ApproverFunc func(ctx context.Context, req Request) (bool, error)

func (f ApproverFunc) Approve(ctx context.Context, req Request) (bool, error) { return f(ctx, req) }

// AlwaysApprove approves everything.
func AlwaysApprove() Approver {
	return ApproverFunc(func(context.Context, Request) (bool, error) { return true, nil })
}

// AlwaysDeny denies everything.
func AlwaysDeny() Approver {
	return ApproverFunc(func(context.Context, Request) (bool, error) { return false, nil })
}
