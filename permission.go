package react

import (
	"context"
	"path"

	"github.com/gosolu/react/llm"
)

// PolicyDecision is a policy verdict.
type PolicyDecision int

const (
	PolicyAllow PolicyDecision = iota
	PolicyDeny
	PolicyAsk
)

func (d PolicyDecision) String() string {
	switch d {
	case PolicyAllow:
		return "allow"
	case PolicyDeny:
		return "deny"
	case PolicyAsk:
		return "ask"
	}
	return "unknown"
}

// PermissionPolicy evaluates a tool call.
type PermissionPolicy interface {
	Check(ctx context.Context, spec ToolSpec, call llm.ToolCall) PolicyDecision
}

// PermissionPolicyFunc adapts a function to PermissionPolicy.
type PermissionPolicyFunc func(ctx context.Context, spec ToolSpec, call llm.ToolCall) PolicyDecision

func (f PermissionPolicyFunc) Check(ctx context.Context, spec ToolSpec, call llm.ToolCall) PolicyDecision {
	return f(ctx, spec, call)
}

// AllowAllPermissionPolicy permits every call.
func AllowAllPermissionPolicy() PermissionPolicy {
	return PermissionPolicyFunc(func(context.Context, ToolSpec, llm.ToolCall) PolicyDecision { return PolicyAllow })
}

// DefaultPermissionPolicy allows read-only and unannotated tools and asks for Destructive ones.
func DefaultPermissionPolicy() PermissionPolicy {
	return PermissionPolicyFunc(func(_ context.Context, s ToolSpec, _ llm.ToolCall) PolicyDecision {
		if s.Annotations.Destructive && !s.Annotations.ReadOnly {
			return PolicyAsk
		}
		return PolicyAllow
	})
}

// PermissionRules matches tool names against glob patterns (path.Match syntax).
// Precedence: PolicyDeny, then PolicyAsk, then PolicyAllow, then Fallback (DefaultPermissionPolicy() if nil).
type PermissionRules struct {
	Allow    []string
	Ask      []string
	Deny     []string
	Fallback PermissionPolicy
}

func (r PermissionRules) Check(ctx context.Context, s ToolSpec, c llm.ToolCall) PolicyDecision {
	switch {
	case match(r.Deny, s.Name):
		return PolicyDeny
	case match(r.Ask, s.Name):
		return PolicyAsk
	case match(r.Allow, s.Name):
		return PolicyAllow
	}
	if r.Fallback != nil {
		return r.Fallback.Check(ctx, s, c)
	}
	return DefaultPermissionPolicy().Check(ctx, s, c)
}

func match(patterns []string, name string) bool {
	for _, p := range patterns {
		if ok, _ := path.Match(p, name); ok {
			return true
		}
	}
	return false
}

// ApprovalRequest is passed to a PermissionApprover.
type ApprovalRequest struct {
	AgentName string
	Spec      ToolSpec
	Call      llm.ToolCall
}

// PermissionApprover asks a human (or system) to approve a call. Approve calls are
// serialized by the agent.
type PermissionApprover interface {
	Approve(ctx context.Context, req ApprovalRequest) (bool, error)
}

// PermissionApproverFunc adapts a function to PermissionApprover.
type PermissionApproverFunc func(ctx context.Context, req ApprovalRequest) (bool, error)

func (f PermissionApproverFunc) Approve(ctx context.Context, req ApprovalRequest) (bool, error) {
	return f(ctx, req)
}

// AlwaysApprove approves everything.
func AlwaysApprove() PermissionApprover {
	return PermissionApproverFunc(func(context.Context, ApprovalRequest) (bool, error) { return true, nil })
}

// AlwaysDeny denies everything.
func AlwaysDeny() PermissionApprover {
	return PermissionApproverFunc(func(context.Context, ApprovalRequest) (bool, error) { return false, nil })
}
