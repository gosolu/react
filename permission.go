package react

import (
	"context"
	"encoding/json"
	"path"
	"strings"
	"sync"
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
	Check(ctx context.Context, spec ToolSpec, call ToolCall) PolicyDecision
}

// PermissionPolicyFunc adapts a function to PermissionPolicy.
type PermissionPolicyFunc func(ctx context.Context, spec ToolSpec, call ToolCall) PolicyDecision

func (f PermissionPolicyFunc) Check(ctx context.Context, spec ToolSpec, call ToolCall) PolicyDecision {
	return f(ctx, spec, call)
}

// AllowAllPermissionPolicy permits every call.
func AllowAllPermissionPolicy() PermissionPolicy {
	return PermissionPolicyFunc(func(context.Context, ToolSpec, ToolCall) PolicyDecision { return PolicyAllow })
}

// DefaultPermissionPolicy allows read-only and unannotated tools, pre-approves tools allowed by active skills,
// and asks for Destructive ones.
func DefaultPermissionPolicy() PermissionPolicy {
	return PermissionPolicyFunc(func(ctx context.Context, s ToolSpec, c ToolCall) PolicyDecision {
		if matchRules(ActiveSkillAllowedTools(ctx), s, c) {
			return PolicyAllow
		}
		if s.Annotations.Destructive && !s.Annotations.ReadOnly {
			return PolicyAsk
		}
		return PolicyAllow
	})
}

// PermissionRules matches tool names against glob patterns (path.Match syntax).
// Precedence: PolicyDeny, then PolicyAsk, then PolicyAllow, then active skill allowed-tools,
// then Fallback (DefaultPermissionPolicy() if nil).
type PermissionRules struct {
	Allow    []string
	Ask      []string
	Deny     []string
	Fallback PermissionPolicy
}

func (r PermissionRules) Check(ctx context.Context, s ToolSpec, c ToolCall) PolicyDecision {
	switch {
	case matchRules(r.Deny, s, c):
		return PolicyDeny
	case matchRules(r.Ask, s, c):
		return PolicyAsk
	case matchRules(r.Allow, s, c):
		return PolicyAllow
	case matchRules(ActiveSkillAllowedTools(ctx), s, c):
		return PolicyAllow
	}
	if r.Fallback != nil {
		return r.Fallback.Check(ctx, s, c)
	}
	return DefaultPermissionPolicy().Check(ctx, s, c)
}

func matchRules(patterns []string, s ToolSpec, c ToolCall) bool {
	for _, p := range patterns {
		if strings.Contains(p, "(") && strings.HasSuffix(p, ")") {
			idx := strings.Index(p, "(")
			toolPattern := p[:idx]
			argPattern := strings.TrimSuffix(p[idx+1:], ")")
			if match([]string{toolPattern}, s.Name) {
				argStr := string(c.Arguments)
				if ok, _ := path.Match(argPattern, argStr); ok {
					return true
				}
				trimmed := strings.Trim(argPattern, "*")
				if trimmed != "" && strings.Contains(argStr, trimmed) {
					return true
				}
				// Also handle command prefix patterns like Bash(git:*) or Bash(jq:*)
				colonPrefix := strings.TrimSuffix(trimmed, ":")
				if colonPrefix != "" {
					var argMap map[string]any
					if json.Unmarshal(c.Arguments, &argMap) == nil {
						for _, field := range []string{"command", "cmd", "script", "input", "query"} {
							if val, ok := argMap[field].(string); ok {
								val = strings.TrimSpace(val)
								if val == colonPrefix || strings.HasPrefix(val, colonPrefix+" ") {
									return true
								}
							}
						}
					}
					if strings.Contains(argStr, `"`+colonPrefix+` `) || strings.Contains(argStr, `"`+colonPrefix+`"`) {
						return true
					}
				}
			}
			continue
		}
		if match([]string{p}, s.Name) {
			return true
		}
	}
	return false
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
	Call      ToolCall
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

// SessionRememberApprover wraps an approver and caches approvals for the duration of a session.
type SessionRememberApprover struct {
	inner    PermissionApprover
	mu       sync.RWMutex
	approved map[string]bool
}

// NewSessionRememberApprover returns an approver that remembers approved tools.
func NewSessionRememberApprover(inner PermissionApprover) *SessionRememberApprover {
	return &SessionRememberApprover{
		inner:    inner,
		approved: make(map[string]bool),
	}
}

// Approve checks if the tool was previously remembered; if not, delegates to inner approver.
func (a *SessionRememberApprover) Approve(ctx context.Context, req ApprovalRequest) (bool, error) {
	rc := RunContextFrom(ctx)
	name := req.Call.Name
	if name == "" {
		name = req.Spec.Name
	}
	key := name
	if rc != nil && rc.SessionID != "" {
		key = rc.SessionID + ":" + name
	}

	a.mu.RLock()
	if a.approved[key] || a.approved[name] {
		a.mu.RUnlock()
		return true, nil
	}
	a.mu.RUnlock()

	ok, err := a.inner.Approve(ctx, req)
	if err != nil || !ok {
		return false, err
	}
	return true, nil
}

// Remember caches future approvals for toolName across all sessions.
func (a *SessionRememberApprover) Remember(toolName string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.approved[toolName] = true
}

// RememberForSession caches future approvals for toolName strictly within sessionID.
func (a *SessionRememberApprover) RememberForSession(sessionID, toolName string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if sessionID != "" {
		a.approved[sessionID+":"+toolName] = true
	} else {
		a.approved[toolName] = true
	}
}
