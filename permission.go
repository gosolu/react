package react

import (
	"context"
	"encoding/json"
	"fmt"
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

// DefaultPermissionPolicy allows read-only and unannotated tools, pre-approves
// non-destructive tools allowed by active skills, and asks for Destructive ones.
func DefaultPermissionPolicy() PermissionPolicy {
	return PermissionPolicyFunc(func(ctx context.Context, s ToolSpec, c ToolCall) PolicyDecision {
		if s.Annotations.Destructive {
			return PolicyAsk
		}
		if matchRules(ActiveSkillAllowedTools(ctx), s, c) {
			return PolicyAllow
		}
		return PolicyAllow
	})
}

// StrictPermissionPolicy allows ReadOnly tools, pre-approves non-destructive tools
// allowed by active skills, and asks for all unannotated or Destructive tools.
func StrictPermissionPolicy() PermissionPolicy {
	return PermissionPolicyFunc(func(ctx context.Context, s ToolSpec, c ToolCall) PolicyDecision {
		if s.Annotations.Destructive {
			return PolicyAsk
		}
		if s.Annotations.ReadOnly {
			return PolicyAllow
		}
		if matchRules(ActiveSkillAllowedTools(ctx), s, c) {
			return PolicyAllow
		}
		return PolicyAsk
	})
}

// PermissionRules matches tool names against glob patterns (path.Match syntax).
// Precedence: PolicyDeny, then PolicyAsk, then PolicyAllow, then active skill allowed-tools
// (for non-destructive tools), then Fallback (DefaultPermissionPolicy() if nil).
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
	case !s.Annotations.Destructive && matchRules(ActiveSkillAllowedTools(ctx), s, c):
		return PolicyAllow
	}
	if r.Fallback != nil {
		return r.Fallback.Check(ctx, s, c)
	}
	return DefaultPermissionPolicy().Check(ctx, s, c)
}

func matchRules(patterns []string, s ToolSpec, c ToolCall) bool {
	for _, p := range patterns {
		if matchRule(p, s, c) {
			return true
		}
	}
	return false
}

// shellBlockedChars defines characters prohibited in shell commands to prevent
// command chaining, redirection, parameter/command expansion, quoting bypasses,
// subshell spawning, globbing, and newline-separated command execution.
const shellBlockedChars = ";|&><$`\n\r\"'\\(){}*?"

func isShellTool(name string) bool {
	switch strings.ToLower(name) {
	case "bash", "sh", "zsh", "shell", "terminal", "exec":
		return true
	default:
		return false
	}
}

func getCommandArg(argMap map[string]any) (string, bool) {
	for _, field := range []string{"command", "cmd", "script"} {
		if val, ok := argMap[field].(string); ok {
			return val, true
		}
	}
	return "", false
}

func matchRule(p string, s ToolSpec, c ToolCall) bool {
	toolPattern, rest, ok := strings.Cut(p, "(")
	if !ok || !strings.HasSuffix(rest, ")") {
		return match([]string{p}, s.Name)
	}

	if !match([]string{toolPattern}, s.Name) {
		return false
	}

	argPattern := strings.TrimSuffix(rest, ")")
	if argPattern == "" {
		return true
	}

	if len(c.Arguments) == 0 {
		return false
	}

	var argMap map[string]any
	if err := json.Unmarshal(c.Arguments, &argMap); err != nil {
		return false
	}

	cmdVal, hasCmd := getCommandArg(argMap)
	if isShellTool(s.Name) || hasCmd {
		if !hasCmd {
			return false
		}
		return matchShellCommand(argPattern, cmdVal)
	}

	return matchStructuredArgs(argPattern, argMap)
}

func matchShellCommand(argPattern, cmdVal string) bool {
	cmdVal = strings.TrimSpace(cmdVal)
	if cmdVal == "" {
		return false
	}

	// Reject all redirection, expansion, chaining, and quoting metacharacters.
	if strings.ContainsAny(cmdVal, shellBlockedChars) {
		return false
	}

	cmdTokens := strings.Fields(cmdVal)
	if len(cmdTokens) == 0 {
		return false
	}

	trimmed := strings.Trim(argPattern, "*")
	if strings.HasSuffix(trimmed, ":") {
		// Colon prefix form: e.g. "git:*" or "git:"
		colonPrefix := strings.TrimSuffix(trimmed, ":")
		prefixTokens := strings.Fields(colonPrefix)
		if len(prefixTokens) == 0 || len(cmdTokens) < len(prefixTokens) {
			return false
		}
		for i := range prefixTokens {
			if cmdTokens[i] != prefixTokens[i] {
				return false
			}
		}
		return true
	}

	// For shell tools without the colon prefix form, prefix matching (e.g. "echo *", "git commit*")
	// is inherently unsound and must default to PolicyAsk.
	// Only an exact, single command with no wildcards or metacharacters is permitted.
	if strings.ContainsAny(argPattern, "*?") || strings.ContainsAny(argPattern, shellBlockedChars) {
		return false
	}

	patternTokens := strings.Fields(argPattern)
	if len(patternTokens) == 0 || len(patternTokens) != len(cmdTokens) {
		return false
	}
	for i := range patternTokens {
		if patternTokens[i] != cmdTokens[i] {
			return false
		}
	}
	return true
}

func matchStructuredArgs(argPattern string, argMap map[string]any) bool {
	if field, valPat, ok := strings.Cut(argPattern, ":"); ok {
		field = strings.TrimSpace(field)
		valPat = strings.TrimSpace(valPat)
		if val, exists := argMap[field]; exists {
			if valPat == "*" || valPat == "" {
				return true
			}
			strVal := fmt.Sprint(val)
			if matchGlob(valPat, strVal) {
				return true
			}
		}
		return false
	}

	for _, v := range argMap {
		if strVal, ok := v.(string); ok {
			if matchGlob(argPattern, strVal) {
				return true
			}
		}
	}
	return false
}

func match(patterns []string, name string) bool {
	for _, p := range patterns {
		if matchGlob(p, name) {
			return true
		}
	}
	return false
}

func matchGlob(pattern, value string) bool {
	if pattern == "*" || pattern == "**" || pattern == value {
		return true
	}
	if ok, err := path.Match(pattern, value); err == nil && ok {
		return true
	}
	if strings.HasSuffix(pattern, "/**") {
		prefix := strings.TrimSuffix(pattern, "/**")
		if value == prefix || strings.HasPrefix(value, prefix+"/") {
			return true
		}
	}
	if strings.HasSuffix(pattern, "/*") {
		prefix := strings.TrimSuffix(pattern, "/*")
		if strings.HasPrefix(value, prefix+"/") {
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

// PermissionApprover asks a human (or system) to approve a call.
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
