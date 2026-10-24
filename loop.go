package react

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gosolu/react/pkg/utf8util"
)

type emitFunc func(Event) bool

// execute is the single ReAct loop shared by Run and Stream.
func (a *Agent) execute(ctx context.Context, sessionID, input string, rcfg runConfig, stream bool, sink emitFunc) (result *Result, err error) {
	if a.closed.Load() {
		return nil, ErrAgentClosed
	}
	depth := depthFrom(ctx)
	if depth > a.cfg.maxDepth {
		return nil, ErrMaxDepth
	}
	if strings.TrimSpace(input) == "" && len(rcfg.parts) == 0 {
		return nil, errors.New("agent: empty input")
	}

	if sessionID != "" {
		unlock, lerr := a.lockSession(ctx, sessionID)
		if lerr != nil {
			return nil, lerr
		}
		defer unlock()
	}

	runStart := time.Now()
	ctx, rb, budgetCancel := newRunBudget(ctx, a.cfg.budget, runStart)
	if budgetCancel != nil {
		defer budgetCancel()
	}

	rc := &RunContext{AgentName: a.cfg.name, SessionID: sessionID, RunID: newID(), Depth: depth}
	var emitMu sync.Mutex
	emit := func(ev Event) bool {
		if ev.Agent == "" { // events forwarded from sub-agents keep their identity
			ev.Agent, ev.Depth, ev.RunID = rc.AgentName, rc.Depth, rc.RunID
			if ev.Turn == 0 {
				ev.Turn = rc.Turn
			}
		}
		emitMu.Lock()
		defer emitMu.Unlock()
		a.hooks.Event(ctx, ev)
		if rcfg.onEvent != nil {
			rcfg.onEvent(ev)
		}
		return sink(ev)
	}

	ctx = WithRunContext(ctx, rc)
	ctx = withHooks(ctx, a.hooks)
	ctx = context.WithValue(ctx, depthKey{}, depth)
	ctx = context.WithValue(ctx, emitterKey{}, emitFunc(emit))

	finalized := false
	finalize := func() {
		if result == nil || finalized {
			return
		}
		finalized = true
		result.RunID = rc.RunID
		result.Usage = rb.usage()
		result.Budget = rb.budgetUsage()
		result.Duration = time.Since(runStart)
		result.StopReason = terminalStopReason(err, result.StopReason)
	}

	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("agent: runtime panic recovered: %v\n%s", r, debug.Stack())
			if result == nil {
				result = &Result{RunID: rc.RunID}
			}
			result.StopReason = StopError
		}
		finalize()
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("agent: hook panic recovered: %v\n%s", r, debug.Stack())
				if result == nil {
					result = &Result{RunID: rc.RunID}
				}
				result.StopReason = StopError
			}
		}()
		if err != nil {
			a.hooks.StopFailure(ctx, rc, err)
		}
		a.hooks.RunEnd(ctx, rc, err)
	}()

	if err = a.hooks.RunStart(ctx, rc); err != nil {
		return nil, err
	}

	pdec, perr := a.hooks.BeforePrompt(ctx, rc, input)
	if perr != nil {
		err = perr
		return nil, err
	}
	if pdec.Action == HookDeny {
		reason := pdec.Reason
		if reason == "" {
			reason = "prompt blocked by hook"
		}
		err = fmt.Errorf("prompt blocked: %s", reason)
		return nil, err
	}
	if pdec.UpdatedPrompt != "" {
		input = pdec.UpdatedPrompt
	}

	// Load history.
	var history []Message
	if sessionID != "" {
		if history, err = a.cfg.store.Load(ctx, sessionID); err != nil {
			return nil, fmt.Errorf("agent: load session: %w", err)
		}
	} else {
		history = CloneMessages(rcfg.history)
	}
	user := Message{Role: RoleUser}
	if input != "" {
		user.Content = append(user.Content, TextPart(sanitizeSystemReminderTags(input)))
	}
	for _, p := range rcfg.parts {
		if p.Type == PartText {
			p.Text = sanitizeSystemReminderTags(p.Text)
		}
		user.Content = append(user.Content, p)
	}
	history = append(history, user)

	result = &Result{RunID: rc.RunID, Messages: []Message{user}}
	pending := []Message{user}
	// flush persists completed turns. It uses a non-cancellable context so a
	// finished turn is saved even if the run is cancelled right after.
	flush := func() error {
		defer func() { pending = nil }()
		if sessionID == "" || len(pending) == 0 {
			return nil
		}
		if err := a.cfg.store.Append(context.WithoutCancel(ctx), sessionID, pending...); err != nil {
			return fmt.Errorf("agent: save session: %w", err)
		}
		return nil
	}

	model := a.cfg.model
	if rcfg.model != "" {
		model = rcfg.model
	}
	activeReg := a.registry
	if len(rcfg.tools) > 0 {
		activeReg = NewToolRegistry()
		for _, t := range a.registry.List() {
			_ = activeReg.Register(t)
		}
		for _, t := range rcfg.tools {
			_ = activeReg.Register(t)
		}
	}
	specs := activeReg.Specs()

	emit(Event{Type: EventRunStart})

	for turn := 1; ; turn++ {
		if turn > a.cfg.maxTurns {
			result.StopReason = StopMaxTurns
			err = ErrMaxTurns
			break
		}
		rc.Turn = turn
		result.Turns = turn
		emit(Event{Type: EventTurnStart})

		if limit := rb.precheck(); limit != "" {
			result.StopReason = StopBudgetExceeded
			err = budgetError(limit)
			emit(Event{Type: EventTurnEnd})
			break
		}

		a.hooks.PreCompact(ctx, rc, len(history))
		msgs, ferr := a.cfg.ctxManager.Fit(ctx, a.system, history, a.cfg.contextBudget)
		if ferr != nil {
			if errors.Is(ferr, ErrBudgetExceeded) {
				result.StopReason = StopBudgetExceeded
			}
			return result, fmt.Errorf("agent: context manager: %w", ferr)
		}
		a.hooks.PostCompact(ctx, rc, len(history), len(msgs))

		if a.cfg.persistCompaction && sessionID != "" && len(msgs) < len(history) {
			history = slices.Clone(msgs)
			if err := a.cfg.store.Replace(context.WithoutCancel(ctx), sessionID, history); err != nil {
				return result, fmt.Errorf("agent: persist compaction: %w", err)
			}
			pending = nil
		}

		if reminders := rc.ConsumeContext(); len(reminders) > 0 {
			var b strings.Builder
			b.WriteString("<system-reminder>\n")
			for _, r := range reminders {
				b.WriteString(r)
				b.WriteString("\n")
			}
			b.WriteString("</system-reminder>")
			msgs = append(msgs, UserText(b.String()))
		}
		req := &Request{
			Model:       model,
			System:      a.system,
			Messages:    msgs,
			Tools:       specs,
			MaxTokens:   a.cfg.maxTokens,
			Temperature: a.cfg.temperature,
		}
		// A forced tool choice applies to the first turn only, otherwise the
		// model could never produce a final answer.
		if rcfg.toolChoice != "" && (turn == 1 || rcfg.toolChoice == ToolChoiceNone || rcfg.toolChoice == ToolChoiceAuto) {
			req.ToolChoice = rcfg.toolChoice
		}
		if err := a.hooks.BeforeModel(ctx, rc, req); err != nil {
			return result, err
		}
		if limit := rb.checkRequestTokens(estimateRequestTokens(req)); limit != "" {
			result.StopReason = StopBudgetExceeded
			err = budgetError(limit)
			emit(Event{Type: EventTurnEnd})
			break
		}
		if limit := rb.reserveModelCall(); limit != "" {
			result.StopReason = StopBudgetExceeded
			err = budgetError(limit)
			emit(Event{Type: EventTurnEnd})
			break
		}

		modelStart := time.Now()
		callCtx := ctx
		var modelCancel context.CancelFunc
		if a.cfg.modelTimeout > 0 {
			callCtx, modelCancel = context.WithTimeout(ctx, a.cfg.modelTimeout)
		}
		resp, merr := a.callModel(callCtx, req, stream, emit)
		if modelCancel != nil {
			modelCancel()
		}
		modelDuration := time.Since(modelStart)
		if merr != nil {
			if rb.wallExpired() {
				result.StopReason = StopBudgetExceeded
				err = budgetError(BudgetWallTime)
				emit(Event{Type: EventTurnEnd})
				break
			}
			if errors.Is(merr, context.DeadlineExceeded) && ctx.Err() == nil {
				result.StopReason = StopTimeout
				err = fmt.Errorf("agent: model call: %w", ErrModelTimeout)
				emit(Event{Type: EventTurnEnd})
				break
			}
			return result, fmt.Errorf("agent: model call: %w", merr)
		}
		rb.addUsage(resp.Usage)
		result.Usage.Add(resp.Usage)
		if err := a.hooks.AfterModel(ctx, rc, resp); err != nil {
			return result, err
		}

		msg := resp.Message
		msg.Role = RoleAssistant
		if msg.Reasoning == "" && resp.Reasoning != "" {
			msg.Reasoning = resp.Reasoning
		}
		for i := range msg.ToolCalls {
			if msg.ToolCalls[i].ID == "" {
				msg.ToolCalls[i].ID = fmt.Sprintf("call_%s_%d_%d", rc.RunID[:8], turn, i)
			}
			if len(msg.ToolCalls[i].Arguments) == 0 {
				msg.ToolCalls[i].Arguments = json.RawMessage("{}")
			}
		}
		result.StopReason = resp.StopReason
		history = append(history, msg)
		pending = append(pending, msg)
		result.Messages = append(result.Messages, msg)
		usage := resp.Usage
		emit(Event{Type: EventMessageComplete, Message: &msg, Usage: &usage, Duration: modelDuration})
		if t := msg.Text(); t != "" {
			result.Output = t
		}
		if limit := rb.exceeded(); limit != "" {
			if len(msg.ToolCalls) == 0 {
				emit(Event{Type: EventTurnEnd})
				result.StopReason = StopBudgetExceeded
				err = budgetError(limit)
				break
			}
			toolMsgs := emitBudgetToolResults(msg.ToolCalls, limit, emit, true)
			pending = append(pending, toolMsgs...)
			result.Messages = append(result.Messages, toolMsgs...)
			if ferr := flush(); ferr != nil {
				return result, ferr
			}
			emit(Event{Type: EventTurnEnd})
			result.StopReason = StopBudgetExceeded
			err = budgetError(limit)
			break
		}

		if len(msg.ToolCalls) == 0 {
			result.Output = msg.Text()
			stopDec, stopErr := a.hooks.BeforeStop(ctx, rc, msg.Text())
			if stopErr != nil {
				return result, stopErr
			}
			if stopDec.Action == HookDeny || stopDec.Action == HookContinueWorking {
				feedback := stopDec.Reason
				if feedback == "" {
					feedback = "Please continue working."
				}
				feedbackMsg := UserText(fmt.Sprintf("<system-reminder>\n%s\n</system-reminder>", feedback))
				history = append(history, feedbackMsg)
				pending = append(pending, feedbackMsg)
				result.Messages = append(result.Messages, feedbackMsg)
				emit(Event{Type: EventTurnEnd})
				continue
			}
			if err = flush(); err != nil {
				return result, err
			}
			emit(Event{Type: EventTurnEnd})
			break
		}

		for i := range msg.ToolCalls {
			c := msg.ToolCalls[i]
			emit(Event{Type: EventToolCallReady, ToolCall: &c, ToolIndex: i})
		}
		if limit := rb.reserveToolCalls(len(msg.ToolCalls)); limit != "" {
			toolMsgs := emitBudgetToolResults(msg.ToolCalls, limit, emit, false)
			pending = append(pending, toolMsgs...)
			result.Messages = append(result.Messages, toolMsgs...)
			if ferr := flush(); ferr != nil {
				return result, ferr
			}
			emit(Event{Type: EventTurnEnd})
			result.StopReason = StopBudgetExceeded
			err = budgetError(limit)
			break
		}
		toolMsgs, batchErr := a.runTools(ctx, rc, activeReg, msg.ToolCalls, emit)
		history = append(history, toolMsgs...)
		pending = append(pending, toolMsgs...)
		result.Messages = append(result.Messages, toolMsgs...)
		if err := flush(); err != nil {
			return result, err
		}
		emit(Event{Type: EventTurnEnd})
		if batchErr != nil {
			return result, batchErr
		}
		if limit := rb.precheck(); limit != "" {
			result.StopReason = StopBudgetExceeded
			err = budgetError(limit)
			break
		}
		if cerr := ctx.Err(); cerr != nil {
			if rb.wallExpired() {
				result.StopReason = StopBudgetExceeded
				err = budgetError(BudgetWallTime)
				break
			}
			return result, cerr
		}
	}

	if ferr := flush(); ferr != nil && err == nil {
		err = ferr
	}
	finalize()
	usage := result.Usage
	emit(Event{Type: EventRunEnd, Result: result, Usage: &usage, Duration: time.Since(runStart)})
	return result, err
}

// callModel performs one model call in the requested mode. In stream mode
// deltas are emitted as they arrive and accumulated into a Response, so both
// modes produce the same Response shape.
func (a *Agent) callModel(ctx context.Context, req *Request, stream bool, emit emitFunc) (*Response, error) {
	if !stream {
		return a.cfg.provider.Generate(ctx, req)
	}
	var acc Accumulator
	for ev, err := range a.cfg.provider.Stream(ctx, req) {
		if err != nil {
			return nil, err
		}
		acc.Add(ev)
		ok := true
		switch ev.Type {
		case StreamTextDelta:
			ok = emit(Event{Type: EventTextDelta, Text: ev.Text})
		case StreamReasoningDelta:
			ok = emit(Event{Type: EventReasoningDelta, Text: ev.Text})
		case StreamToolCallStart:
			ok = emit(Event{Type: EventToolCallStart, ToolIndex: ev.Index, ToolCall: &ToolCall{ID: ev.ToolCallID, Name: ev.ToolName}})
		case StreamToolCallDelta:
			ok = emit(Event{Type: EventToolCallDelta, ToolIndex: ev.Index, Text: ev.ArgsDelta})
		}
		if !ok {
			if cerr := ctx.Err(); cerr != nil {
				return nil, cerr
			}
			return nil, context.Canceled
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return acc.Response(), nil
}

// runTools executes a batch of tool calls and returns tool-result messages in
// call order. EventToolResult and the messages carry the bounded result.
func (a *Agent) runTools(ctx context.Context, rc *RunContext, reg *ToolRegistry, calls []ToolCall, emit emitFunc) ([]Message, error) {
	mws := a.middleware
	if a.cfg.toolTimeout > 0 {
		// Innermost, so approval waits do not count against the tool timeout.
		mws = append(append([]ToolMiddleware(nil), mws...), TimeoutMiddleware(a.cfg.toolTimeout))
	}
	exec := ChainToolMiddleware(func(ctx context.Context, call ToolCall) (ToolResult, error) {
		t, _ := reg.Get(call.Name)
		return t.Call(ctx, call.Arguments)
	}, mws...)

	pipeline := func(ctx context.Context, call ToolCall) (ToolResult, error) {
		return a.handleCall(ctx, rc, reg, call, exec, emit)
	}
	isSeq := func(name string) bool {
		t, ok := reg.Get(name)
		return ok && t.Spec().Annotations.Sequential
	}
	ex := ToolExecutor{MaxParallel: a.cfg.maxParallel}
	results := ex.Run(ctx, calls, isSeq, pipeline, func(i int, cr *ToolCallResult) {
		content, truncated := truncateParts(cr.Result.Content, a.toolOutputLimit(reg, cr.Call.Name))
		cr.Result.Content = content
		cr.Result.Truncated = truncated
		call, res := cr.Call, cr.Result
		emit(Event{Type: EventToolResult, ToolCall: &call, ToolIndex: i, ToolResult: &res, Duration: cr.Duration})
	})

	batchErr := a.hooks.AfterToolBatch(ctx, rc, results)

	msgs := make([]Message, len(results))
	for i, cr := range results {
		content := cr.Result.Content
		if len(content) == 0 {
			content = []Part{TextPart("(no output)")}
		}
		msgs[i] = ToolResultMessage(cr.Call.ID, cr.Call.Name, content, cr.Result.IsError)
	}
	return msgs, batchErr
}

// toolOutputLimit resolves the model-facing output limit for one tool.
func (a *Agent) toolOutputLimit(reg *ToolRegistry, name string) int {
	limit := a.cfg.maxToolOutput
	if t, ok := reg.Get(name); ok {
		if n := t.Spec().MaxOutput; n > 0 {
			limit = n
		}
	}
	return limit
}

// emitBudgetToolResults builds and emits error tool results for calls that
// were not executed because the tool-call budget was exhausted. Every call gets
// a result so the assistant tool-use block stays valid.
func emitBudgetToolResults(calls []ToolCall, limit BudgetLimit, emit emitFunc, ready bool) []Message {
	text := fmt.Sprintf("Error: tool budget exceeded (%s)", limit)
	msgs := make([]Message, len(calls))
	for i, call := range calls {
		if ready {
			emit(Event{Type: EventToolCallReady, ToolCall: &call, ToolIndex: i})
		}
		res := ErrorResult(text)
		msgs[i] = ToolResultMessage(call.ID, call.Name, res.Content, true)
		emit(Event{Type: EventToolResult, ToolCall: &call, ToolIndex: i, ToolResult: &res})
	}
	return msgs
}

// truncateParts returns a bounded, sanitized copy of the text parts of a tool result.
func truncateParts(parts []Part, limit int) ([]Part, bool) {
	if len(parts) == 0 {
		return nil, false
	}
	out := make([]Part, len(parts))
	truncated := false
	for i, p := range parts {
		if p.Type == PartText {
			p.Text = sanitizeSystemReminderTags(p.Text)
			if limit > 0 && len(p.Text) > limit {
				origLen := len(p.Text)
				p.Text = fmt.Sprintf("%s\n...[output truncated from %d bytes]", utf8util.TruncateString(p.Text, limit), origLen)
				truncated = true
			}
		}
		out[i] = p
	}
	return out, truncated
}

func (a *Agent) checkPolicy(ctx context.Context, spec ToolSpec, call ToolCall) PolicyDecision {
	dec := a.cfg.policy.Check(ctx, spec, call)
	// NEVER downgrade PolicyAsk to PolicyAllow for tools annotated as Destructive.
	// Active skills can only pre-approve non-destructive tools.
	if dec == PolicyAsk && !spec.Annotations.Destructive && matchRules(ActiveSkillAllowedTools(ctx), spec, call) {
		dec = PolicyAllow
	}
	return dec
}

// handleCall applies validation, hooks, permission policy, and middleware to a
// single call. Hook rewrites (BeforeTool) are applied first, followed by policy
// evaluation on final arguments.
func (a *Agent) handleCall(ctx context.Context, rc *RunContext, reg *ToolRegistry, call ToolCall, exec ToolHandler, emit emitFunc) (ToolResult, error) {
	t, ok := reg.Get(call.Name)
	if !ok {
		var names []string
		for _, s := range reg.Specs() {
			names = append(names, s.Name)
		}
		return ErrorResult(fmt.Sprintf("Error: unknown tool %q. Available tools: %s", call.Name, strings.Join(names, ", "))), nil
	}
	if !json.Valid(call.Arguments) {
		return ErrorResult(fmt.Sprintf("Error: arguments for %s are not valid JSON: %s", call.Name, call.Arguments)), nil
	}
	spec := t.Spec()

	// 1. Apply BeforeTool hook rewrites first so policy evaluates final arguments.
	origArgs := call.Arguments
	d, err := a.hooks.BeforeTool(ctx, rc, &call)
	if err != nil {
		return ToolResult{}, err
	}
	switch d.Action {
	case HookDeny:
		reason := d.Reason
		if reason == "" {
			reason = "blocked by hook"
		}
		return ErrorResult("Error: tool call blocked: " + reason), nil
	case HookOverride:
		res := d.Result
		if err := a.hooks.AfterTool(ctx, rc, call, &res); err != nil {
			return res, err
		}
		return res, nil
	}

	if !bytes.Equal(call.Arguments, origArgs) {
		if !json.Valid(call.Arguments) {
			return ErrorResult(fmt.Sprintf("Error: hook rewrote arguments for %s to invalid JSON: %s", call.Name, call.Arguments)), nil
		}
		slog.Info("tool call arguments rewritten by BeforeTool", "tool", call.Name, "from", redactSecrets(origArgs), "to", redactSecrets(call.Arguments))
		emit(Event{
			Type:         EventToolCallRewritten,
			ToolCall:     &call,
			OriginalArgs: origArgs,
			Text:         fmt.Sprintf("BeforeTool rewrote arguments for %s from %s to %s", call.Name, redactSecrets(origArgs), redactSecrets(call.Arguments)),
		})
	}

	// 2. Evaluate permission policy on the final arguments.
	dec := a.checkPolicy(ctx, spec, call)

	switch dec {
	case PolicyDeny:
		a.hooks.PermissionDenied(ctx, rc, call, "denied by policy")
		return ErrorResult("Error: this tool call was denied by policy."), nil
	case PolicyAsk:
		req := ApprovalRequest{AgentName: a.cfg.name, Spec: spec, Call: call}
		hookDec, decErr := a.hooks.PermissionRequest(ctx, rc, req)
		if decErr != nil {
			return ToolResult{}, decErr
		}
		var approved bool
		if hookDec.Handled {
			approved = hookDec.Approved
			if len(hookDec.UpdatedArguments) > 0 && !bytes.Equal(hookDec.UpdatedArguments, call.Arguments) {
				reqArgs := call.Arguments
				call.Arguments = hookDec.UpdatedArguments
				if !json.Valid(call.Arguments) {
					return ErrorResult(fmt.Sprintf("Error: OnPermissionRequest rewrote arguments for %s to invalid JSON: %s", call.Name, call.Arguments)), nil
				}
				slog.Info("tool call arguments rewritten by OnPermissionRequest", "tool", call.Name, "from", redactSecrets(reqArgs), "to", redactSecrets(call.Arguments))
				emit(Event{
					Type:         EventToolCallRewritten,
					ToolCall:     &call,
					OriginalArgs: reqArgs,
					Text:         fmt.Sprintf("OnPermissionRequest rewrote arguments for %s from %s to %s", call.Name, redactSecrets(reqArgs), redactSecrets(call.Arguments)),
				})

				// Re-evaluate policy on updated arguments.
				newDec := a.checkPolicy(ctx, spec, call)
				if newDec == PolicyDeny {
					a.hooks.PermissionDenied(ctx, rc, call, "denied by policy after argument rewrite")
					return ErrorResult("Error: this tool call was denied by policy after argument rewrite."), nil
				}
				if newDec == PolicyAsk && !approved {
					a.hooks.PermissionDenied(ctx, rc, call, "denied by policy")
					return ErrorResult("Error: this tool call was denied by policy."), nil
				}
			}
		} else {
			var err error
			approved, err = a.askApproval(ctx, spec, call, emit)
			if err != nil {
				return ToolResult{}, fmt.Errorf("approval failed: %w", err)
			}
		}
		if !approved {
			a.hooks.PermissionDenied(ctx, rc, call, "denied by user")
			return ErrorResult("Error: the user denied this tool call."), nil
		}
	}

	// 3. Execute tool handler with approved final arguments.
	res, err := exec(ctx, call)
	if err != nil {
		if ferr := a.hooks.ToolFailure(ctx, rc, call, err); ferr != nil {
			return res, ferr
		}
		return res, err
	}
	if err := a.hooks.AfterTool(ctx, rc, call, &res); err != nil {
		return res, err
	}
	return res, nil
}

func (a *Agent) askApproval(ctx context.Context, spec ToolSpec, call ToolCall, emit emitFunc) (bool, error) {
	if a.cfg.approver == nil {
		return false, nil
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	req := ApprovalRequest{AgentName: a.cfg.name, Spec: spec, Call: call}
	emit(Event{Type: EventApprovalRequested, ToolCall: &call, Approval: &req})
	return a.cfg.approver.Approve(ctx, req)
}

// sanitizeSystemReminderTags escapes literal <system-reminder> tags in untrusted
// user or tool content so they cannot spoof framework system reminders.
func sanitizeSystemReminderTags(s string) string {
	if !strings.Contains(strings.ToLower(s), "system-reminder") {
		return s
	}
	r := strings.NewReplacer(
		"<system-reminder>", "&lt;system-reminder&gt;",
		"</system-reminder>", "&lt;/system-reminder&gt;",
		"<SYSTEM-REMINDER>", "&lt;SYSTEM-REMINDER&gt;",
		"</SYSTEM-REMINDER>", "&lt;/SYSTEM-REMINDER&gt;",
	)
	return r.Replace(s)
}

// redactSecrets masks sensitive fields in JSON arguments before logging.
func redactSecrets(raw json.RawMessage) string {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return string(raw)
	}
	redacted := false
	for k := range m {
		lower := strings.ToLower(k)
		if strings.Contains(lower, "password") || strings.Contains(lower, "secret") ||
			strings.Contains(lower, "token") || strings.Contains(lower, "api_key") ||
			strings.Contains(lower, "apikey") || strings.Contains(lower, "authorization") ||
			strings.Contains(lower, "credential") || strings.Contains(lower, "private_key") {
			m[k] = "[REDACTED]"
			redacted = true
		}
	}
	if !redacted {
		return string(raw)
	}
	b, err := json.Marshal(m)
	if err != nil {
		return string(raw)
	}
	return string(b)
}

// terminalStopReason maps a run error to a stop reason. Explicit terminal
// reasons set by the loop are preserved when there is no error.
func terminalStopReason(err error, current StopReason) StopReason {
	if err == nil {
		if current == "" {
			return StopEnd
		}
		return current
	}
	switch {
	case errors.Is(err, ErrBudgetExceeded):
		return StopBudgetExceeded
	case errors.Is(err, ErrModelTimeout):
		return StopTimeout
	case errors.Is(err, ErrMaxTurns):
		return StopMaxTurns
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return StopCanceled
	default:
		return StopError
	}
}

func newID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
