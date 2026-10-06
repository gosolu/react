package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/gosolu/react/hook"
	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/permission"
	"github.com/gosolu/react/tool"
)

type emitFunc func(Event) bool

// execute is the single ReAct loop shared by Run and Stream.
func (a *Agent) execute(ctx context.Context, sessionID, input string, rcfg runConfig, stream bool, sink emitFunc) (result *Result, err error) {
	depth := depthFrom(ctx)
	if depth > a.cfg.maxDepth {
		return nil, ErrMaxDepth
	}
	if strings.TrimSpace(input) == "" && len(rcfg.parts) == 0 {
		return nil, errors.New("agent: empty input")
	}

	rc := &hook.RunContext{AgentName: a.cfg.name, SessionID: sessionID, RunID: newID(), Depth: depth}
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

	ctx = hook.WithRunContext(ctx, rc)
	ctx = context.WithValue(ctx, depthKey{}, depth)
	ctx = context.WithValue(ctx, emitterKey{}, emitFunc(emit))
	defer func() { a.hooks.RunEnd(ctx, rc, err) }()

	if err := a.hooks.RunStart(ctx, rc); err != nil {
		return nil, err
	}

	// Load history.
	var history []llm.Message
	if sessionID != "" {
		if history, err = a.cfg.store.Load(ctx, sessionID); err != nil {
			return nil, fmt.Errorf("agent: load session: %w", err)
		}
	} else {
		history = llm.CloneMessages(rcfg.history)
	}
	user := llm.Message{Role: llm.RoleUser}
	if input != "" {
		user.Content = append(user.Content, llm.TextPart(input))
	}
	user.Content = append(user.Content, rcfg.parts...)
	history = append(history, user)

	result = &Result{Messages: []llm.Message{user}}
	pending := []llm.Message{user}
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
	var specs []llm.ToolSpec
	for _, s := range a.registry.Specs() {
		specs = append(specs, s.LLM())
	}

	emit(Event{Type: EventRunStart})

	for turn := 1; ; turn++ {
		if turn > a.cfg.maxTurns {
			result.StopReason = llm.StopMaxTurns
			err = ErrMaxTurns
			break
		}
		rc.Turn = turn
		result.Turns = turn
		emit(Event{Type: EventTurnStart})

		msgs, ferr := a.cfg.ctxManager.Fit(ctx, a.system, history, a.cfg.contextBudget)
		if ferr != nil {
			return result, fmt.Errorf("agent: context manager: %w", ferr)
		}
		req := &llm.Request{
			Model:       model,
			System:      a.system,
			Messages:    msgs,
			Tools:       specs,
			MaxTokens:   a.cfg.maxTokens,
			Temperature: a.cfg.temperature,
		}
		// A forced tool choice applies to the first turn only, otherwise the
		// model could never produce a final answer.
		if rcfg.toolChoice != "" && (turn == 1 || rcfg.toolChoice == llm.ToolChoiceNone || rcfg.toolChoice == llm.ToolChoiceAuto) {
			req.ToolChoice = rcfg.toolChoice
		}
		if err := a.hooks.BeforeModel(ctx, rc, req); err != nil {
			return result, err
		}

		resp, merr := a.callModel(ctx, req, stream, emit)
		if merr != nil {
			return result, fmt.Errorf("agent: model call: %w", merr)
		}
		if err := a.hooks.AfterModel(ctx, rc, resp); err != nil {
			return result, err
		}

		msg := resp.Message
		msg.Role = llm.RoleAssistant
		for i := range msg.ToolCalls {
			if msg.ToolCalls[i].ID == "" {
				msg.ToolCalls[i].ID = fmt.Sprintf("call_%s_%d_%d", rc.RunID[:8], turn, i)
			}
			if len(msg.ToolCalls[i].Arguments) == 0 {
				msg.ToolCalls[i].Arguments = json.RawMessage("{}")
			}
		}
		result.Usage.Add(resp.Usage)
		result.StopReason = resp.StopReason
		history = append(history, msg)
		pending = append(pending, msg)
		result.Messages = append(result.Messages, msg)
		usage := resp.Usage
		emit(Event{Type: EventMessageComplete, Message: &msg, Usage: &usage})

		if len(msg.ToolCalls) == 0 {
			result.Output = msg.Text()
			if err := flush(); err != nil {
				return result, err
			}
			emit(Event{Type: EventTurnEnd})
			break
		}
		if t := msg.Text(); t != "" {
			result.Output = t
		}

		for i := range msg.ToolCalls {
			c := msg.ToolCalls[i]
			emit(Event{Type: EventToolCallReady, ToolCall: &c, ToolIndex: i})
		}
		toolMsgs := a.runTools(ctx, rc, msg.ToolCalls, emit)
		history = append(history, toolMsgs...)
		pending = append(pending, toolMsgs...)
		result.Messages = append(result.Messages, toolMsgs...)
		if err := flush(); err != nil {
			return result, err
		}
		emit(Event{Type: EventTurnEnd})
		if cerr := ctx.Err(); cerr != nil {
			return result, cerr
		}
	}

	if ferr := flush(); ferr != nil && err == nil {
		err = ferr
	}
	usage := result.Usage
	emit(Event{Type: EventRunEnd, Result: result, Usage: &usage})
	return result, err
}

// callModel performs one model call in the requested mode. In stream mode
// deltas are emitted as they arrive and accumulated into a Response, so both
// modes produce the same Response shape.
func (a *Agent) callModel(ctx context.Context, req *llm.Request, stream bool, emit emitFunc) (*llm.Response, error) {
	if !stream {
		return a.cfg.provider.Generate(ctx, req)
	}
	var acc llm.Accumulator
	for ev, err := range a.cfg.provider.Stream(ctx, req) {
		if err != nil {
			return nil, err
		}
		acc.Add(ev)
		switch ev.Type {
		case llm.StreamTextDelta:
			emit(Event{Type: EventTextDelta, Text: ev.Text})
		case llm.StreamReasoningDelta:
			emit(Event{Type: EventReasoningDelta, Text: ev.Text})
		case llm.StreamToolCallStart:
			emit(Event{Type: EventToolCallStart, ToolIndex: ev.Index, ToolCall: &llm.ToolCall{ID: ev.ToolCallID, Name: ev.ToolName}})
		case llm.StreamToolCallDelta:
			emit(Event{Type: EventToolCallDelta, ToolIndex: ev.Index, Text: ev.ArgsDelta})
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return acc.Response(), nil
}

// runTools executes a batch of tool calls and returns tool-result messages in
// call order.
func (a *Agent) runTools(ctx context.Context, rc *hook.RunContext, calls []llm.ToolCall, emit emitFunc) []llm.Message {
	mws := a.middleware
	if a.cfg.toolTimeout > 0 {
		// Innermost, so approval waits do not count against the tool timeout.
		mws = append(append([]tool.Middleware(nil), mws...), tool.TimeoutMiddleware(a.cfg.toolTimeout))
	}
	exec := tool.Chain(func(ctx context.Context, call llm.ToolCall) (tool.Result, error) {
		t, _ := a.registry.Get(call.Name)
		return t.Call(ctx, call.Arguments)
	}, mws...)

	pipeline := func(ctx context.Context, call llm.ToolCall) (tool.Result, error) {
		return a.handleCall(ctx, rc, call, exec, emit)
	}
	isSeq := func(name string) bool {
		t, ok := a.registry.Get(name)
		return ok && t.Spec().Annotations.Sequential
	}
	ex := tool.Executor{MaxParallel: a.cfg.maxParallel}
	results := ex.Run(ctx, calls, isSeq, pipeline, func(i int, cr tool.CallResult) {
		call, res := cr.Call, cr.Result
		emit(Event{Type: EventToolResult, ToolCall: &call, ToolIndex: i, ToolResult: &res})
	})

	msgs := make([]llm.Message, len(results))
	for i, cr := range results {
		content := cr.Result.Content
		if len(content) == 0 {
			content = []llm.Part{llm.TextPart("(no output)")}
		}
		msgs[i] = llm.ToolResult(cr.Call.ID, cr.Call.Name, content, cr.Result.IsError)
	}
	return msgs
}

// handleCall applies validation, permission policy, hooks and middleware to a
// single call.
func (a *Agent) handleCall(ctx context.Context, rc *hook.RunContext, call llm.ToolCall, exec tool.Handler, emit emitFunc) (tool.Result, error) {
	t, ok := a.registry.Get(call.Name)
	if !ok {
		var names []string
		for _, s := range a.registry.Specs() {
			names = append(names, s.Name)
		}
		return tool.ErrorResult(fmt.Sprintf("Error: unknown tool %q. Available tools: %s", call.Name, strings.Join(names, ", "))), nil
	}
	if !json.Valid(call.Arguments) {
		return tool.ErrorResult(fmt.Sprintf("Error: arguments for %s are not valid JSON: %s", call.Name, call.Arguments)), nil
	}
	spec := t.Spec()

	switch a.cfg.policy.Check(ctx, spec, call) {
	case permission.Deny:
		return tool.ErrorResult("Error: this tool call was denied by policy."), nil
	case permission.Ask:
		approved, err := a.askApproval(ctx, spec, call, emit)
		if err != nil {
			return tool.Result{}, fmt.Errorf("approval failed: %w", err)
		}
		if !approved {
			return tool.ErrorResult("Error: the user denied this tool call."), nil
		}
	}

	d, err := a.hooks.BeforeTool(ctx, rc, &call)
	if err != nil {
		return tool.Result{}, err
	}
	var res tool.Result
	switch d.Action {
	case hook.Deny:
		reason := d.Reason
		if reason == "" {
			reason = "blocked by hook"
		}
		return tool.ErrorResult("Error: tool call blocked: " + reason), nil
	case hook.Override:
		res = d.Result
	default:
		res, err = exec(ctx, call)
		if err != nil {
			res = errorResult(err, res)
		}
	}
	if err := a.hooks.AfterTool(ctx, rc, call, &res); err != nil {
		return res, err
	}
	return res, nil
}

func (a *Agent) askApproval(ctx context.Context, spec tool.Spec, call llm.ToolCall, emit emitFunc) (bool, error) {
	req := permission.Request{AgentName: a.cfg.name, Spec: spec, Call: call}
	emit(Event{Type: EventApprovalRequested, ToolCall: &call, Approval: &req})
	if a.cfg.approver == nil {
		return false, nil
	}
	a.approvalMu.Lock()
	defer a.approvalMu.Unlock()
	if err := ctx.Err(); err != nil {
		return false, err
	}
	return a.cfg.approver.Approve(ctx, req)
}

func errorResult(err error, partial tool.Result) tool.Result {
	msg := err.Error()
	if errors.Is(err, context.DeadlineExceeded) {
		msg = "tool call timed out"
	}
	partial.IsError = true
	partial.Content = append([]llm.Part{llm.TextPart("Error: " + msg)}, partial.Content...)
	return partial
}

func newID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
