// Package mock provides a scripted react.Provider for tests and offline demos.
package mock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"sync"

	"github.com/gosolu/react"
)

// ErrExhausted is returned when the script has no more steps.
var ErrExhausted = errors.New("mock: script exhausted")

// Step is one scripted model turn.
type Step struct {
	Response *react.Response
	Err      error
}

// Text scripts a final text answer.
func Text(s string) Step {
	return Step{Response: &react.Response{
		Message:    react.AssistantText(s),
		StopReason: react.StopEnd,
		Usage:      react.Usage{InputTokens: 10, OutputTokens: len(s)/4 + 1},
	}}
}

// ToolCalls scripts an assistant turn that calls tools, with optional
// leading text.
func ToolCalls(text string, calls ...react.ToolCall) Step {
	msg := react.Message{Role: react.RoleAssistant, ToolCalls: calls}
	if text != "" {
		msg.Content = []react.Part{react.TextPart(text)}
	}
	return Step{Response: &react.Response{Message: msg, StopReason: react.StopToolCall, Usage: react.Usage{InputTokens: 10, OutputTokens: 5}}}
}

// Error scripts a provider failure.
func Error(err error) Step { return Step{Err: err} }

var callSeq struct {
	sync.Mutex
	n int
}

// Call builds a tool call with an auto-generated ID.
func Call(name string, args any) react.ToolCall {
	var raw json.RawMessage
	switch a := args.(type) {
	case string:
		raw = json.RawMessage(a)
	case json.RawMessage:
		raw = a
	default:
		b, err := json.Marshal(a)
		if err != nil {
			panic(err)
		}
		raw = b
	}
	callSeq.Lock()
	callSeq.n++
	id := fmt.Sprintf("call_%d", callSeq.n)
	callSeq.Unlock()
	return react.ToolCall{ID: id, Name: name, Arguments: raw}
}

// HandlerFunc computes a response dynamically from the request.
type HandlerFunc func(ctx context.Context, req *react.Request) (*react.Response, error)

// Provider is a scripted provider. It is safe for concurrent use.
type Provider struct {
	// ChunkSize controls how responses are split when streaming (default 4).
	ChunkSize int

	mu       sync.Mutex
	steps    []Step
	next     int
	handler  HandlerFunc
	requests []react.Request
}

// New returns a provider that replays steps in order.
func New(steps ...Step) *Provider { return &Provider{steps: steps} }

// NewFunc returns a provider that computes each response with h.
func NewFunc(h HandlerFunc) *Provider { return &Provider{handler: h} }

// Name implements react.Provider.
func (p *Provider) Name() string { return "mock" }

// Requests returns copies of all requests received so far.
func (p *Provider) Requests() []react.Request {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]react.Request(nil), p.requests...)
}

func (p *Provider) respond(ctx context.Context, req *react.Request) (*react.Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	cp := *req
	cp.Messages = react.CloneMessages(req.Messages)
	cp.Tools = append([]react.ToolSpec(nil), req.Tools...)

	p.mu.Lock()
	p.requests = append(p.requests, cp)
	h := p.handler
	var step Step
	var exhausted bool
	if h == nil {
		if p.next >= len(p.steps) {
			exhausted = true
		} else {
			step = p.steps[p.next]
			p.next++
		}
	}
	p.mu.Unlock()

	if h != nil {
		return h(ctx, &cp)
	}
	if exhausted {
		return nil, ErrExhausted
	}
	if step.Err != nil {
		return nil, step.Err
	}
	r := *step.Response
	r.Message = r.Message.Clone()
	return &r, nil
}

// Generate implements react.Provider.
func (p *Provider) Generate(ctx context.Context, req *react.Request) (*react.Response, error) {
	return p.respond(ctx, req)
}

// Stream implements react.Provider by chunking the scripted response.
func (p *Provider) Stream(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		resp, err := p.respond(ctx, req)
		if err != nil {
			yield(react.StreamEvent{}, err)
			return
		}
		size := p.ChunkSize
		if size == 0 {
			size = 4
		}
		for ev, err := range react.StreamResponse(resp, size) {
			if ctx.Err() != nil {
				yield(react.StreamEvent{}, ctx.Err())
				return
			}
			if !yield(ev, err) {
				return
			}
		}
	}
}
