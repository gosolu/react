package react

import (
	"encoding/json"
	"iter"
	"strings"
)

// StreamEventType identifies a StreamEvent.
type StreamEventType int

const (
	// StreamTextDelta carries a fragment of assistant text in Text.
	StreamTextDelta StreamEventType = iota + 1
	// StreamReasoningDelta carries a fragment of reasoning text in Text.
	StreamReasoningDelta
	// StreamToolCallStart announces a new tool call at Index with ToolCallID/ToolName.
	StreamToolCallStart
	// StreamToolCallDelta carries a fragment of JSON arguments for the call at Index.
	StreamToolCallDelta
	// StreamUsage reports token usage (fields that are zero are "unknown").
	StreamUsage
	// StreamDone signals the end of the response with StopReason.
	StreamDone
)

// StreamEvent is a single incremental event from a streaming provider.
type StreamEvent struct {
	Type       StreamEventType
	Text       string
	Index      int
	ToolCallID string
	ToolName   string
	ArgsDelta  string
	Usage      *Usage
	StopReason StopReason
}

type toolAcc struct {
	id, name string
	args     strings.Builder
}

// Accumulator folds a sequence of StreamEvents into a Response.
// The zero value is ready to use.
type Accumulator struct {
	text      strings.Builder
	reasoning strings.Builder
	calls     []*toolAcc
	byIndex   map[int]*toolAcc
	usage     Usage
	stop      StopReason
}

// Add applies an event.
func (a *Accumulator) Add(ev StreamEvent) {
	switch ev.Type {
	case StreamTextDelta:
		a.text.WriteString(ev.Text)
	case StreamReasoningDelta:
		a.reasoning.WriteString(ev.Text)
	case StreamToolCallStart:
		c := a.call(ev.Index)
		if ev.ToolCallID != "" {
			c.id = ev.ToolCallID
		}
		if ev.ToolName != "" {
			c.name = ev.ToolName
		}
		c.args.WriteString(ev.ArgsDelta)
	case StreamToolCallDelta:
		a.call(ev.Index).args.WriteString(ev.ArgsDelta)
	case StreamUsage:
		if ev.Usage != nil {
			if ev.Usage.InputTokens > 0 {
				a.usage.InputTokens = ev.Usage.InputTokens
			}
			if ev.Usage.OutputTokens > 0 {
				a.usage.OutputTokens = ev.Usage.OutputTokens
			}
		}
	case StreamDone:
		if ev.StopReason != "" {
			a.stop = ev.StopReason
		}
	}
}

func (a *Accumulator) call(idx int) *toolAcc {
	if a.byIndex == nil {
		a.byIndex = map[int]*toolAcc{}
	}
	c, ok := a.byIndex[idx]
	if !ok {
		c = &toolAcc{}
		a.byIndex[idx] = c
		a.calls = append(a.calls, c)
	}
	return c
}

// Text returns the assistant text accumulated so far.
func (a *Accumulator) Text() string { return a.text.String() }

// Response builds the final response.
func (a *Accumulator) Response() *Response {
	msg := Message{Role: RoleAssistant}
	if t := a.text.String(); t != "" {
		msg.Content = []Part{TextPart(t)}
	}
	for _, c := range a.calls {
		args := strings.TrimSpace(c.args.String())
		if args == "" {
			args = "{}"
		}
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{ID: c.id, Name: c.name, Arguments: json.RawMessage(args)})
	}
	stop := a.stop
	if stop == "" || (stop == StopEnd && len(msg.ToolCalls) > 0) {
		if len(msg.ToolCalls) > 0 {
			stop = StopToolCall
		} else {
			stop = StopEnd
		}
	}
	return &Response{Message: msg, Reasoning: a.reasoning.String(), StopReason: stop, Usage: a.usage}
}

// StreamResponse converts a complete Response into a stream of events,
// splitting text and arguments into chunks of chunkSize bytes (<=0 means one
// chunk). Useful for providers without native streaming and for tests.
func StreamResponse(resp *Response, chunkSize int) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		if resp.Reasoning != "" {
			for _, c := range chunk(resp.Reasoning, chunkSize) {
				if !yield(StreamEvent{Type: StreamReasoningDelta, Text: c}, nil) {
					return
				}
			}
		}
		if text := resp.Message.Text(); text != "" {
			for _, c := range chunk(text, chunkSize) {
				if !yield(StreamEvent{Type: StreamTextDelta, Text: c}, nil) {
					return
				}
			}
		}
		for i, call := range resp.Message.ToolCalls {
			if !yield(StreamEvent{Type: StreamToolCallStart, Index: i, ToolCallID: call.ID, ToolName: call.Name}, nil) {
				return
			}
			for _, c := range chunk(string(call.Arguments), chunkSize) {
				if !yield(StreamEvent{Type: StreamToolCallDelta, Index: i, ArgsDelta: c}, nil) {
					return
				}
			}
		}
		if resp.Usage.Total() > 0 {
			u := resp.Usage
			if !yield(StreamEvent{Type: StreamUsage, Usage: &u}, nil) {
				return
			}
		}
		yield(StreamEvent{Type: StreamDone, StopReason: resp.StopReason}, nil)
	}
}

func chunk(s string, n int) []string {
	if n <= 0 || len(s) <= n {
		if s == "" {
			return nil
		}
		return []string{s}
	}
	var out []string
	for len(s) > n {
		out = append(out, s[:n])
		s = s[n:]
	}
	if len(s) > 0 {
		out = append(out, s)
	}
	return out
}
