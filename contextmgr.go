// Context management keeps conversation history within a model's context
// window.
package react

import (
	"context"
	"slices"

)

// ContextManager trims or compacts messages so that system+messages fit within
// budget tokens. A budget <= 0 disables management. Implementations must not
// split an assistant tool-call message from its tool results and must not
// mutate the input slice.
type ContextManager interface {
	Fit(ctx context.Context, system string, msgs []Message, budget int) ([]Message, error)
}

// TokenEstimator estimates the token count of a message.
type TokenEstimator func(Message) int

// EstimateTokens is a cheap heuristic (~4 bytes per token plus overhead).
func EstimateTokens(m Message) int {
	n := 4
	for _, p := range m.Content {
		switch p.Type {
		case PartText:
			n += len(p.Text) / 4
		case PartImage:
			n += 1000
		}
	}
	for _, c := range m.ToolCalls {
		n += (len(c.Name) + len(c.Arguments)) / 4
	}
	return n
}

// EstimateText estimates tokens for a plain string.
func EstimateText(s string) int { return len(s) / 4 }

func estimator(e TokenEstimator) TokenEstimator {
	if e == nil {
		return EstimateTokens
	}
	return e
}

// unit is a span [start, end) of messages that must be kept or dropped together.
type unit struct{ start, end, tokens int }

func splitUnits(msgs []Message, est TokenEstimator) []unit {
	var us []unit
	for i := 0; i < len(msgs); {
		j := i + 1
		if msgs[i].Role == RoleAssistant && len(msgs[i].ToolCalls) > 0 {
			for j < len(msgs) && msgs[j].Role == RoleTool {
				j++
			}
		}
		t := 0
		for k := i; k < j; k++ {
			t += est(msgs[k])
		}
		us = append(us, unit{i, j, t})
		i = j
	}
	return us
}

// lastUserUnit returns the index of the last unit that starts with a user message.
func lastUserUnit(msgs []Message, us []unit) int {
	for i, u := range slices.Backward(us) {
		if msgs[u.start].Role == RoleUser {
			return i
		}
	}
	return len(us) - 1
}

// ContextWindow drops the oldest messages until the conversation fits. It always
// keeps the most recent user message and everything after it, and makes the
// retained history start with a user message.
type ContextWindow struct {
	Estimator TokenEstimator
}

// Fit implements ContextManager.
func (w ContextWindow) Fit(_ context.Context, system string, msgs []Message, budget int) ([]Message, error) {
	return window(system, msgs, budget, estimator(w.Estimator)), nil
}

func window(system string, msgs []Message, budget int, est TokenEstimator) []Message {
	if budget <= 0 || len(msgs) == 0 {
		return msgs
	}
	avail := budget - EstimateText(system)
	us := splitUnits(msgs, est)
	total := 0
	for _, u := range us {
		total += u.tokens
	}
	if total <= avail {
		return msgs
	}
	keep := lastUserUnit(msgs, us)
	start := 0
	for total > avail && start < keep {
		total -= us[start].tokens
		start++
	}
	for start < keep && msgs[us[start].start].Role != RoleUser {
		start++
	}
	return msgs[us[start].start:]
}
