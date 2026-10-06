// Package contextmgr keeps conversation history within a model's context
// window.
package contextmgr

import (
	"context"
	"slices"

	"github.com/gosolu/react/llm"
)

// Manager trims or compacts messages so that system+messages fit within
// budget tokens. A budget <= 0 disables management. Implementations must not
// split an assistant tool-call message from its tool results and must not
// mutate the input slice.
type Manager interface {
	Fit(ctx context.Context, system string, msgs []llm.Message, budget int) ([]llm.Message, error)
}

// Estimator estimates the token count of a message.
type Estimator func(llm.Message) int

// EstimateTokens is a cheap heuristic (~4 bytes per token plus overhead).
func EstimateTokens(m llm.Message) int {
	n := 4
	for _, p := range m.Content {
		switch p.Type {
		case llm.PartText:
			n += len(p.Text) / 4
		case llm.PartImage:
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

func estimator(e Estimator) Estimator {
	if e == nil {
		return EstimateTokens
	}
	return e
}

// unit is a span [start, end) of messages that must be kept or dropped together.
type unit struct{ start, end, tokens int }

func splitUnits(msgs []llm.Message, est Estimator) []unit {
	var us []unit
	for i := 0; i < len(msgs); {
		j := i + 1
		if msgs[i].Role == llm.RoleAssistant && len(msgs[i].ToolCalls) > 0 {
			for j < len(msgs) && msgs[j].Role == llm.RoleTool {
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
func lastUserUnit(msgs []llm.Message, us []unit) int {
	for i, u := range slices.Backward(us) {
		if msgs[u.start].Role == llm.RoleUser {
			return i
		}
	}
	return len(us) - 1
}

// Window drops the oldest messages until the conversation fits. It always
// keeps the most recent user message and everything after it, and makes the
// retained history start with a user message.
type Window struct {
	Estimator Estimator
}

// Fit implements Manager.
func (w Window) Fit(_ context.Context, system string, msgs []llm.Message, budget int) ([]llm.Message, error) {
	return window(system, msgs, budget, estimator(w.Estimator)), nil
}

func window(system string, msgs []llm.Message, budget int, est Estimator) []llm.Message {
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
	for start < keep && msgs[us[start].start].Role != llm.RoleUser {
		start++
	}
	return msgs[us[start].start:]
}
