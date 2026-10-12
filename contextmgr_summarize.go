package react

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
)

// DefaultSummaryPrompt instructs the model how to compact history.
const DefaultSummaryPrompt = `You compress conversation history for an AI agent.
Write a concise but complete summary of the conversation below, preserving:
user goals and constraints, decisions made, important facts and values,
tool calls performed and their key results, and any unfinished work.
Output only the summary.`

// ContextSummarizer replaces older history with an LLM-written summary when the
// conversation exceeds the budget. Recent history (up to RecentRatio of the
// budget, always including the latest user message) is kept verbatim.
// Summaries are cached by content hash so repeated calls with the same prefix
// do not re-invoke the model.
type ContextSummarizer struct {
	Provider    Provider
	Model       string
	Prompt      string         // defaults to DefaultSummaryPrompt
	RecentRatio float64        // fraction of budget reserved for recent messages (default 0.5)
	MaxTokens   int            // summary length limit (default 1024)
	Estimator   TokenEstimator // defaults to EstimateTokens

	mu    sync.Mutex
	cache map[string]string
}

// Fit implements ContextManager.
func (s *ContextSummarizer) Fit(ctx context.Context, system string, msgs []Message, budget int) ([]Message, error) {
	est := estimator(s.Estimator)
	if budget <= 0 || len(msgs) == 0 {
		return msgs, nil
	}
	avail := budget - EstimateText(system)
	us := splitUnits(msgs, est)
	total := 0
	for _, u := range us {
		total += u.tokens
	}
	if total <= avail {
		return msgs, nil
	}

	ratio := s.RecentRatio
	if ratio <= 0 || ratio >= 1 {
		ratio = 0.5
	}
	// Choose the earliest user-starting unit whose suffix fits in the recent budget.
	keep := lastUserUnit(msgs, us)
	cut := keep
	suffix := 0
	for i := len(us) - 1; i >= keep; i-- {
		suffix += us[i].tokens
	}
	for i := keep - 1; i >= 0; i-- {
		suffix += us[i].tokens
		if float64(suffix) > ratio*float64(avail) {
			break
		}
		if msgs[us[i].start].Role == RoleUser {
			cut = i
		}
	}
	cutMsg := us[cut].start
	if cutMsg == 0 {
		return window(system, msgs, budget, est), nil
	}

	summary, err := s.summarize(ctx, msgs[:cutMsg])
	if err != nil {
		return nil, fmt.Errorf("contextmgr: summarize: %w", err)
	}
	out := make([]Message, 0, len(msgs)-cutMsg+2)
	out = append(out,
		UserText("[Summary of the earlier conversation]\n"+summary),
		AssistantText("Understood. I'll continue from this summary."),
	)
	out = append(out, msgs[cutMsg:]...)
	return window(system, out, budget, est), nil
}

func (s *ContextSummarizer) summarize(ctx context.Context, msgs []Message) (string, error) {
	transcript := Render(msgs, 2000)
	sum := sha256.Sum256([]byte(transcript))
	key := hex.EncodeToString(sum[:])

	s.mu.Lock()
	if v, ok := s.cache[key]; ok {
		s.mu.Unlock()
		return v, nil
	}
	s.mu.Unlock()

	prompt := s.Prompt
	if prompt == "" {
		prompt = DefaultSummaryPrompt
	}
	maxTok := s.MaxTokens
	if maxTok <= 0 {
		maxTok = 1024
	}
	resp, err := s.Provider.Generate(ctx, &Request{
		Model:     s.Model,
		System:    prompt,
		Messages:  []Message{UserText(transcript)},
		MaxTokens: maxTok,
	})
	if err != nil {
		return "", err
	}
	text := strings.TrimSpace(resp.Message.Text())

	s.mu.Lock()
	if s.cache == nil {
		s.cache = map[string]string{}
	}
	if len(s.cache) > 64 {
		clear(s.cache)
	}
	s.cache[key] = text
	s.mu.Unlock()
	return text, nil
}

// Render formats messages as a plain-text transcript, truncating long tool
// results to maxResult bytes.
func Render(msgs []Message, maxResult int) string {
	var b strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case RoleTool:
			t := m.Text()
			if maxResult > 0 && len(t) > maxResult {
				t = t[:maxResult] + "...[truncated]"
			}
			status := "result"
			if m.IsError {
				status = "error"
			}
			fmt.Fprintf(&b, "[tool %s %s]: %s\n", m.Name, status, t)
		default:
			if t := m.Text(); t != "" {
				fmt.Fprintf(&b, "%s: %s\n", m.Role, t)
			}
			for _, c := range m.ToolCalls {
				fmt.Fprintf(&b, "%s called tool %s(%s)\n", m.Role, c.Name, c.Arguments)
			}
		}
	}
	return b.String()
}
