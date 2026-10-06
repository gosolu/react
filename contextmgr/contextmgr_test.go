package contextmgr

import (
	"context"
	"strings"
	"testing"

	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/llm/mock"
)

func long(s string) string { return strings.Repeat(s, 400) } // ~100 tokens

func history() []llm.Message {
	return []llm.Message{
		llm.UserText(long("a")),
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "1", Name: "t", Arguments: []byte(`{}`)}}},
		llm.ToolResult("1", "t", []llm.Part{llm.TextPart(long("r"))}, false),
		llm.AssistantText(long("b")),
		llm.UserText(long("c")),
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "2", Name: "t", Arguments: []byte(`{}`)}}},
		llm.ToolResult("2", "t", []llm.Part{llm.TextPart(long("r"))}, false),
		llm.AssistantText(long("d")),
		llm.UserText(long("e")),
	}
}

func validate(t *testing.T, out []llm.Message) {
	t.Helper()
	if len(out) == 0 || out[0].Role != llm.RoleUser {
		t.Fatalf("history must start with user: %+v", out)
	}
	for i, m := range out {
		if m.Role == llm.RoleTool {
			prev := out[i-1]
			if prev.Role != llm.RoleTool && len(prev.ToolCalls) == 0 {
				t.Fatalf("orphan tool result at %d", i)
			}
		}
	}
	if out[len(out)-1].Text() != long("e") {
		t.Fatal("latest user message dropped")
	}
}

func TestWindow(t *testing.T) {
	msgs := history()
	all, _ := Window{}.Fit(context.Background(), "", msgs, 100000)
	if len(all) != len(msgs) {
		t.Fatal("dropped messages despite budget")
	}
	for _, budget := range []int{50, 250, 450, 600} {
		out, _ := Window{}.Fit(context.Background(), "", msgs, budget)
		validate(t, out)
	}
	out, _ := Window{}.Fit(context.Background(), "", msgs, 450)
	if len(out) != 5 {
		t.Fatalf("expected trim to second turn, got %d msgs", len(out))
	}
}

func TestSummarize(t *testing.T) {
	p := mock.New(mock.Text("SUMMARY"))
	s := &Summarize{Provider: p, Model: "m"}
	out, err := s.Fit(context.Background(), "", history(), 500)
	if err != nil {
		t.Fatal(err)
	}
	validate(t, out)
	if !strings.Contains(out[0].Text(), "SUMMARY") {
		t.Fatalf("no summary: %q", out[0].Text())
	}
	// Second call hits the cache (the mock would return ErrExhausted otherwise).
	if _, err := s.Fit(context.Background(), "", history(), 500); err != nil {
		t.Fatal(err)
	}
	if len(p.Requests()) != 1 {
		t.Fatalf("summarizer called %d times", len(p.Requests()))
	}
}
