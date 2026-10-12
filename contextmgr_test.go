package react

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"
)

func long(s string) string { return strings.Repeat(s, 400) } // ~100 tokens

func history() []Message {
	return []Message{
		UserText(long("a")),
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "t", Arguments: []byte(`{}`)}}},
		ToolResultMessage("1", "t", []Part{TextPart(long("r"))}, false),
		AssistantText(long("b")),
		UserText(long("c")),
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "2", Name: "t", Arguments: []byte(`{}`)}}},
		ToolResultMessage("2", "t", []Part{TextPart(long("r"))}, false),
		AssistantText(long("d")),
		UserText(long("e")),
	}
}

func validate(t *testing.T, out []Message) {
	t.Helper()
	if len(out) == 0 || out[0].Role != RoleUser {
		t.Fatalf("history must start with user: %+v", out)
	}
	for i, m := range out {
		if m.Role == RoleTool {
			prev := out[i-1]
			if prev.Role != RoleTool && len(prev.ToolCalls) == 0 {
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
	all, _ := ContextWindow{}.Fit(context.Background(), "", msgs, 100000)
	if len(all) != len(msgs) {
		t.Fatal("dropped messages despite budget")
	}
	for _, budget := range []int{50, 250, 450, 600} {
		out, _ := ContextWindow{}.Fit(context.Background(), "", msgs, budget)
		validate(t, out)
	}
	out, _ := ContextWindow{}.Fit(context.Background(), "", msgs, 450)
	if len(out) != 5 {
		t.Fatalf("expected trim to second turn, got %d msgs", len(out))
	}
}

type testSummarizeProvider struct {
	calls int
	text  string
}

func (p *testSummarizeProvider) Name() string { return "test-summarize" }
func (p *testSummarizeProvider) Generate(ctx context.Context, req *Request) (*Response, error) {
	p.calls++
	if p.calls > 1 {
		return nil, errors.New("exhausted")
	}
	return &Response{Message: AssistantText(p.text)}, nil
}
func (p *testSummarizeProvider) Stream(ctx context.Context, req *Request) iter.Seq2[StreamEvent, error] {
	return StreamResponse(&Response{Message: AssistantText(p.text)}, 0)
}

func TestSummarize(t *testing.T) {
	p := &testSummarizeProvider{text: "SUMMARY"}
	s := &ContextSummarizer{Provider: p, Model: "m"}
	out, err := s.Fit(context.Background(), "", history(), 500)
	if err != nil {
		t.Fatal(err)
	}
	validate(t, out)
	if !strings.Contains(out[0].Text(), "SUMMARY") {
		t.Fatalf("no summary: %q", out[0].Text())
	}
	// Second call hits the cache (the provider would return exhausted otherwise).
	if _, err := s.Fit(context.Background(), "", history(), 500); err != nil {
		t.Fatal(err)
	}
	if p.calls != 1 {
		t.Fatalf("summarizer called %d times", p.calls)
	}
}
