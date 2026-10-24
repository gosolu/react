package tests

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider/mock"
)

func noopTool(name string) react.Tool {
	return react.NewTool(name, "does nothing", func(context.Context, struct{}) (string, error) {
		return "ok", nil
	})
}

func TestBudgetModelCalls(t *testing.T) {
	p := mock.New(
		mock.ToolCalls("thinking", mock.Call("noop", `{}`)),
		mock.Text("done"),
	)
	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(noopTool("noop")),
		react.WithBudget(react.BudgetLimits{MaxModelCalls: 1}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "", "hi")
	if !errors.Is(err, react.ErrBudgetExceeded) {
		t.Fatalf("err = %v, want ErrBudgetExceeded", err)
	}
	if res == nil {
		t.Fatal("expected a partial result")
	}
	if res.StopReason != react.StopBudgetExceeded {
		t.Fatalf("stop reason = %q", res.StopReason)
	}
	if res.Budget.Exceeded != react.BudgetModelCalls {
		t.Fatalf("exceeded = %q", res.Budget.Exceeded)
	}
	if res.Budget.ModelCalls != 1 {
		t.Fatalf("model calls = %d", res.Budget.ModelCalls)
	}
	if got := len(p.Requests()); got != 1 {
		t.Fatalf("provider calls = %d, want 1", got)
	}
}

func TestBudgetToolCalls(t *testing.T) {
	p := mock.New(mock.ToolCalls("", mock.Call("noop", `{}`), mock.Call("noop2", `{}`)))
	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(noopTool("noop"), noopTool("noop2")),
		react.WithBudget(react.BudgetLimits{MaxToolCalls: 1}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "", "hi")
	if !errors.Is(err, react.ErrBudgetExceeded) {
		t.Fatalf("err = %v, want ErrBudgetExceeded", err)
	}
	if res.Budget.Exceeded != react.BudgetToolCalls {
		t.Fatalf("exceeded = %q", res.Budget.Exceeded)
	}
	if res.Budget.ToolCalls != 0 {
		t.Fatalf("tool calls = %d, want 0 (batch must not run)", res.Budget.ToolCalls)
	}
	if got := len(p.Requests()); got != 1 {
		t.Fatalf("provider calls = %d, want 1", got)
	}

	var toolMsgs int
	for _, m := range res.Messages {
		if m.Role != react.RoleTool {
			continue
		}
		toolMsgs++
		if !m.IsError || !strings.Contains(m.Text(), "budget") {
			t.Fatalf("expected an error tool result, got %q", m.Text())
		}
	}
	if toolMsgs != 2 {
		t.Fatalf("tool messages = %d, want 2", toolMsgs)
	}
}

func TestBudgetTotalTokens(t *testing.T) {
	p := mock.New(mock.Text("hello"))
	a, err := react.New(
		react.WithProvider(p),
		react.WithBudget(react.BudgetLimits{MaxTotalTokens: 11}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "", "hi")
	if !errors.Is(err, react.ErrBudgetExceeded) {
		t.Fatalf("err = %v, want ErrBudgetExceeded", err)
	}
	if res.Budget.Exceeded != react.BudgetTotalTokens {
		t.Fatalf("exceeded = %q", res.Budget.Exceeded)
	}
	if res.Output != "hello" {
		t.Fatalf("output = %q, want the partial answer", res.Output)
	}
	if res.Usage.Total() <= 11 {
		t.Fatalf("usage = %d, want over the limit", res.Usage.Total())
	}
}

func TestBudgetRequestTokens(t *testing.T) {
	p := mock.New(mock.Text("hello"))
	a, err := react.New(
		react.WithProvider(p),
		react.WithBudget(react.BudgetLimits{MaxRequestTokens: 1}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "", "hi")
	if !errors.Is(err, react.ErrBudgetExceeded) {
		t.Fatalf("err = %v, want ErrBudgetExceeded", err)
	}
	if res.Budget.Exceeded != react.BudgetRequestTokens {
		t.Fatalf("exceeded = %q", res.Budget.Exceeded)
	}
	if got := len(p.Requests()); got != 0 {
		t.Fatalf("provider calls = %d, want 0", got)
	}
}

func TestBudgetWallTime(t *testing.T) {
	p := mock.NewFunc(func(ctx context.Context, _ *react.Request) (*react.Response, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	a, err := react.New(
		react.WithProvider(p),
		react.WithBudget(react.BudgetLimits{MaxWallTime: 20 * time.Millisecond}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "", "hi")
	if !errors.Is(err, react.ErrBudgetExceeded) {
		t.Fatalf("err = %v, want ErrBudgetExceeded", err)
	}
	if res.StopReason != react.StopBudgetExceeded {
		t.Fatalf("stop reason = %q", res.StopReason)
	}
	if res.Budget.Exceeded != react.BudgetWallTime {
		t.Fatalf("exceeded = %q", res.Budget.Exceeded)
	}
}

func TestModelTimeout(t *testing.T) {
	p := mock.NewFunc(func(ctx context.Context, _ *react.Request) (*react.Response, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	a, err := react.New(
		react.WithProvider(p),
		react.WithModelTimeout(20*time.Millisecond),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "", "hi")
	if !errors.Is(err, react.ErrModelTimeout) {
		t.Fatalf("err = %v, want ErrModelTimeout", err)
	}
	if res.StopReason != react.StopTimeout {
		t.Fatalf("stop reason = %q", res.StopReason)
	}
	if got := len(p.Requests()); got != 1 {
		t.Fatalf("provider calls = %d, want 1", got)
	}
}

func TestSubagentUsageAggregated(t *testing.T) {
	childProvider := mock.New(mock.Text("child"))
	child, err := react.New(react.WithProvider(childProvider), react.WithName("child"))
	if err != nil {
		t.Fatal(err)
	}

	parentProvider := mock.New(
		mock.ToolCalls("delegating", mock.Call("delegate", map[string]string{"task": "do work"})),
		mock.Text("parent"),
	)
	parent, err := react.New(
		react.WithProvider(parentProvider),
		react.WithName("parent"),
		react.WithTools(child.AsTool("delegate", "delegate work")),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := parent.Run(context.Background(), "", "hi")
	if err != nil {
		t.Fatal(err)
	}
	if res.Usage.InputTokens != 30 || res.Usage.OutputTokens != 9 {
		t.Fatalf("usage = %+v, want input 30 output 9", res.Usage)
	}
	if res.Budget.ModelCalls != 3 {
		t.Fatalf("model calls = %d, want 3", res.Budget.ModelCalls)
	}
}

func TestParentBudgetCapsSubagent(t *testing.T) {
	childProvider := mock.New(mock.Text("child"))
	child, err := react.New(react.WithProvider(childProvider), react.WithName("child"))
	if err != nil {
		t.Fatal(err)
	}
	parentProvider := mock.New(
		mock.ToolCalls("delegating", mock.Call("delegate", map[string]string{"task": "do work"})),
		mock.Text("parent"),
	)
	parent, err := react.New(
		react.WithProvider(parentProvider),
		react.WithName("parent"),
		react.WithTools(child.AsTool("delegate", "delegate work")),
		react.WithBudget(react.BudgetLimits{MaxModelCalls: 1}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := parent.Run(context.Background(), "", "hi")
	if !errors.Is(err, react.ErrBudgetExceeded) {
		t.Fatalf("err = %v, want ErrBudgetExceeded", err)
	}
	if res.StopReason != react.StopBudgetExceeded {
		t.Fatalf("stop reason = %q", res.StopReason)
	}
	if res.Budget.ModelCalls != 1 {
		t.Fatalf("model calls = %d, want 1", res.Budget.ModelCalls)
	}
	if got := len(parentProvider.Requests()); got != 1 {
		t.Fatalf("parent provider calls = %d, want 1", got)
	}
}

func TestBudgetZeroIsUnlimited(t *testing.T) {
	p := mock.New(mock.Text("done"))
	a, err := react.New(react.WithProvider(p), react.WithBudget(react.BudgetLimits{}))
	if err != nil {
		t.Fatal(err)
	}
	res, err := a.Run(context.Background(), "", "hi")
	if err != nil {
		t.Fatal(err)
	}
	if res.Budget.Exceeded != "" {
		t.Fatalf("exceeded = %q", res.Budget.Exceeded)
	}
	if res.StopReason != react.StopEnd {
		t.Fatalf("stop reason = %q", res.StopReason)
	}
}

type badSchemaTool struct{}

func (badSchemaTool) Spec() react.ToolSpec { return react.ToolSpec{Name: "bad"} }
func (badSchemaTool) Call(context.Context, json.RawMessage) (react.ToolResult, error) {
	return react.TextResult(""), nil
}

func TestValidateRejectsBadConfig(t *testing.T) {
	p := mock.New(mock.Text("done"))

	if _, err := react.New(react.WithProvider(p), react.WithTools(badSchemaTool{})); err == nil {
		t.Fatal("expected a missing-schema validation error")
	}
	if _, err := react.New(react.WithProvider(p), react.WithBudget(react.BudgetLimits{MaxModelCalls: -1})); err == nil {
		t.Fatal("expected a negative-budget validation error")
	}
	if _, err := react.New(react.WithProvider(p), react.WithMaxTurns(0)); err == nil {
		t.Fatal("expected a max-turns validation error")
	}
	contradictoryTool := react.NewTool("contradictory", "both destructive and readonly", func(context.Context, struct{}) (string, error) {
		return "ok", nil
	}, react.Destructive(), react.ReadOnly())
	if _, err := react.New(react.WithProvider(p), react.WithTools(contradictoryTool)); err == nil {
		t.Fatal("expected a validation error for tool marked both Destructive and ReadOnly")
	}

	a, err := react.New(react.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Validate(); err != nil {
		t.Fatalf("valid agent failed validation: %v", err)
	}
}

func TestBudgetUsageAccounting(t *testing.T) {
	p := mock.New(
		mock.ToolCalls("", mock.Call("noop", `{}`)),
		mock.Text("done"),
	)
	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(noopTool("noop")),
		react.WithBudget(react.BudgetLimits{MaxModelCalls: 10, MaxToolCalls: 10}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "", "hi")
	if err != nil {
		t.Fatal(err)
	}
	if res.Budget.ModelCalls != 2 || res.Budget.ToolCalls != 1 {
		t.Fatalf("budget = %+v, want 2 model and 1 tool call", res.Budget)
	}
	if res.Budget.InputTokens != res.Usage.InputTokens || res.Budget.OutputTokens != res.Usage.OutputTokens {
		t.Fatalf("budget tokens %+v do not match usage %+v", res.Budget, res.Usage)
	}
	if res.Budget.Wall <= 0 {
		t.Fatalf("wall = %s", res.Budget.Wall)
	}
	if res.Budget.Exceeded != "" {
		t.Fatalf("exceeded = %q", res.Budget.Exceeded)
	}
}

func TestSubagentBudgetExceededDoesNotPoisonParent(t *testing.T) {
	// Child sub-agent has a tight limit of 1 model call, but tries to make 2 calls.
	childProvider := mock.New(
		mock.ToolCalls("child step 1", mock.Call("noop", `{}`)),
		mock.Text("child step 2"),
	)
	child, err := react.New(
		react.WithProvider(childProvider),
		react.WithName("child"),
		react.WithTools(noopTool("noop")),
		react.WithBudget(react.BudgetLimits{MaxModelCalls: 1}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Parent has plenty of budget (MaxModelCalls: 10) and should continue after child hits its local limit.
	parentProvider := mock.New(
		mock.ToolCalls("delegating to child", mock.Call("delegate", map[string]string{"task": "explore"})),
		mock.Text("parent completed despite child budget limit"),
	)
	parent, err := react.New(
		react.WithProvider(parentProvider),
		react.WithName("parent"),
		react.WithTools(child.AsTool("delegate", "delegate task")),
		react.WithBudget(react.BudgetLimits{MaxModelCalls: 10}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := parent.Run(context.Background(), "", "start")
	if err != nil {
		t.Fatalf("expected parent to succeed after child hit local budget, got err: %v", err)
	}
	if res.StopReason != react.StopEnd {
		t.Fatalf("expected StopEnd, got %q", res.StopReason)
	}
	if res.Budget.Exceeded != "" {
		t.Fatalf("expected parent Budget.Exceeded to be empty, got %q", res.Budget.Exceeded)
	}
	if res.Output != "parent completed despite child budget limit" {
		t.Fatalf("unexpected output: %q", res.Output)
	}
	// Parent made 2 model calls + child made 1 model call = 3 total
	if res.Budget.ModelCalls != 3 {
		t.Fatalf("expected 3 total model calls across tree, got %d", res.Budget.ModelCalls)
	}
}

func TestConcurrentSubagentsBudgetIsolation(t *testing.T) {
	// Subagent A has MaxModelCalls: 2 and makes 2 model calls.
	pA := mock.New(
		mock.ToolCalls("A1", mock.Call("noop", `{}`)),
		mock.Text("A2 done"),
	)
	subA, err := react.New(
		react.WithProvider(pA),
		react.WithName("subA"),
		react.WithTools(noopTool("noop")),
		react.WithBudget(react.BudgetLimits{MaxModelCalls: 2}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Subagent B has MaxModelCalls: 3 and makes 3 model calls.
	pB := mock.New(
		mock.ToolCalls("B1", mock.Call("noop", `{}`)),
		mock.ToolCalls("B2", mock.Call("noop", `{}`)),
		mock.Text("B3 done"),
	)
	subB, err := react.New(
		react.WithProvider(pB),
		react.WithName("subB"),
		react.WithTools(noopTool("noop")),
		react.WithBudget(react.BudgetLimits{MaxModelCalls: 3}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Parent invokes subA and subB concurrently in a single turn.
	parentProvider := mock.New(
		mock.ToolCalls("parallel delegation",
			mock.Call("call_a", map[string]string{"task": "work A"}),
			mock.Call("call_b", map[string]string{"task": "work B"}),
		),
		mock.Text("both subagents finished"),
	)
	parent, err := react.New(
		react.WithProvider(parentProvider),
		react.WithName("parent"),
		react.WithMaxParallelTools(4),
		react.WithTools(
			subA.AsTool("call_a", "run subA"),
			subB.AsTool("call_b", "run subB"),
		),
		react.WithBudget(react.BudgetLimits{MaxModelCalls: 10}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := parent.Run(context.Background(), "", "go")
	if err != nil {
		t.Fatalf("parallel subagents failed: %v", err)
	}
	for _, m := range res.Messages {
		if m.Role == react.RoleTool && m.IsError {
			t.Fatalf("subagent tool unexpectedly failed: %s", m.Text())
		}
	}
	// Total model calls = 2 (parent) + 2 (subA) + 3 (subB) = 7
	if res.Budget.ModelCalls != 7 {
		t.Fatalf("expected 7 total model calls, got %d", res.Budget.ModelCalls)
	}
}

func TestContextSummarizerCountsAgainstBudget(t *testing.T) {
	sumProvider := mock.New(mock.Text("short summary"))
	summarizer := &react.ContextSummarizer{
		Provider: sumProvider,
	}
	mainProvider := mock.New(mock.Text("final response"))

	// Budget allows only 1 model call; summarization + main call requires 2 model calls.
	a, err := react.New(
		react.WithProvider(mainProvider),
		react.WithContextManager(summarizer, 20),
		react.WithBudget(react.BudgetLimits{MaxModelCalls: 1}),
	)
	if err != nil {
		t.Fatal(err)
	}

	history := []react.Message{
		react.UserText(strings.Repeat("earlier message content ", 20)),
		react.AssistantText(strings.Repeat("earlier assistant reply ", 20)),
	}
	res, err := a.Run(context.Background(), "", "new question", react.WithHistory(history))
	if !errors.Is(err, react.ErrBudgetExceeded) {
		t.Fatalf("expected ErrBudgetExceeded when summarizer + main model call exceed MaxModelCalls=1, got %v", err)
	}
	if res == nil || res.StopReason != react.StopBudgetExceeded {
		t.Fatalf("expected StopBudgetExceeded, got %+v", res)
	}
}
