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
