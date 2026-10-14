package tests

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider/mock"
)

func TestSessionRememberApprover(t *testing.T) {
	ctx := context.Background()
	var callCount int32

	base := react.PermissionApproverFunc(func(ctx context.Context, req react.ApprovalRequest) (bool, error) {
		atomic.AddInt32(&callCount, 1)
		return true, nil
	})

	remember := react.NewSessionRememberApprover(base)
	req := react.ApprovalRequest{
		Call: react.ToolCall{Name: "rm_rf"},
	}

	// 1. First call - goes to base approver
	ok, err := remember.Approve(ctx, req)
	if err != nil || !ok {
		t.Fatalf("expected approved: %v, %v", ok, err)
	}
	if atomic.LoadInt32(&callCount) != 1 {
		t.Fatalf("expected callCount 1, got %d", atomic.LoadInt32(&callCount))
	}

	// Remember "rm_rf"
	remember.Remember("rm_rf")

	// 2. Second call for "rm_rf" - should bypass base approver
	ok, err = remember.Approve(ctx, req)
	if err != nil || !ok {
		t.Fatalf("expected approved: %v, %v", ok, err)
	}
	if atomic.LoadInt32(&callCount) != 1 {
		t.Fatalf("expected callCount still 1, got %d", atomic.LoadInt32(&callCount))
	}

	// 3. Different tool - should invoke base approver
	reqOther := react.ApprovalRequest{
		Call: react.ToolCall{Name: "format"},
	}
	ok, err = remember.Approve(ctx, reqOther)
	if err != nil || !ok {
		t.Fatalf("expected approved: %v, %v", ok, err)
	}
	if atomic.LoadInt32(&callCount) != 2 {
		t.Fatalf("expected callCount 2, got %d", atomic.LoadInt32(&callCount))
	}
}

func TestWithMaxToolOutput(t *testing.T) {
	ctx := context.Background()

	bigTool := react.NewTool("generate_big", "generates big output", func(ctx context.Context, in struct{}) (string, error) {
		return strings.Repeat("A", 100), nil
	})

	p := mock.New(
		mock.ToolCalls("calling...", mock.Call("generate_big", `{}`)),
		mock.Text("done"),
	)

	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(bigTool),
		react.WithMaxToolOutput(30),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(ctx, "trunc-sess", "run big")
	if err != nil {
		t.Fatal(err)
	}

	// Check messages for truncation marker
	var foundTruncated bool
	for _, m := range res.Messages {
		if m.Role == react.RoleTool {
			txt := m.Text()
			if strings.Contains(txt, "...[output truncated from 100 bytes]") && strings.HasPrefix(txt, strings.Repeat("A", 30)) {
				foundTruncated = true
				break
			}
		}
	}
	if !foundTruncated {
		t.Fatalf("expected tool message to be truncated, messages: %+v", res.Messages)
	}
}

func TestWithRunTools(t *testing.T) {
	ctx := context.Background()

	runOnlyTool := react.NewTool("run_only_tool", "available only in this run", func(ctx context.Context, in struct{}) (string, error) {
		return "run_only_ok", nil
	})

	p := mock.New(
		mock.ToolCalls("invoking...", mock.Call("run_only_tool", `{}`)),
		mock.Text("completed"),
	)

	a, err := react.New(react.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}

	hasTool := func() bool {
		for _, s := range a.Tools() {
			if s.Name == "run_only_tool" {
				return true
			}
		}
		return false
	}

	// Permanent agent should NOT have run_only_tool
	if hasTool() {
		t.Fatal("agent should not have run_only_tool registered permanently")
	}

	// Execute with run-level tool option
	res, err := a.Run(ctx, "run-tool-sess", "call run only tool", react.WithRunTools(runOnlyTool))
	if err != nil {
		t.Fatalf("Run with WithRunTools failed: %v", err)
	}
	if res.Output != "completed" {
		t.Fatalf("unexpected output: %s", res.Output)
	}

	// Still not in agent registry
	if hasTool() {
		t.Fatal("agent permanently modified by WithRunTools")
	}
}
