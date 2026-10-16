package tests

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
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

func TestToolFailureBatchAndSinglePrefix(t *testing.T) {
	ctx := context.Background()
	failTool := react.NewTool("fail_tool", "fails", func(ctx context.Context, in struct{}) (string, error) {
		return "", errors.New("underlying_failure")
	})

	p := mock.New(
		mock.ToolCalls("calling...", mock.Call("fail_tool", `{}`)),
		mock.Text("done"),
	)

	var batchErrSeen error
	var batchResultText string
	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(failTool),
		react.WithHooks(react.Hooks{
			AfterToolBatch: func(ctx context.Context, rc *react.RunContext, results []react.ToolCallResult) error {
				if len(results) > 0 {
					batchErrSeen = results[0].Err
					batchResultText = results[0].Result.Text()
				}
				return nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(ctx, "fail-sess", "run fail")
	if err != nil {
		t.Fatal(err)
	}
	if batchErrSeen == nil || batchErrSeen.Error() != "underlying_failure" {
		t.Fatalf("expected batchErr 'underlying_failure', got: %v", batchErrSeen)
	}
	// Result text should have exactly one "Error: " prefix
	if strings.Count(batchResultText, "Error: ") != 1 {
		t.Fatalf("expected single 'Error: ' prefix, got: %q", batchResultText)
	}

	for _, m := range res.Messages {
		if m.Role == react.RoleTool {
			if strings.Count(m.Text(), "Error: ") != 1 {
				t.Fatalf("expected single 'Error: ' in tool message, got: %q", m.Text())
			}
		}
	}
}

func TestMatchRulesCommandChainingBlocked(t *testing.T) {
	bashSpec := react.ToolSpec{
		Name:        "Bash",
		Description: "Bash tool",
		Annotations: react.ToolAnnotations{Destructive: true},
	}

	rules := react.PermissionRules{
		Allow: []string{"Bash(git:*)"},
	}

	ctx := context.Background()

	// Safe git command should be allowed
	safeCall := react.ToolCall{
		Name:      "Bash",
		Arguments: json.RawMessage(`{"command":"git status"}`),
	}
	if dec := rules.Check(ctx, bashSpec, safeCall); dec != react.PolicyAllow {
		t.Fatalf("expected safe git command to be PolicyAllow, got %v", dec)
	}

	// Command chaining with semicolon must NOT be auto-allowed
	semiCall := react.ToolCall{
		Name:      "Bash",
		Arguments: json.RawMessage(`{"command":"git status; rm -rf /"}`),
	}
	if dec := rules.Check(ctx, bashSpec, semiCall); dec == react.PolicyAllow {
		t.Fatalf("expected semicolon command chaining to be blocked from PolicyAllow, got %v", dec)
	}

	// Command chaining with && must NOT be auto-allowed
	andCall := react.ToolCall{
		Name:      "Bash",
		Arguments: json.RawMessage(`{"command":"git status && evil"}`),
	}
	if dec := rules.Check(ctx, bashSpec, andCall); dec == react.PolicyAllow {
		t.Fatalf("expected && command chaining to be blocked from PolicyAllow, got %v", dec)
	}

	// Command substitution with backticks must NOT be auto-allowed
	btCall := react.ToolCall{
		Name:      "Bash",
		Arguments: json.RawMessage(`{"command":"git ` + "`rm -rf /`" + `"}`),
	}
	if dec := rules.Check(ctx, bashSpec, btCall); dec == react.PolicyAllow {
		t.Fatalf("expected backtick command substitution to be blocked from PolicyAllow, got %v", dec)
	}
}

func TestFileStoreAppendCorruptJSON(t *testing.T) {
	tmp := t.TempDir()
	fs, err := react.NewFileStore(tmp)
	if err != nil {
		t.Fatal(err)
	}

	// Write corrupt JSON to session file
	sessionID := "corrupt-test"
	p := filepath.Join(tmp, "636f72727570742d74657374.json") // hex of "corrupt-test"
	if err := os.WriteFile(p, []byte("NOT_JSON{{{"), 0600); err != nil {
		t.Fatal(err)
	}

	// Append should return an error, not overwrite
	err = fs.Append(context.Background(), sessionID, react.UserText("hello"))
	if err == nil {
		t.Fatal("expected error when appending to corrupt session file")
	}
	if !strings.Contains(err.Error(), "failed to parse session file") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestStreamResponseReasoningSignature(t *testing.T) {
	resp := &react.Response{
		Message: react.Message{
			Role:               react.RoleAssistant,
			Content:            []react.Part{react.TextPart("answer")},
			Reasoning:          "thinking...",
			ReasoningSignature: "crypto_sig_999",
		},
		Reasoning:  "thinking...",
		StopReason: react.StopEnd,
	}

	var acc react.Accumulator
	for ev, err := range react.StreamResponse(resp, 4) {
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(ev)
	}

	accumResp := acc.Response()
	if accumResp.Message.ReasoningSignature != "crypto_sig_999" {
		t.Fatalf("expected ReasoningSignature 'crypto_sig_999', got %q", accumResp.Message.ReasoningSignature)
	}
}
