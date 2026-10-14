package tests

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider/mock"
)

func TestBeforePromptHook(t *testing.T) {
	t.Run("deny prompt", func(t *testing.T) {
		p := mock.New(mock.Text("should not reach"))
		a, err := react.New(react.WithProvider(p), react.WithHooks(react.Hooks{
			BeforePrompt: func(ctx context.Context, rc *react.RunContext, prompt string) (react.HookPromptDecision, error) {
				if prompt == "unsafe input" {
					return react.HookPromptDecision{Action: react.HookDeny, Reason: "disallowed prompt"}, nil
				}
				return react.HookPromptDecision{Action: react.HookContinue}, nil
			},
		}))
		if err != nil {
			t.Fatal(err)
		}

		_, err = a.Run(context.Background(), "", "unsafe input")
		if err == nil || err.Error() != "prompt blocked: disallowed prompt" {
			t.Fatalf("expected prompt blocked, got: %v", err)
		}
	})

	t.Run("rewrite prompt and inject context", func(t *testing.T) {
		var capturedPrompt string
		p := mock.NewFunc(func(ctx context.Context, req *react.Request) (*react.Response, error) {
			capturedPrompt = req.Messages[0].Text()
			// Second message should be system reminder
			if len(req.Messages) < 2 || req.Messages[1].Text() != "<system-reminder>\nuser is admin\n</system-reminder>" {
				t.Fatalf("unexpected messages: %v", req.Messages)
			}
			return &react.Response{Message: react.AssistantText("done")}, nil
		})

		a, err := react.New(react.WithProvider(p), react.WithHooks(react.Hooks{
			BeforePrompt: func(ctx context.Context, rc *react.RunContext, prompt string) (react.HookPromptDecision, error) {
				return react.HookPromptDecision{
					Action:            react.HookContinue,
					UpdatedPrompt:     "transformed: " + prompt,
					AdditionalContext: "user is admin",
				}, nil
			},
		}))
		if err != nil {
			t.Fatal(err)
		}

		res, err := a.Run(context.Background(), "", "original")
		if err != nil {
			t.Fatal(err)
		}
		if res.Output != "done" {
			t.Fatalf("unexpected output: %s", res.Output)
		}
		if capturedPrompt != "transformed: original" {
			t.Fatalf("unexpected captured prompt: %s", capturedPrompt)
		}
	})
}

func TestBeforeStopHook(t *testing.T) {
	var stopCalls int
	p := mock.New(
		mock.Text("first attempt"),
		mock.Text("final verified answer"),
	)

	a, err := react.New(react.WithProvider(p), react.WithHooks(react.Hooks{
		BeforeStop: func(ctx context.Context, rc *react.RunContext, output string) (react.HookDecision, error) {
			stopCalls++
			if stopCalls == 1 {
				// Prevent stopping on turn 1
				return react.HookDecision{
					Action: react.HookContinueWorking,
					Reason: "Please run tests before finishing.",
				}, nil
			}
			return react.HookDecision{Action: react.HookContinue}, nil
		},
	}))
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "", "do work")
	if err != nil {
		t.Fatal(err)
	}
	if stopCalls != 2 {
		t.Fatalf("expected 2 stop hook calls, got %d", stopCalls)
	}
	if res.Output != "final verified answer" {
		t.Fatalf("unexpected final output: %s", res.Output)
	}
	if res.Turns != 2 {
		t.Fatalf("expected 2 turns, got %d", res.Turns)
	}
}

func TestToolHooksAndMatchers(t *testing.T) {
	toolExecuted := false
	testTool := react.NewTool("test_tool", "test", func(ctx context.Context, in map[string]string) (string, error) {
		toolExecuted = true
		return "tool output: " + in["cmd"], nil
	})

	var (
		matchedCalls    int
		afterBatchCalls int
		updatedArgsSeen string
	)

	p := mock.New(
		mock.ToolCalls("calling...", mock.Call("test_tool", map[string]string{"cmd": "original"})),
		mock.Text("finished"),
	)

	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(testTool),
		react.WithHooks(react.Hooks{
			BeforeTool: react.MatchTool("test_*", func(ctx context.Context, rc *react.RunContext, call *react.ToolCall) (react.HookDecision, error) {
				matchedCalls++
				return react.HookDecision{
					Action:            react.HookContinue,
					UpdatedArguments:  json.RawMessage(`{"cmd":"updated_by_hook"}`),
					AdditionalContext: "tool executed safely",
				}, nil
			}),
			AfterTool: react.MatchToolAfter("test_tool", func(ctx context.Context, rc *react.RunContext, call react.ToolCall, res *react.ToolResult) error {
				var parsed map[string]string
				_ = json.Unmarshal(call.Arguments, &parsed)
				updatedArgsSeen = parsed["cmd"]
				return nil
			}),
			AfterToolBatch: func(ctx context.Context, rc *react.RunContext, results []react.ToolCallResult) error {
				afterBatchCalls++
				if len(results) != 1 || results[0].Call.Name != "test_tool" {
					t.Fatalf("unexpected batch results: %v", results)
				}
				return nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "", "run tool")
	if err != nil {
		t.Fatal(err)
	}
	if !toolExecuted {
		t.Fatal("expected tool to execute")
	}
	if matchedCalls != 1 {
		t.Fatalf("expected 1 matched BeforeTool call, got %d", matchedCalls)
	}
	if updatedArgsSeen != "updated_by_hook" {
		t.Fatalf("expected updated args 'updated_by_hook', got %q", updatedArgsSeen)
	}
	if afterBatchCalls != 1 {
		t.Fatalf("expected 1 AfterToolBatch call, got %d", afterBatchCalls)
	}
	if res.Output != "finished" {
		t.Fatalf("unexpected output: %s", res.Output)
	}
}

func TestToolFailureHook(t *testing.T) {
	var failureReported error
	failingTool := react.NewTool("fail_tool", "failing tool", func(ctx context.Context, in any) (string, error) {
		return "", errors.New("boom")
	})

	p := mock.New(
		mock.ToolCalls("failing...", mock.Call("fail_tool", nil)),
		mock.Text("handled error"),
	)

	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(failingTool),
		react.WithHooks(react.Hooks{
			OnToolFailure: func(ctx context.Context, rc *react.RunContext, call react.ToolCall, err error) error {
				failureReported = err
				return nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	_, err = a.Run(context.Background(), "", "test failure")
	if err != nil {
		t.Fatal(err)
	}
	if failureReported == nil || failureReported.Error() != "boom" {
		t.Fatalf("unexpected failure reported: %v", failureReported)
	}
}

func TestPermissionHooks(t *testing.T) {
	t.Run("auto approve by hook", func(t *testing.T) {
		var permissionRequestSeen bool
		cmdTool := react.NewTool("cmd", "cmd", func(ctx context.Context, in map[string]string) (string, error) {
			return "executed: " + in["c"], nil
		})

		p := mock.New(
			mock.ToolCalls("running...", mock.Call("cmd", map[string]string{"c": "ls"})),
			mock.Text("ok"),
		)

		a, err := react.New(
			react.WithProvider(p),
			react.WithTools(cmdTool),
			react.WithPolicy(react.PermissionPolicyFunc(func(ctx context.Context, spec react.ToolSpec, call react.ToolCall) react.PolicyDecision {
				return react.PolicyAsk
			})),
			react.WithHooks(react.Hooks{
				OnPermissionRequest: func(ctx context.Context, rc *react.RunContext, req react.ApprovalRequest) (react.HookApprovalDecision, error) {
					permissionRequestSeen = true
					return react.HookApprovalDecision{
						Handled:  true,
						Approved: true,
					}, nil
				},
			}),
		)
		if err != nil {
			t.Fatal(err)
		}

		res, err := a.Run(context.Background(), "", "run")
		if err != nil {
			t.Fatal(err)
		}
		if !permissionRequestSeen {
			t.Fatal("expected OnPermissionRequest to be called")
		}
		if res.Output != "ok" {
			t.Fatalf("unexpected output: %s", res.Output)
		}
	})

	t.Run("permission denied hook", func(t *testing.T) {
		var deniedReason string
		cmdTool := react.NewTool("cmd", "cmd", func(ctx context.Context, in map[string]string) (string, error) {
			return "executed", nil
		})

		p := mock.New(
			mock.ToolCalls("denying...", mock.Call("cmd", map[string]string{"c": "rm"})),
			mock.Text("denied handled"),
		)

		a, err := react.New(
			react.WithProvider(p),
			react.WithTools(cmdTool),
			react.WithPolicy(react.PermissionPolicyFunc(func(ctx context.Context, spec react.ToolSpec, call react.ToolCall) react.PolicyDecision {
				return react.PolicyDeny
			})),
			react.WithHooks(react.Hooks{
				OnPermissionDenied: func(ctx context.Context, rc *react.RunContext, call react.ToolCall, reason string) {
					deniedReason = reason
				},
			}),
		)
		if err != nil {
			t.Fatal(err)
		}

		res, err := a.Run(context.Background(), "", "run")
		if err != nil {
			t.Fatal(err)
		}
		if deniedReason != "denied by policy" {
			t.Fatalf("unexpected denied reason: %s", deniedReason)
		}
		if res.Output != "denied handled" {
			t.Fatalf("unexpected output: %s", res.Output)
		}
	})
}

func TestSubagentHooks(t *testing.T) {
	subP := mock.New(mock.Text("subagent finished"))
	subAgent, err := react.New(react.WithProvider(subP), react.WithName("worker"))
	if err != nil {
		t.Fatal(err)
	}

	var (
		startedAgent string
		startedTask  string
		stoppedAgent string
		stoppedOut   string
	)

	parentP := mock.New(
		mock.ToolCalls("delegating...", mock.Call("delegate_worker", map[string]string{"task": "do research"})),
		mock.Text("parent finished"),
	)

	parentAgent, err := react.New(
		react.WithProvider(parentP),
		react.WithName("supervisor"),
		react.WithTools(subAgent.AsTool("delegate_worker", "delegate to worker")),
		react.WithHooks(react.Hooks{
			SubagentStart: func(ctx context.Context, rc *react.RunContext, agentName, task string) {
				startedAgent = agentName
				startedTask = task
			},
			SubagentStop: func(ctx context.Context, rc *react.RunContext, agentName, task, output string, err error) {
				stoppedAgent = agentName
				stoppedOut = output
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := parentAgent.Run(context.Background(), "", "delegate task")
	if err != nil {
		t.Fatal(err)
	}
	if startedAgent != "delegate_worker" || startedTask != "do research" {
		t.Fatalf("unexpected SubagentStart: agent=%s task=%s", startedAgent, startedTask)
	}
	if stoppedAgent != "delegate_worker" || stoppedOut != "subagent finished" {
		t.Fatalf("unexpected SubagentStop: agent=%s out=%s", stoppedAgent, stoppedOut)
	}
	if res.Output != "parent finished" {
		t.Fatalf("unexpected parent output: %s", res.Output)
	}
}

func TestCompactionHooks(t *testing.T) {
	var (
		preCalled  atomic.Bool
		postCalled atomic.Bool
	)

	p := mock.New(mock.Text("done"))

	a, err := react.New(
		react.WithProvider(p),
		react.WithHooks(react.Hooks{
			PreCompact: func(ctx context.Context, rc *react.RunContext, messageCount int) {
				preCalled.Store(true)
			},
			PostCompact: func(ctx context.Context, rc *react.RunContext, beforeCount, afterCount int) {
				postCalled.Store(true)
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	_, err = a.Run(context.Background(), "", "hello")
	if err != nil {
		t.Fatal(err)
	}
	if !preCalled.Load() || !postCalled.Load() {
		t.Fatalf("compaction hooks not called: pre=%v post=%v", preCalled.Load(), postCalled.Load())
	}
}

func TestStopFailureHook(t *testing.T) {
	var failureSeen error
	p := mock.New(mock.Error(errors.New("upstream provider failure")))

	a, err := react.New(
		react.WithProvider(p),
		react.WithHooks(react.Hooks{
			OnStopFailure: func(ctx context.Context, rc *react.RunContext, err error) {
				failureSeen = err
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	_, err = a.Run(context.Background(), "", "run")
	if err == nil {
		t.Fatal("expected run error")
	}
	if failureSeen == nil || failureSeen.Error() != "agent: model call: upstream provider failure" {
		t.Fatalf("unexpected failure captured: %v", failureSeen)
	}
}

func TestRunContextConcurrency(t *testing.T) {
	rc := &react.RunContext{}
	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(2)
		go func(val int) {
			defer wg.Done()
			rc.Set("key", val)
			rc.AddContext("reminder")
		}(i)
		go func() {
			defer wg.Done()
			rc.Get("key")
			_ = rc.ConsumeContext()
		}()
	}
	wg.Wait()
}
