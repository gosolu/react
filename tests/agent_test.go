package tests

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider/mock"
)

type calcArgs struct {
	A int `json:"a"`
	B int `json:"b"`
}

func setupTestTools() (react.Tool, react.Tool) {
	add := react.NewTool("add", "adds two numbers", func(ctx context.Context, in calcArgs) (int, error) {
		return in.A + in.B, nil
	})
	dangerous := react.NewTool("destroy", "destructive tool", func(ctx context.Context, in struct{}) (string, error) {
		return "destroyed", nil
	}, react.Destructive())
	return add, dangerous
}

func TestAgentStreamAndRunParity(t *testing.T) {
	add, _ := setupTestTools()

	makeProvider := func() *mock.Provider {
		return mock.New(
			mock.ToolCalls("thinking...", mock.Call("add", `{"a": 2, "b": 3}`)),
			mock.Text("The answer is 5."),
		)
	}

	ctx := context.Background()

	// 1. Test Run (non-stream)
	p1 := makeProvider()
	a1, err := react.New(
		react.WithProvider(p1),
		react.WithTools(add),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	resRun, err := a1.Run(ctx, "session-1", "calculate 2+3")
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if resRun.Output != "The answer is 5." {
		t.Fatalf("Run output mismatch: %q", resRun.Output)
	}
	if resRun.Turns != 2 {
		t.Fatalf("Run turns = %d, want 2", resRun.Turns)
	}

	// 2. Test Stream
	p2 := makeProvider()
	a2, err := react.New(
		react.WithProvider(p2),
		react.WithTools(add),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	var streamResult *react.Result
	var textDeltas []string
	for ev, err := range a2.Stream(ctx, "session-2", "calculate 2+3") {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if ev.Type == react.EventTextDelta {
			textDeltas = append(textDeltas, ev.Text)
		}
		if ev.Type == react.EventRunEnd {
			streamResult = ev.Result
		}
	}

	if streamResult == nil {
		t.Fatal("Stream never yielded EventRunEnd with Result")
	}
	if streamResult.Output != resRun.Output {
		t.Fatalf("Stream Output = %q, want %q", streamResult.Output, resRun.Output)
	}
	if len(streamResult.Messages) != len(resRun.Messages) {
		t.Fatalf("Messages count mismatch: stream=%d, run=%d", len(streamResult.Messages), len(resRun.Messages))
	}
	joinedDeltas := strings.Join(textDeltas, "")
	if !strings.Contains(joinedDeltas, "5.") {
		t.Fatalf("Text deltas missing expected output: %q", joinedDeltas)
	}
}

func TestAgentMaxTurns(t *testing.T) {
	// Infinitely loop tool calls
	p := mock.NewFunc(func(ctx context.Context, req *react.Request) (*react.Response, error) {
		return &react.Response{
			Message: react.Message{
				Role:      react.RoleAssistant,
				ToolCalls: []react.ToolCall{mock.Call("loop", `{}`)},
			},
			StopReason: react.StopToolCall,
		}, nil
	})

	loopTool := react.NewTool("loop", "loop", func(ctx context.Context, in struct{}) (string, error) {
		return "ok", nil
	})

	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(loopTool),
		react.WithMaxTurns(3),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "s1", "start")
	if err != react.ErrMaxTurns {
		t.Fatalf("expected ErrMaxTurns, got %v", err)
	}
	if res.Turns != 3 {
		t.Fatalf("expected 3 turns, got %d", res.Turns)
	}
}

func TestAgentPermissions(t *testing.T) {
	_, dangerous := setupTestTools()
	p := mock.New(
		mock.ToolCalls("", mock.Call("destroy", `{}`)),
		mock.Text("Done."),
	)

	var approved atomic.Bool
	approver := react.PermissionApproverFunc(func(ctx context.Context, req react.ApprovalRequest) (bool, error) {
		return approved.Load(), nil
	})

	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(dangerous),
		react.WithPermission(react.DefaultPermissionPolicy(), approver),
	)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Denied
	approved.Store(false)
	res, err := a.Run(context.Background(), "s1", "destroy it")
	if err != nil {
		t.Fatal(err)
	}
	// Verify that the tool message in history reflects user denial
	toolMsg := res.Messages[2]
	if !toolMsg.IsError || !strings.Contains(toolMsg.Text(), "denied") {
		t.Fatalf("expected denial tool result, got: %+v", toolMsg)
	}

	// 2. Approved
	p2 := mock.New(
		mock.ToolCalls("", mock.Call("destroy", `{}`)),
		mock.Text("Done."),
	)
	approved.Store(true)
	a2, _ := react.New(
		react.WithProvider(p2),
		react.WithTools(dangerous),
		react.WithPermission(react.DefaultPermissionPolicy(), approver),
	)
	res2, err := a2.Run(context.Background(), "s2", "destroy it")
	if err != nil {
		t.Fatal(err)
	}
	toolMsg2 := res2.Messages[2]
	if toolMsg2.IsError || !strings.Contains(toolMsg2.Text(), "destroyed") {
		t.Fatalf("expected successful execution, got: %+v", toolMsg2)
	}
}

func TestAgentSubAgent(t *testing.T) {
	childProvider := mock.New(
		mock.Text("child answer"),
	)
	child, err := react.New(
		react.WithName("child-agent"),
		react.WithProvider(childProvider),
	)
	if err != nil {
		t.Fatal(err)
	}

	childTool := child.AsTool("ask_child", "delegate to child")

	parentProvider := mock.New(
		mock.ToolCalls("delegating", mock.Call("ask_child", `{"task": "do work"}`)),
		mock.Text("final parent answer"),
	)

	var childEventsReceived []react.Event
	parent, err := react.New(
		react.WithName("parent-agent"),
		react.WithProvider(parentProvider),
		react.WithTools(childTool),
		react.WithHooks(react.Hooks{
			OnEvent: func(ctx context.Context, ev react.Event) {
				if ev.Agent == "child-agent" {
					childEventsReceived = append(childEventsReceived, ev)
				}
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := parent.Run(context.Background(), "s1", "start")
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "final parent answer" {
		t.Fatalf("output = %q", res.Output)
	}
	if len(childEventsReceived) == 0 {
		t.Fatal("expected forwarded events from child agent")
	}
	if childEventsReceived[0].Depth != 1 {
		t.Fatalf("expected child depth 1, got %d", childEventsReceived[0].Depth)
	}
}

func TestStreamBreakCancellation(t *testing.T) {
	p := mock.New(
		mock.Text("long text stream that should be stopped early"),
	)
	p.ChunkSize = 1 // 1 char per chunk

	a, err := react.New(react.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}

	count := 0
	for ev, err := range a.Stream(context.Background(), "s1", "hi") {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Type == react.EventTextDelta {
			count++
			if count == 3 {
				break // early break
			}
		}
	}
	// Verify no hang or leak
	time.Sleep(10 * time.Millisecond)
}

func TestAgentSessionConcurrency(t *testing.T) {
	p := mock.New(
		mock.Text("response 1"),
		mock.Text("response 2"),
		mock.Text("response 3"),
		mock.Text("response 4"),
		mock.Text("response 5"),
	)

	a, err := react.New(
		react.WithProvider(p),
		react.WithStore(react.NewInMemory()),
	)
	if err != nil {
		t.Fatal(err)
	}

	const n = 5
	errCh := make(chan error, n)
	for i := 0; i < n; i++ {
		go func(idx int) {
			_, err := a.Run(context.Background(), "concurrent-session", "hello")
			errCh <- err
		}(i)
	}

	for i := 0; i < n; i++ {
		if err := <-errCh; err != nil {
			t.Fatalf("concurrent Run failed: %v", err)
		}
	}

	history, err := a.Store().Load(context.Background(), "concurrent-session")
	if err != nil {
		t.Fatalf("load history failed: %v", err)
	}
	// Each run should append exactly 1 user message and 1 assistant message in order:
	// Total messages = 2 * n = 10.
	if len(history) != 2*n {
		t.Fatalf("expected %d messages, got %d", 2*n, len(history))
	}
	for i := 0; i < len(history); i += 2 {
		if history[i].Role != react.RoleUser {
			t.Errorf("msg %d expected RoleUser, got %s", i, history[i].Role)
		}
		if history[i+1].Role != react.RoleAssistant {
			t.Errorf("msg %d expected RoleAssistant, got %s", i+1, history[i+1].Role)
		}
	}
}

func TestAgentReasoningPersistence(t *testing.T) {
	add, _ := setupTestTools()
	p := mock.New(
		mock.Step{
			Response: &react.Response{
				Message: react.Message{
					Role: react.RoleAssistant,
					ToolCalls: []react.ToolCall{
						mock.Call("add", `{"a": 1, "b": 2}`),
					},
				},
				Reasoning:  "Step 1: compute 1+2",
				StopReason: react.StopToolCall,
			},
		},
		mock.Step{
			Response: &react.Response{
				Message: react.Message{
					Role:    react.RoleAssistant,
					Content: []react.Part{react.TextPart("3")},
				},
				Reasoning:  "Step 2: final answer is 3",
				StopReason: react.StopEnd,
			},
		},
	)

	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(add),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "reasoning-session", "compute 1+2")
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "3" {
		t.Fatalf("expected 3, got %q", res.Output)
	}

	history, err := a.Store().Load(context.Background(), "reasoning-session")
	if err != nil {
		t.Fatal(err)
	}

	// Turn 1 assistant message should have Reasoning preserved:
	var foundAssistantReasoning bool
	for _, m := range history {
		if m.Role == react.RoleAssistant && m.Reasoning == "Step 1: compute 1+2" {
			foundAssistantReasoning = true
			break
		}
	}
	if !foundAssistantReasoning {
		t.Fatal("expected assistant message in history to preserve Reasoning")
	}
}

func TestUsageCacheTokens(t *testing.T) {
	u1 := react.Usage{
		InputTokens:      100,
		OutputTokens:     50,
		CacheReadTokens:  40,
		CacheWriteTokens: 10,
	}
	u2 := react.Usage{
		InputTokens:      200,
		OutputTokens:     80,
		CacheReadTokens:  60,
		CacheWriteTokens: 20,
	}
	u1.Add(u2)

	if u1.InputTokens != 300 {
		t.Errorf("expected 300 input tokens, got %d", u1.InputTokens)
	}
	if u1.OutputTokens != 130 {
		t.Errorf("expected 130 output tokens, got %d", u1.OutputTokens)
	}
	if u1.CacheReadTokens != 100 {
		t.Errorf("expected 100 cache read tokens, got %d", u1.CacheReadTokens)
	}
	if u1.CacheWriteTokens != 30 {
		t.Errorf("expected 30 cache write tokens, got %d", u1.CacheWriteTokens)
	}
	if u1.Total() != 430 {
		t.Errorf("expected 430 total tokens, got %d", u1.Total())
	}
}
