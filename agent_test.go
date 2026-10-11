package react

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/llm/mock"
)

type calcArgs struct {
	A int `json:"a"`
	B int `json:"b"`
}

func setupTestTools() (Tool, Tool) {
	add := NewTool("add", "adds two numbers", func(ctx context.Context, in calcArgs) (int, error) {
		return in.A + in.B, nil
	})
	dangerous := NewTool("destroy", "destructive tool", func(ctx context.Context, in struct{}) (string, error) {
		return "destroyed", nil
	}, Destructive())
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
	a1, err := New(
		WithProvider(p1),
		WithTools(add),
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
	a2, err := New(
		WithProvider(p2),
		WithTools(add),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	var streamResult *Result
	var textDeltas []string
	for ev, err := range a2.Stream(ctx, "session-2", "calculate 2+3") {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if ev.Type == EventTextDelta {
			textDeltas = append(textDeltas, ev.Text)
		}
		if ev.Type == EventRunEnd {
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
	p := mock.NewFunc(func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
		return &llm.Response{
			Message: llm.Message{
				Role:      llm.RoleAssistant,
				ToolCalls: []llm.ToolCall{mock.Call("loop", `{}`)},
			},
			StopReason: llm.StopToolUse,
		}, nil
	})

	loopTool := NewTool("loop", "loop", func(ctx context.Context, in struct{}) (string, error) {
		return "ok", nil
	})

	a, err := New(
		WithProvider(p),
		WithTools(loopTool),
		WithMaxTurns(3),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(context.Background(), "s1", "start")
	if err != ErrMaxTurns {
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
	approver := PermissionApproverFunc(func(ctx context.Context, req ApprovalRequest) (bool, error) {
		return approved.Load(), nil
	})

	a, err := New(
		WithProvider(p),
		WithTools(dangerous),
		WithPermission(DefaultPermissionPolicy(), approver),
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
	a2, _ := New(
		WithProvider(p2),
		WithTools(dangerous),
		WithPermission(DefaultPermissionPolicy(), approver),
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
	child, err := New(
		WithName("child-agent"),
		WithProvider(childProvider),
	)
	if err != nil {
		t.Fatal(err)
	}

	childTool := child.AsTool("ask_child", "delegate to child")

	parentProvider := mock.New(
		mock.ToolCalls("delegating", mock.Call("ask_child", `{"task": "do work"}`)),
		mock.Text("final parent answer"),
	)

	var childEventsReceived []Event
	parent, err := New(
		WithName("parent-agent"),
		WithProvider(parentProvider),
		WithTools(childTool),
		WithHooks(Hooks{
			OnEvent: func(ctx context.Context, ev Event) {
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

	a, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}

	count := 0
	for ev, err := range a.Stream(context.Background(), "s1", "hi") {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Type == EventTextDelta {
			count++
			if count == 3 {
				break // early break
			}
		}
	}
	// Verify no hang or leak
	time.Sleep(10 * time.Millisecond)
}
