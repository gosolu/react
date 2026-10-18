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

func TestTruncationMarkerAndFullEventResult(t *testing.T) {
	long := react.NewTool("long", "returns a long string", func(context.Context, struct{}) (string, error) {
		return strings.Repeat("x", 100), nil
	})
	p := mock.New(
		mock.ToolCalls("", mock.Call("long", `{}`)),
		mock.Text("done"),
	)
	var (
		batchText      string
		batchTruncated bool
	)
	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(long),
		react.WithMaxToolOutput(10),
		react.WithHooks(react.Hooks{
			AfterToolBatch: func(_ context.Context, _ *react.RunContext, results []react.ToolCallResult) error {
				if len(results) > 0 {
					batchText = results[0].Result.Text()
					batchTruncated = results[0].Result.Truncated
				}
				return nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	var toolEvents []react.Event
	res, err := a.Run(context.Background(), "", "hi", react.WithEventHandler(func(ev react.Event) {
		if ev.Type == react.EventToolResult {
			toolEvents = append(toolEvents, ev)
		}
	}))
	if err != nil {
		t.Fatal(err)
	}

	var toolText string
	for _, m := range res.Messages {
		if m.Role == react.RoleTool {
			toolText = m.Text()
		}
	}
	if !strings.Contains(toolText, "output truncated from 100 bytes") {
		t.Fatalf("expected a truncation marker, got %q", toolText)
	}
	if len(toolText) >= 100 {
		t.Fatalf("model-facing tool text was not bounded: %d bytes", len(toolText))
	}
	if len(toolEvents) != 1 {
		t.Fatalf("tool events = %d, want 1", len(toolEvents))
	}
	if toolEvents[0].ToolResult == nil || !toolEvents[0].ToolResult.Truncated {
		t.Fatal("expected Truncated on the event result")
	}
	if got := len(toolEvents[0].ToolResult.Text()); got != 100 {
		t.Fatalf("full event output length = %d, want 100", got)
	}
	if len(batchText) != 100 || !batchTruncated {
		t.Fatalf("AfterToolBatch saw len=%d truncated=%v, want the full result with the flag", len(batchText), batchTruncated)
	}
	if toolEvents[0].Duration <= 0 {
		t.Fatalf("tool duration = %s", toolEvents[0].Duration)
	}
	if res.Duration <= 0 {
		t.Fatalf("run duration = %s", res.Duration)
	}
}

func TestPerToolOutputLimit(t *testing.T) {
	makeTool := func(name string, opts ...react.ToolOption) react.Tool {
		return react.NewTool(name, "returns 100 bytes", func(context.Context, struct{}) (string, error) {
			return strings.Repeat("y", 100), nil
		}, opts...)
	}
	small := makeTool("small", react.WithToolMaxOutput(5))
	big := makeTool("big")

	p := mock.New(
		mock.ToolCalls("", mock.Call("small", `{}`), mock.Call("big", `{}`)),
		mock.Text("done"),
	)
	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(small, big),
		react.WithMaxToolOutput(1000),
	)
	if err != nil {
		t.Fatal(err)
	}

	events := map[string]react.Event{}
	_, err = a.Run(context.Background(), "", "hi", react.WithEventHandler(func(ev react.Event) {
		if ev.Type == react.EventToolResult && ev.ToolCall != nil {
			events[ev.ToolCall.Name] = ev
		}
	}))
	if err != nil {
		t.Fatal(err)
	}
	smallEvent, ok := events["small"]
	if !ok || smallEvent.ToolResult == nil || !smallEvent.ToolResult.Truncated {
		t.Fatal("small tool should be truncated by its per-tool limit")
	}
	bigEvent, ok := events["big"]
	if !ok || bigEvent.ToolResult == nil || bigEvent.ToolResult.Truncated {
		t.Fatal("big tool should not be truncated")
	}
	if got := len(bigEvent.ToolResult.Text()); got != 100 {
		t.Fatalf("big full output = %d, want 100", got)
	}
	if got := len(smallEvent.ToolResult.Text()); got != 100 {
		t.Fatalf("small full output = %d, want 100", got)
	}
}

func TestEventDurations(t *testing.T) {
	p := mock.New(
		mock.ToolCalls("", mock.Call("noop", `{}`)),
		mock.Text("done"),
	)
	a, err := react.New(react.WithProvider(p), react.WithTools(noopTool("noop")))
	if err != nil {
		t.Fatal(err)
	}

	var events []react.Event
	res, err := a.Run(context.Background(), "", "hi", react.WithEventHandler(func(ev react.Event) {
		events = append(events, ev)
	}))
	if err != nil {
		t.Fatal(err)
	}

	var sawTool, sawModel, sawRun bool
	for _, ev := range events {
		switch ev.Type {
		case react.EventToolResult:
			sawTool = true
			if ev.Duration <= 0 {
				t.Fatalf("tool event duration = %s", ev.Duration)
			}
		case react.EventMessageComplete:
			sawModel = true
			if ev.Duration <= 0 {
				t.Fatalf("model event duration = %s", ev.Duration)
			}
		case react.EventRunEnd:
			sawRun = true
			if ev.Duration <= 0 {
				t.Fatalf("run event duration = %s", ev.Duration)
			}
			if ev.Result == nil || ev.Result.RunID == "" {
				t.Fatal("expected a run result with an ID")
			}
		}
	}
	if !sawTool || !sawModel || !sawRun {
		t.Fatalf("missing events: tool=%v model=%v run=%v", sawTool, sawModel, sawRun)
	}
	if res.Duration <= 0 {
		t.Fatalf("result duration = %s", res.Duration)
	}
}

func TestRunIDAndStopReasons(t *testing.T) {
	t.Run("normal", func(t *testing.T) {
		newAgent := func() *react.Agent {
			a, err := react.New(react.WithProvider(mock.New(mock.Text("done"))))
			if err != nil {
				t.Fatal(err)
			}
			return a
		}
		r1, err := newAgent().Run(context.Background(), "", "hi")
		if err != nil {
			t.Fatal(err)
		}
		r2, err := newAgent().Run(context.Background(), "", "again")
		if err != nil {
			t.Fatal(err)
		}
		if r1.RunID == "" || r2.RunID == "" {
			t.Fatal("expected non-empty run IDs")
		}
		if r1.RunID == r2.RunID {
			t.Fatal("expected unique run IDs")
		}
		if r1.StopReason != react.StopEnd {
			t.Fatalf("stop reason = %q", r1.StopReason)
		}
	})

	t.Run("max turns", func(t *testing.T) {
		p := mock.NewFunc(func(context.Context, *react.Request) (*react.Response, error) {
			return &react.Response{
				Message: react.Message{
					Role:      react.RoleAssistant,
					ToolCalls: []react.ToolCall{{ID: "c1", Name: "noop", Arguments: json.RawMessage(`{}`)}},
				},
				StopReason: react.StopToolCall,
			}, nil
		})
		a, err := react.New(
			react.WithProvider(p),
			react.WithTools(noopTool("noop")),
			react.WithMaxTurns(1),
		)
		if err != nil {
			t.Fatal(err)
		}
		res, err := a.Run(context.Background(), "", "hi")
		if !errors.Is(err, react.ErrMaxTurns) {
			t.Fatalf("err = %v, want ErrMaxTurns", err)
		}
		if res.StopReason != react.StopMaxTurns {
			t.Fatalf("stop reason = %q", res.StopReason)
		}
	})

	t.Run("provider error", func(t *testing.T) {
		p := mock.New(mock.Error(errors.New("boom")))
		a, err := react.New(react.WithProvider(p))
		if err != nil {
			t.Fatal(err)
		}
		res, err := a.Run(context.Background(), "", "hi")
		if err == nil {
			t.Fatal("expected an error")
		}
		if res.StopReason != react.StopError {
			t.Fatalf("stop reason = %q", res.StopReason)
		}
	})

	t.Run("canceled", func(t *testing.T) {
		p := mock.NewFunc(func(ctx context.Context, _ *react.Request) (*react.Response, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		})
		a, err := react.New(react.WithProvider(p))
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		res, err := a.Run(ctx, "", "hi")
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("err = %v, want DeadlineExceeded", err)
		}
		if res.StopReason != react.StopCanceled {
			t.Fatalf("stop reason = %q", res.StopReason)
		}
	})
}
