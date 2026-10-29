package tests

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider/mock"
)

// add and destroy are deliberately unannotated: under the default policy
// unannotated tools are auto-allowed, so any blocking observed here comes
// from the allowlist, not from permission policy.
func allowlistTools(destroyCalls *atomic.Int32) (react.Tool, react.Tool) {
	add := react.NewTool("add", "adds two numbers", func(ctx context.Context, in calcArgs) (int, error) {
		return in.A + in.B, nil
	})
	destroy := react.NewTool("destroy", "side-effect tool", func(ctx context.Context, in struct{}) (string, error) {
		destroyCalls.Add(1)
		return "destroyed", nil
	})
	return add, destroy
}

func TestWithAllowedToolsBlocksExecution(t *testing.T) {
	var destroyCalls atomic.Int32
	add, destroy := allowlistTools(&destroyCalls)
	p := mock.New(
		mock.ToolCalls("", mock.Call("destroy", `{}`)),
		mock.Text("done"),
	)
	a, err := react.New(react.WithProvider(p), react.WithTools(add, destroy))
	if err != nil {
		t.Fatal(err)
	}
	res, err := a.Run(context.Background(), "allow-session", "go", react.WithAllowedTools("add"))
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "done" {
		t.Fatalf("output = %q, want %q", res.Output, "done")
	}
	if destroyCalls.Load() != 0 {
		t.Fatalf("destroy ran %d times despite allowlist", destroyCalls.Load())
	}
	msgs, err := a.Memory().Load(context.Background(), "allow-session")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, m := range msgs {
		if m.Role == react.RoleTool && strings.Contains(m.Text(), "unknown tool") {
			found = true
		}
	}
	if !found {
		t.Fatalf("no unknown-tool error in history: %+v", msgs)
	}
}

func TestWithAllowedToolsEmptyMeansNoTools(t *testing.T) {
	var destroyCalls atomic.Int32
	add, destroy := allowlistTools(&destroyCalls)
	p := mock.New(
		mock.ToolCalls("", mock.Call("add", `{"a": 1, "b": 2}`)),
		mock.Text("done"),
	)
	a, err := react.New(react.WithProvider(p), react.WithTools(add, destroy))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Run(context.Background(), "allow-empty", "go", react.WithAllowedTools()); err != nil {
		t.Fatal(err)
	}
	reqs := p.Requests()
	if len(reqs) == 0 || len(reqs[0].Tools) != 0 {
		t.Fatalf("expected zero tool specs in first request, got %+v", reqs)
	}
}

func TestAsToolWithAllowedTools(t *testing.T) {
	var destroyCalls atomic.Int32
	add, destroy := allowlistTools(&destroyCalls)

	childMock := mock.New(
		mock.ToolCalls("", mock.Call("destroy", `{}`)),
		mock.Text("child done"),
	)
	child, err := react.New(react.WithProvider(childMock), react.WithTools(add, destroy))
	if err != nil {
		t.Fatal(err)
	}

	parentMock := mock.New(
		mock.ToolCalls("", mock.Call("delegate", map[string]any{"task": "do it"})),
		mock.Text("parent done"),
	)
	parent, err := react.New(
		react.WithProvider(parentMock),
		react.WithTools(child.AsToolWithAllowedTools("delegate", "delegate work", "add")),
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err := parent.Run(context.Background(), "parent-session", "go")
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "parent done" {
		t.Fatalf("output = %q, want %q", res.Output, "parent done")
	}
	if destroyCalls.Load() != 0 {
		t.Fatalf("child destroy ran %d times despite delegation allowlist", destroyCalls.Load())
	}
	// The child model must never even see the disallowed spec.
	reqs := childMock.Requests()
	if len(reqs) == 0 {
		t.Fatal("child provider received no requests")
	}
	var names []string
	for _, s := range reqs[0].Tools {
		names = append(names, s.Name)
	}
	if len(names) != 1 || names[0] != "add" {
		t.Fatalf("child request tools = %v, want [add]", names)
	}
}

func TestAsToolUnrestrictedStillAllowsAll(t *testing.T) {
	var destroyCalls atomic.Int32
	add, destroy := allowlistTools(&destroyCalls)

	childMock := mock.New(
		mock.ToolCalls("", mock.Call("destroy", `{}`)),
		mock.Text("child done"),
	)
	child, err := react.New(react.WithProvider(childMock), react.WithTools(add, destroy))
	if err != nil {
		t.Fatal(err)
	}
	parentMock := mock.New(
		mock.ToolCalls("", mock.Call("delegate", map[string]any{"task": "do it"})),
		mock.Text("parent done"),
	)
	parent, err := react.New(
		react.WithProvider(parentMock),
		react.WithTools(child.AsTool("delegate", "delegate work")),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := parent.Run(context.Background(), "parent-session-2", "go"); err != nil {
		t.Fatal(err)
	}
	if destroyCalls.Load() != 1 {
		t.Fatalf("destroy ran %d times, want 1 (unrestricted AsTool must not filter)", destroyCalls.Load())
	}
}
