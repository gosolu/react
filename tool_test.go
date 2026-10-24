package react

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type nested struct {
	Tags []string `json:"tags,omitempty"`
}

type sample struct {
	Name   string            `json:"name" desc:"the name"`
	Count  int               `json:"count,omitempty"`
	Ratio  *float64          `json:"ratio"`
	Mode   string            `json:"mode" enum:"fast,slow"`
	Level  int               `json:"level" enum:"1,2"`
	Meta   map[string]string `json:"meta" required:"false"`
	Nested nested            `json:"nested"`
	Self   *sample           `json:"self,omitempty"`
	Skip   string            `json:"-"`
}

func TestSchema(t *testing.T) {
	var s map[string]any
	if err := json.Unmarshal(SchemaFor[sample](), &s); err != nil {
		t.Fatal(err)
	}
	props := s["properties"].(map[string]any)
	if len(props) != 8 {
		t.Fatalf("props = %v", props)
	}
	if props["name"].(map[string]any)["description"] != "the name" {
		t.Fatal("desc missing")
	}
	if props["level"].(map[string]any)["enum"].([]any)[0] != float64(1) {
		t.Fatal("int enum")
	}
	if props["ratio"].(map[string]any)["type"] != "number" {
		t.Fatal("ratio")
	}
	req := s["required"].([]any)
	got := map[string]bool{}
	for _, r := range req {
		got[r.(string)] = true
	}
	for _, want := range []string{"name", "mode", "level", "nested"} {
		if !got[want] {
			t.Fatalf("missing required %s in %v", want, req)
		}
	}
	for _, not := range []string{"count", "ratio", "meta", "self"} {
		if got[not] {
			t.Fatalf("%s should be optional", not)
		}
	}
}

type addIn struct{ A, B int }

func TestTypedTool(t *testing.T) {
	add := NewTool("add", "add", func(_ context.Context, in addIn) (int, error) { return in.A + in.B, nil })
	r, err := add.Call(context.Background(), json.RawMessage(`{"A":2,"B":3}`))
	if err != nil || r.Text() != "5" {
		t.Fatalf("%v %v", r, err)
	}
	r, _ = add.Call(context.Background(), json.RawMessage(`{"A":"x"}`))
	if !r.IsError {
		t.Fatal("expected invalid-arg error result")
	}
}

func TestRegistry(t *testing.T) {
	r := NewToolRegistry()
	tl := NewRawTool(ToolSpec{Name: "a"}, nil)
	if err := r.Register(tl); err != nil {
		t.Fatal(err)
	}
	if r.Register(tl) == nil {
		t.Fatal("dup allowed")
	}
	if r.Register(NewRawTool(ToolSpec{Name: "bad name"}, nil)) == nil {
		t.Fatal("bad name allowed")
	}
}

func TestExecutorParallelOrderAndErrors(t *testing.T) {
	var running, peak atomic.Int32
	h := func(ctx context.Context, c ToolCall) (ToolResult, error) {
		n := running.Add(1)
		defer running.Add(-1)
		for {
			p := peak.Load()
			if n <= p || peak.CompareAndSwap(p, n) {
				break
			}
		}
		switch c.Name {
		case "panic":
			panic("boom")
		case "fail":
			return ToolResult{}, errors.New("nope")
		case "slow":
			select {
			case <-time.After(time.Second):
			case <-ctx.Done():
				return ToolResult{}, ctx.Err()
			}
		case "seq":
			if running.Load() != 1 {
				return ErrorResult("not alone"), nil
			}
		}
		time.Sleep(50 * time.Millisecond)
		return TextResult(c.ID), nil
	}
	calls := []ToolCall{{ID: "1", Name: "ok"}, {ID: "2", Name: "ok"}, {ID: "3", Name: "panic"}, {ID: "4", Name: "fail"}, {ID: "5", Name: "seq"}, {ID: "6", Name: "slow"}}
	ex := ToolExecutor{MaxParallel: 4, Timeout: 200 * time.Millisecond}
	var done atomic.Int32
	start := time.Now()
	res := ex.Run(context.Background(), calls, func(n string) bool { return n == "seq" }, h, func(int, *ToolCallResult) { done.Add(1) })
	if time.Since(start) > 600*time.Millisecond {
		t.Fatalf("too slow: %v", time.Since(start))
	}
	if done.Load() != 6 || peak.Load() < 2 {
		t.Fatalf("done=%d peak=%d", done.Load(), peak.Load())
	}
	if res[0].Result.Text() != "1" || res[1].Result.Text() != "2" {
		t.Fatal("order")
	}
	if !res[2].Result.IsError || !strings.Contains(res[2].Result.Text(), "panicked") {
		t.Fatal("panic not converted")
	}
	if !res[3].Result.IsError || res[3].Err == nil {
		t.Fatal("error not converted")
	}
	if res[4].Result.IsError {
		t.Fatal("sequential tool ran concurrently")
	}
	if !res[5].Result.IsError || !strings.Contains(res[5].Result.Text(), "timed out") {
		t.Fatalf("timeout: %v", res[5].Result.Text())
	}
}

func TestToolExecutorOnDoneMutatesStoredResult(t *testing.T) {
	h := func(context.Context, ToolCall) (ToolResult, error) {
		return TextResult("original"), nil
	}
	var calls int
	res := (ToolExecutor{}).Run(
		context.Background(),
		[]ToolCall{{ID: "1", Name: "x"}},
		nil,
		h,
		func(_ int, cr *ToolCallResult) {
			calls++
			cr.Result.Content = []Part{TextPart("changed")}
			cr.Result.Truncated = true
		},
	)
	if calls != 1 {
		t.Fatalf("onDone calls = %d, want 1", calls)
	}
	if got := res[0].Result.Text(); got != "changed" {
		t.Fatalf("stored result = %q, want changed", got)
	}
	if !res[0].Result.Truncated {
		t.Fatal("stored Truncated flag was not mutated")
	}
}

type embedA struct {
	*embedB
	FieldA string `json:"field_a"`
}

type embedB struct {
	*embedA
	FieldB int `json:"field_b"`
}

func TestSchemaMutuallyRecursiveEmbeddedStructs(t *testing.T) {
	raw := SchemaFor[embedA]()
	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("invalid schema JSON: %v", err)
	}
	props, ok := s["properties"].(map[string]any)
	if !ok {
		t.Fatalf("missing properties: %v", s)
	}
	if props["field_a"] == nil || props["field_b"] == nil {
		t.Fatalf("expected both field_a and field_b in flattened schema, got: %v", props)
	}
}

func TestExecutorSemaphoreCancellationAndUncooperativeTool(t *testing.T) {
	t.Run("cancelled context aborts waiting semaphore callers", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		var executed atomic.Int32
		started := make(chan struct{})
		release := make(chan struct{})
		defer close(release)

		h := func(ctx context.Context, c ToolCall) (ToolResult, error) {
			n := executed.Add(1)
			if n == 1 {
				close(started)
				select {
				case <-release:
				case <-ctx.Done():
					return ToolResult{}, ctx.Err()
				}
			}
			return TextResult("ok"), nil
		}

		go func() {
			<-started
			// Cancel while call 1 holds the only semaphore slot and calls 2..4 are waiting.
			time.Sleep(15 * time.Millisecond)
			cancel()
		}()

		calls := []ToolCall{
			{ID: "1", Name: "t"},
			{ID: "2", Name: "t"},
			{ID: "3", Name: "t"},
			{ID: "4", Name: "t"},
		}
		ex := ToolExecutor{MaxParallel: 1}
		res := ex.Run(ctx, calls, nil, h, nil)
		if got := executed.Load(); got != 1 {
			t.Fatalf("expected only 1 tool call to execute before cancellation, got %d", got)
		}
		for i, r := range res {
			if !r.Result.IsError || r.Err == nil {
				t.Fatalf("call %d expected canceled error result, got %+v", i, r)
			}
		}
	})

	t.Run("uncooperative tool does not deadlock executor", func(t *testing.T) {
		blockForever := make(chan struct{})
		defer close(blockForever)

		h := func(_ context.Context, c ToolCall) (ToolResult, error) {
			<-blockForever // completely ignores ctx.Done()
			return TextResult("unreachable"), nil
		}

		ex := ToolExecutor{MaxParallel: 1, Timeout: 30 * time.Millisecond}
		start := time.Now()
		res := ex.Run(context.Background(), []ToolCall{{ID: "1", Name: "hung"}}, nil, h, nil)
		if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
			t.Fatalf("executor blocked too long on uncooperative tool: %v", elapsed)
		}
		if !res[0].Result.IsError || !strings.Contains(res[0].Result.Text(), "timed out") {
			t.Fatalf("expected timeout error result, got: %+v", res[0])
		}
	})
}
