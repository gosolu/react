package react

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

func TestInMemory(t *testing.T) {
	runStoreTest(t, func() Store { return NewInMemory() })
}

// runStoreTest exercises a Store implementation for conformance.
func runStoreTest(t *testing.T, newStore func() Store) {
	ctx := context.Background()

	t.Run("EmptyLoad", func(t *testing.T) {
		s := newStore()
		msgs, err := s.Load(ctx, "missing")
		if err != nil || len(msgs) != 0 {
			t.Fatalf("Load(missing) = %v, %v", msgs, err)
		}
	})

	t.Run("AppendLoadRoundTrip", func(t *testing.T) {
		s := newStore()
		in := []Message{
			UserText("hi"),
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "t", Arguments: []byte(`{"x":1}`)}}},
			ToolResultMessage("c1", "t", []Part{TextPart("ok")}, true),
		}
		if err := s.Append(ctx, "a", in[0]); err != nil {
			t.Fatal(err)
		}
		if err := s.Append(ctx, "a", in[1:]...); err != nil {
			t.Fatal(err)
		}
		got, err := s.Load(ctx, "a")
		if err != nil || len(got) != 3 {
			t.Fatalf("Load = %d msgs, %v", len(got), err)
		}
		if got[0].Text() != "hi" || got[1].ToolCalls[0].Name != "t" || string(got[1].ToolCalls[0].Arguments) != `{"x":1}` ||
			got[2].ToolCallID != "c1" || !got[2].IsError || got[2].Text() != "ok" {
			t.Fatalf("round trip mismatch: %+v", got)
		}
		if other, _ := s.Load(ctx, "b"); len(other) != 0 {
			t.Fatal("sessions not isolated")
		}
	})

	t.Run("NoAliasing", func(t *testing.T) {
		s := newStore()
		m := UserText("orig")
		_ = s.Append(ctx, "a", m)
		m.Content[0].Text = "mutated"
		got, _ := s.Load(ctx, "a")
		got[0].Content[0].Text = "mutated2"
		again, _ := s.Load(ctx, "a")
		if again[0].Text() != "orig" {
			t.Fatalf("store aliased caller memory: %q", again[0].Text())
		}
	})

	t.Run("ReplaceDelete", func(t *testing.T) {
		s := newStore()
		_ = s.Append(ctx, "a", UserText("1"), UserText("2"))
		if err := s.Replace(ctx, "a", []Message{UserText("summary")}); err != nil {
			t.Fatal(err)
		}
		got, _ := s.Load(ctx, "a")
		if len(got) != 1 || got[0].Text() != "summary" {
			t.Fatalf("Replace: %+v", got)
		}
		if err := s.Delete(ctx, "a"); err != nil {
			t.Fatal(err)
		}
		got, _ = s.Load(ctx, "a")
		if len(got) != 0 {
			t.Fatal("Delete did not clear")
		}
	})

	t.Run("Concurrent", func(t *testing.T) {
		s := newStore()
		var wg sync.WaitGroup
		for i := range 20 {
			wg.Go(func() {
				_ = s.Append(ctx, "c", UserText(fmt.Sprint(i)))
				_, _ = s.Load(ctx, "c")
			})
		}
		wg.Wait()
		got, _ := s.Load(ctx, "c")
		if len(got) != 20 {
			t.Fatalf("got %d messages", len(got))
		}
	})
}
