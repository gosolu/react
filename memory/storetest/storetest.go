// Package storetest provides a conformance suite for memory.Store
// implementations.
package storetest

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/memory"
)

// Run exercises a Store implementation. newStore must return a fresh, empty store.
func Run(t *testing.T, newStore func() memory.Store) {
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
		in := []llm.Message{
			llm.UserText("hi"),
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "t", Arguments: []byte(`{"x":1}`)}}},
			llm.ToolResult("c1", "t", []llm.Part{llm.TextPart("ok")}, true),
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
		m := llm.UserText("orig")
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
		_ = s.Append(ctx, "a", llm.UserText("1"), llm.UserText("2"))
		if err := s.Replace(ctx, "a", []llm.Message{llm.UserText("summary")}); err != nil {
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
			wg.Add(1)
			go func() {
				defer wg.Done()
				_ = s.Append(ctx, "c", llm.UserText(fmt.Sprint(i)))
				_, _ = s.Load(ctx, "c")
			}()
		}
		wg.Wait()
		got, _ := s.Load(ctx, "c")
		if len(got) != 20 {
			t.Fatalf("got %d messages", len(got))
		}
	})
}
