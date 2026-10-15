package tests

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider/mock"
)

func TestLoopToolOutputUTF8Truncation(t *testing.T) {
	ctx := context.Background()

	// Tool returning a long CJK string
	// "日本語テキスト" is 7 characters * 3 bytes = 21 bytes. Repeat 10 times = 210 bytes.
	cjkTool := react.NewTool("cjk_tool", "returns cjk", func(ctx context.Context, in struct{}) (string, error) {
		return strings.Repeat("日本語テキスト", 10), nil
	})

	p := mock.New(
		mock.ToolCalls("calling...", mock.Call("cjk_tool", `{}`)),
		mock.Text("Done"),
	)

	// Set maxToolOutput to 31 bytes (which falls in the middle of a 3-byte CJK rune: 31 % 3 = 1)
	a, err := react.New(
		react.WithProvider(p),
		react.WithTools(cjkTool),
		react.WithMaxToolOutput(31),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := a.Run(ctx, "cjk-sess", "run cjk")
	if err != nil {
		t.Fatal(err)
	}

	for _, m := range res.Messages {
		if m.Role == react.RoleTool {
			txt := m.Text()
			if !utf8.ValidString(txt) {
				t.Fatalf("tool message contains invalid UTF-8: %q", txt)
			}
			if !strings.Contains(txt, "...[output truncated from 210 bytes]") {
				t.Fatalf("expected truncation marker, got: %s", txt)
			}
			// Sliced length before marker should be 30 bytes (10 CJK characters)
			parts := strings.Split(txt, "\n...[output truncated from 210 bytes]")
			if len(parts[0]) != 30 {
				t.Fatalf("expected 30 bytes before marker, got %d bytes: %q", len(parts[0]), parts[0])
			}
		}
	}
}

func TestTruncateMiddlewareUTF8(t *testing.T) {
	mw := react.TruncateMiddleware(10) // 10 bytes limit
	handler := mw(func(ctx context.Context, call react.ToolCall) (react.ToolResult, error) {
		// "🎉🎊🎈" = 3 emojis * 4 bytes = 12 bytes
		return react.TextResult("🎉🎊🎈"), nil
	})

	res, err := handler(context.Background(), react.ToolCall{ID: "1", Name: "test"})
	if err != nil {
		t.Fatal(err)
	}

	txt := res.Text()
	if !utf8.ValidString(txt) {
		t.Fatalf("TruncateMiddleware result is not valid UTF-8: %q", txt)
	}
	// 10 bytes limit: 2 emojis = 8 bytes. Third emoji doesn't fit (8+4=12 > 10).
	// Truncated text should be "🎉🎊" (8 bytes), truncated bytes = 12 - 8 = 4 bytes.
	expectedPrefix := "🎉🎊\n...[truncated 4 bytes]"
	if txt != expectedPrefix {
		t.Fatalf("got %q, want %q", txt, expectedPrefix)
	}
}

func TestRenderUTF8Truncation(t *testing.T) {
	// Tool message with CJK text
	toolMsg := react.ToolResultMessage("c1", "test", []react.Part{
		react.TextPart("你好世界"), // 12 bytes
	}, false)

	// maxResult = 7 (cuts in middle of 3rd CJK rune "世" at bytes 6..8)
	rendered := react.Render([]react.Message{toolMsg}, 7)
	if !utf8.ValidString(rendered) {
		t.Fatalf("rendered transcript contains invalid UTF-8: %q", rendered)
	}
	// 7 bytes should truncate to "你好" (6 bytes) + "...[truncated]"
	if !strings.Contains(rendered, "你好...[truncated]") {
		t.Fatalf("expected '你好...[truncated]' in %q", rendered)
	}
}
