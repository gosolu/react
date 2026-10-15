package provider

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestAPIErrorUTF8Truncation(t *testing.T) {
	// 200 CJK characters * 3 bytes = 600 bytes (> 500 bytes limit)
	longBody := strings.Repeat("错误", 100) // "错误" is 6 bytes (2 chars)
	err := &APIError{
		Provider:   "test",
		StatusCode: 500,
		Body:       longBody,
	}

	msg := err.Error()
	if !utf8.ValidString(msg) {
		t.Fatalf("APIError message contains invalid UTF-8: %q", msg)
	}
	if !strings.HasSuffix(msg, "...") {
		t.Fatalf("expected suffix '...', got %q", msg)
	}
}
