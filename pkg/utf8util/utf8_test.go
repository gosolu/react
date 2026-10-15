package utf8util

import (
	"testing"
	"unicode/utf8"
)

func TestTruncateString(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		maxBytes int
		want     string
	}{
		{
			name:     "empty string",
			input:    "",
			maxBytes: 10,
			want:     "",
		},
		{
			name:     "maxBytes zero",
			input:    "hello",
			maxBytes: 0,
			want:     "",
		},
		{
			name:     "maxBytes negative",
			input:    "hello",
			maxBytes: -1,
			want:     "",
		},
		{
			name:     "ascii within limit",
			input:    "hello",
			maxBytes: 10,
			want:     "hello",
		},
		{
			name:     "ascii exact limit",
			input:    "hello",
			maxBytes: 5,
			want:     "hello",
		},
		{
			name:     "ascii truncated",
			input:    "hello world",
			maxBytes: 5,
			want:     "hello",
		},
		{
			name:     "accented 2-byte at boundary",
			input:    "café", // c(1)+a(1)+f(1)+é(2) = 5 bytes
			maxBytes: 5,
			want:     "café",
		},
		{
			name:     "accented 2-byte truncated in middle",
			input:    "café",
			maxBytes: 4, // cuts inside é (bytes 3..4)
			want:     "caf",
		},
		{
			name:     "cjk 3-byte exact boundary",
			input:    "你好世界", // each rune is 3 bytes (total 12 bytes)
			maxBytes: 6,
			want:     "你好",
		},
		{
			name:     "cjk 3-byte truncated byte 1 of 3",
			input:    "你好世界",
			maxBytes: 4, // 3 bytes ("你") + 1 byte of "好"
			want:     "你",
		},
		{
			name:     "cjk 3-byte truncated byte 2 of 3",
			input:    "你好世界",
			maxBytes: 5, // 3 bytes ("你") + 2 bytes of "好"
			want:     "你",
		},
		{
			name:     "cjk less than one rune",
			input:    "你好",
			maxBytes: 2,
			want:     "",
		},
		{
			name:     "emoji 4-byte exact boundary",
			input:    "hi😀bye", // hi(2) + 😀(4) + bye(3) = 9 bytes
			maxBytes: 6,
			want:     "hi😀",
		},
		{
			name:     "emoji 4-byte truncated at byte 1",
			input:    "hi😀bye",
			maxBytes: 3, // hi(2) + 1 byte of 😀
			want:     "hi",
		},
		{
			name:     "emoji 4-byte truncated at byte 2",
			input:    "hi😀bye",
			maxBytes: 4, // hi(2) + 2 bytes of 😀
			want:     "hi",
		},
		{
			name:     "emoji 4-byte truncated at byte 3",
			input:    "hi😀bye",
			maxBytes: 5, // hi(2) + 3 bytes of 😀
			want:     "hi",
		},
		{
			name:     "emoji less than one rune",
			input:    "🎉", // 4 bytes
			maxBytes: 3,
			want:     "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := TruncateString(tc.input, tc.maxBytes)
			if got != tc.want {
				t.Fatalf("TruncateString(%q, %d) = %q (len %d), want %q (len %d)",
					tc.input, tc.maxBytes, got, len(got), tc.want, len(tc.want))
			}
			if !utf8.ValidString(got) {
				t.Fatalf("TruncateString result is not valid UTF-8: %q", got)
			}
		})
	}
}

func TestTruncateBytes(t *testing.T) {
	input := []byte("こんにちは世界😀！")
	for limit := 0; limit <= len(input)+2; limit++ {
		got := TruncateBytes(input, limit)
		if len(got) > limit {
			t.Fatalf("TruncateBytes len %d > limit %d", len(got), limit)
		}
		if !utf8.Valid(got) {
			t.Fatalf("TruncateBytes at limit %d produced invalid UTF-8: %v", limit, got)
		}
	}
}
