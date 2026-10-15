// Package utf8util provides utility functions for UTF-8 encoded text and byte slices.
package utf8util

import "unicode/utf8"

// TruncateString truncates s to at most maxBytes, ensuring it does not split
// a multi-byte UTF-8 rune. If len(s) <= maxBytes, s is returned unchanged.
func TruncateString(s string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(s) <= maxBytes {
		return s
	}
	for maxBytes > 0 && !utf8.RuneStart(s[maxBytes]) {
		maxBytes--
	}
	return s[:maxBytes]
}

// TruncateBytes truncates b to at most maxBytes, ensuring it does not split
// a multi-byte UTF-8 rune. If len(b) <= maxBytes, b is returned unchanged.
func TruncateBytes(b []byte, maxBytes int) []byte {
	if maxBytes <= 0 {
		return b[:0]
	}
	if len(b) <= maxBytes {
		return b
	}
	for maxBytes > 0 && !utf8.RuneStart(b[maxBytes]) {
		maxBytes--
	}
	return b[:maxBytes]
}
