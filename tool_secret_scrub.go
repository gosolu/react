package react

import (
	"encoding/base64"
	"regexp"
	"strings"
)

// ScrubSecrets redacts secret-looking material from free-form tool output
// text. It is deliberately pattern-based, not a secret detector: anything
// matching one of the patterns below is replaced with [REDACTED], even if
// the value turns out to be fake. Callers that need exactness should use
// structured redaction at the source instead.
//
// Covered shapes:
//   - PEM private-key blocks.
//   - Bearer tokens and well-known token formats (sk-*, GitHub gh*_*, AWS
//     AKIA keys, Slack xox*, GitLab glpat-*).
//   - Sensitive key/value pairs, JSON ("api_key": "...") and plain
//     (api_key=..., token: ...) alike.
//   - Base64 (standard and URL-safe) blobs that decode to text containing
//     a sensitive key or a token format above. Ordinary base64 that decodes
//     to innocuous text is left untouched.
//
// ScrubSecrets is idempotent: scrubbed output contains no pattern that
// would be scrubbed again.
func ScrubSecrets(s string) string {
	if s == "" {
		return s
	}
	out := pemBlockRe.ReplaceAllString(s, "[REDACTED PRIVATE KEY]")
	// Authorization headers first: the generic key/value pass below would
	// otherwise treat "Bearer" itself as the value of "Authorization" and
	// leave the real token behind.
	out = authHeaderRe.ReplaceAllString(out, "${1}[REDACTED]")
	out = bearerRe.ReplaceAllString(out, "Bearer [REDACTED]")
	out = tokenFormatRe.ReplaceAllString(out, "[REDACTED]")
	out = jsonSecretValueRe.ReplaceAllString(out, "${1}[REDACTED]${2}")
	out = plainSecretValueRe.ReplaceAllString(out, "${1}[REDACTED]")
	out = base64CandidateRe.ReplaceAllStringFunc(out, func(tok string) string {
		if decodesToSecret(tok) {
			return "[REDACTED]"
		}
		return tok
	})
	return out
}

// ScrubToolResult returns a copy of res with every text part scrubbed.
// Non-text parts, IsError, Truncated and Meta pass through unchanged.
func ScrubToolResult(res ToolResult) ToolResult {
	out := res
	if len(res.Content) == 0 {
		return out
	}
	content := make([]Part, len(res.Content))
	copy(content, res.Content)
	for i, p := range content {
		if p.Type == PartText {
			content[i].Text = ScrubSecrets(p.Text)
		}
	}
	out.Content = content
	return out
}

var (
	pemBlockRe = regexp.MustCompile(`-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----`)

	// "Authorization: Bearer <token>" (or "Authorization: <token>") as one
	// unit, so the generic key/value pass cannot split it.
	authHeaderRe = regexp.MustCompile(`(?i)(\bAuthorization\s*[:=]\s*)(?:Bearer\s+)?[^\s,;"'}]+`)

	bearerRe = regexp.MustCompile(`(?i)\bBearer\s+[A-Za-z0-9\-._~+/=]{8,}`)

	// Distinctive token prefixes; the trailing length bound keeps short
	// identifiers such as "sketch" out while still catching real tokens.
	tokenFormatRe = regexp.MustCompile(`\b(?:sk-[A-Za-z0-9\-_]{8,}|gh[opsu]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9\-]{8,}|glpat-[A-Za-z0-9\-_]{8,})\b`)

	sensitiveKey = `(?:password|passwd|pwd|secret|token|api[_-]?key|apikey|authorization|credential|private[_-]?key)`

	jsonSecretValueRe = regexp.MustCompile(`(?i)("(?:` + sensitiveKey + `)"\s*:\s*")[^"]*(")`)

	plainSecretValueRe = regexp.MustCompile(`(?i)\b((?:` + sensitiveKey + `)\s*[:=]\s*)[^\s,;"'}]+`)

	// No trailing word boundary: a padded token ends in "=", a non-word
	// character, so \b would force the match to drop the padding and the
	// remainder would fail to decode. The 12-character minimum plus the
	// decode-and-inspect check in decodesToSecret keeps ordinary words
	// and hex strings untouched.
	base64CandidateRe = regexp.MustCompile(`\b[A-Za-z0-9+/]{12,}={0,2}`)
)

// decodesToSecret reports whether tok is a base64 blob whose decoded text
// looks secret-bearing: it contains a sensitive key/value pair or a known
// token format. Undecodable tokens and innocuous payloads return false.
func decodesToSecret(tok string) bool {
	for _, enc := range []*base64.Encoding{
		base64.StdEncoding, base64.RawStdEncoding,
		base64.URLEncoding, base64.RawURLEncoding,
	} {
		decoded, err := enc.DecodeString(tok)
		if err != nil || len(decoded) == 0 {
			continue
		}
		text := string(decoded)
		if !isMostlyPrintable(text) {
			continue
		}
		lower := strings.ToLower(text)
		if strings.Contains(lower, "password") || strings.Contains(lower, "secret") ||
			strings.Contains(lower, "token") || strings.Contains(lower, "api_key") ||
			strings.Contains(lower, "apikey") || strings.Contains(lower, "authorization") ||
			strings.Contains(lower, "credential") || strings.Contains(lower, "private_key") ||
			tokenFormatRe.MatchString(text) || bearerRe.MatchString(text) {
			return true
		}
	}
	return false
}

func isMostlyPrintable(s string) bool {
	if s == "" {
		return false
	}
	printable := 0
	for _, r := range s {
		if r == '\n' || r == '\r' || r == '\t' || (r >= 0x20 && r < 0x7f) {
			printable++
		}
	}
	return float64(printable)/float64(len([]rune(s))) >= 0.9
}
