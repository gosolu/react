package react

import (
	"context"
	"strings"
	"testing"
)

func TestScrubSecrets(t *testing.T) {
	cases := []struct {
		name   string
		in     string
		want   string // exact expected output, "" means only "must not contain" is checked
		banned []string
	}{
		{
			name: "plain sk token",
			in:   "key is sk-abcdefghij1234567890 done",
			want: "key is [REDACTED] done",
		},
		{
			name: "json api key",
			in:   `{"api_key": "supersecretvalue", "other": "keep"}`,
			want: `{"api_key": "[REDACTED]", "other": "keep"}`,
		},
		{
			name:   "env assignment",
			in:     "export OPENAI_API_KEY=sk-abcdefghij1234567890",
			banned: []string{"sk-abcdefghij1234567890"},
		},
		{
			name: "bearer header",
			in:   "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789",
			want: "Authorization: [REDACTED]",
		},
		{
			name: "github token",
			in:   "token ghp_abcdefghijklmnopqrstuvwxyz1234567890 end",
			want: "token [REDACTED] end",
		},
		{
			name: "aws key",
			in:   "id AKIAIOSFODNN7EXAMPLE here",
			want: "id [REDACTED] here",
		},
		{
			name: "pem block",
			in:   "cert:\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA7\n-----END RSA PRIVATE KEY-----\nend",
			want: "cert:\n[REDACTED PRIVATE KEY]\nend",
		},
		{
			name: "plain key value",
			in:   "config: password=hunter2, user=bob",
			want: "config: password=[REDACTED], user=bob",
		},
		{
			name: "base64 json secret",
			in:   "blob eyJhcGlfa2V5Ijoic3VwZXJzZWNyZXR2YWx1ZSJ9 end",
			want: "blob [REDACTED] end",
		},
		{
			name: "base64 sk token",
			in:   "blob c2stYWJjZGVmZ2hpajEyMzQ1Njc4OTA= end",
			want: "blob [REDACTED] end",
		},
		{
			name: "base64 token assignment",
			in:   "blob dG9rZW49aHVudGVyMg== end",
			want: "blob [REDACTED] end",
		},
		{
			name: "innocuous base64 untouched",
			in:   "blob aGVsbG8gd29ybGQsIGp1c3QgYSBncmVldGluZw== end",
			want: "blob aGVsbG8gd29ybGQsIGp1c3QgYSBncmVldGluZw== end",
		},
		{
			name: "innocuous text untouched",
			in:   "the quick brown fox jumps over the lazy dog",
			want: "the quick brown fox jumps over the lazy dog",
		},
		{
			name:   "embedded secret in output",
			in:     "output: read /etc/passwd ok; leaked token=ghp_abcdefghijklmnopqrstuvwxyz1234567890 in log",
			banned: []string{"ghp_abcdefghijklmnopqrstuvwxyz1234567890"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ScrubSecrets(tc.in)
			if tc.want != "" && got != tc.want {
				t.Fatalf("ScrubSecrets(%q) = %q, want %q", tc.in, got, tc.want)
			}
			for _, b := range tc.banned {
				if strings.Contains(got, b) {
					t.Fatalf("ScrubSecrets(%q) = %q, still contains %q", tc.in, got, b)
				}
			}
			// Idempotence: scrubbing twice changes nothing.
			if again := ScrubSecrets(got); again != got {
				t.Fatalf("not idempotent: first %q, second %q", got, again)
			}
		})
	}
}

func TestScrubToolResult(t *testing.T) {
	res := TextResult("api_key=supersecretvalue")
	res.Meta = map[string]any{"note": "kept"}
	scrubbed := ScrubToolResult(res)
	if strings.Contains(scrubbed.Text(), "supersecretvalue") {
		t.Fatalf("result still contains secret: %q", scrubbed.Text())
	}
	if scrubbed.Meta["note"] != "kept" {
		t.Fatalf("Meta not preserved: %+v", scrubbed.Meta)
	}
	// Original is untouched.
	if !strings.Contains(res.Text(), "supersecretvalue") {
		t.Fatalf("original result was mutated: %q", res.Text())
	}
}

func TestRedactSecretsMiddleware(t *testing.T) {
	h := RedactSecretsMiddleware()(func(ctx context.Context, call ToolCall) (ToolResult, error) {
		return TextResult("Bearer abcdefghijklmnopqrstuvwxyz0123456789"), nil
	})
	res, err := h(context.Background(), ToolCall{Name: "leak"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(res.Text(), "abcdefghijklmnopqrstuvwxyz0123456789") {
		t.Fatalf("middleware did not scrub: %q", res.Text())
	}
	if !strings.Contains(res.Text(), "Bearer [REDACTED]") {
		t.Fatalf("unexpected scrubbed output: %q", res.Text())
	}
}
