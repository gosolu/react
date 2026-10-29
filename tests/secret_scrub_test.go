package tests

import (
	"context"
	"strings"
	"testing"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider/mock"
)

// The tool returns output embedding a plain secret, a JSON secret, and a
// base64-encoded secret — the three shapes PR1 must catch before the
// result reaches model history or the session store.
func leakTool() react.Tool {
	return react.NewTool("leak", "returns secret-laden output", func(ctx context.Context, in struct{}) (string, error) {
		return "plain api_key=supersecretvalue; json {\"token\": \"tok_abcdef123456\"}; " +
			"encoded eyJhcGlfa2V5Ijoic3VwZXJzZWNyZXR2YWx1ZSJ9", nil
	})
}

func runLeakAgent(t *testing.T, storeDir string, opts ...react.Option) *react.FileStore {
	t.Helper()
	store, err := react.NewFileStore(storeDir)
	if err != nil {
		t.Fatal(err)
	}
	p := mock.New(
		mock.ToolCalls("", mock.Call("leak", `{}`)),
		mock.Text("done"),
	)
	base := []react.Option{
		react.WithProvider(p),
		react.WithTools(leakTool()),
		react.WithStore(store),
	}
	a, err := react.New(append(base, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Run(context.Background(), "scrub-session", "go"); err != nil {
		t.Fatal(err)
	}
	return store
}

func TestScrubToolSecretsAgentLevel(t *testing.T) {
	store := runLeakAgent(t, t.TempDir(), react.WithScrubToolSecrets())
	msgs, err := store.Load(context.Background(), "scrub-session")
	if err != nil {
		t.Fatal(err)
	}
	var all strings.Builder
	for _, m := range msgs {
		all.WriteString(m.Text())
		all.WriteString("\n")
	}
	persisted := all.String()
	for _, banned := range []string{
		"supersecretvalue",
		"tok_abcdef123456",
		"eyJhcGlfa2V5Ijoic3VwZXJzZWNyZXR2YWx1ZSJ9",
	} {
		if strings.Contains(persisted, banned) {
			t.Fatalf("persisted history still contains %q:\n%s", banned, persisted)
		}
	}
	if !strings.Contains(persisted, "[REDACTED]") {
		t.Fatalf("persisted history has no redaction marker:\n%s", persisted)
	}
}

func TestNoScrubByDefault(t *testing.T) {
	// Control: without the option, output is stored verbatim. This pins
	// the opt-in behaviour so a future default change is a deliberate diff.
	store := runLeakAgent(t, t.TempDir())
	msgs, err := store.Load(context.Background(), "scrub-session")
	if err != nil {
		t.Fatal(err)
	}
	var all strings.Builder
	for _, m := range msgs {
		all.WriteString(m.Text())
	}
	if !strings.Contains(all.String(), "supersecretvalue") {
		t.Fatalf("expected verbatim secret without option, got:\n%s", all.String())
	}
}
