package react

import (
	"context"
	"strings"
	"testing"
)

func TestFileStoreCacheCoherence(t *testing.T) {
	ctx := context.Background()
	fs, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := fs.Append(ctx, "s", UserText("one")); err != nil {
		t.Fatal(err)
	}
	if err := fs.Append(ctx, "s", UserText("two")); err != nil {
		t.Fatal(err)
	}
	msgs, err := fs.Load(ctx, "s")
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 || msgs[0].Text() != "one" || msgs[1].Text() != "two" {
		t.Fatalf("cached load = %v, want [one two]", msgs)
	}
	// A fresh instance over the same directory sees the same history:
	// the cache never diverges from what was written.
	fs2, err := NewFileStore(fs.dir)
	if err != nil {
		t.Fatal(err)
	}
	msgs2, err := fs2.Load(ctx, "s")
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs2) != 2 {
		t.Fatalf("fresh instance load = %d messages, want 2", len(msgs2))
	}
}

func TestFileStoreCacheDoesNotAlias(t *testing.T) {
	ctx := context.Background()
	fs, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := fs.Append(ctx, "s", UserText("original")); err != nil {
		t.Fatal(err)
	}
	msgs, err := fs.Load(ctx, "s")
	if err != nil {
		t.Fatal(err)
	}
	msgs[0].Content[0].Text = "mutated"
	again, err := fs.Load(ctx, "s")
	if err != nil {
		t.Fatal(err)
	}
	if again[0].Text() != "original" {
		t.Fatalf("Load returned aliased cache: %q", again[0].Text())
	}
}

func TestFileStoreCacheReplaceAndDelete(t *testing.T) {
	ctx := context.Background()
	fs, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := fs.Append(ctx, "s", UserText("old")); err != nil {
		t.Fatal(err)
	}
	if err := fs.Replace(ctx, "s", []Message{UserText("new")}); err != nil {
		t.Fatal(err)
	}
	msgs, err := fs.Load(ctx, "s")
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 1 || msgs[0].Text() != "new" {
		t.Fatalf("after Replace, load = %v", msgs)
	}
	if err := fs.Delete(ctx, "s"); err != nil {
		t.Fatal(err)
	}
	msgs, err = fs.Load(ctx, "s")
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 0 {
		t.Fatalf("after Delete, load = %v", msgs)
	}
}

func TestFileStoreCacheInvalidate(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	fs, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := fs.Append(ctx, "s", UserText("cached")); err != nil {
		t.Fatal(err)
	}
	// A second instance rewrites the session out from under the first.
	fs2, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := fs2.Replace(ctx, "s", []Message{UserText("external")}); err != nil {
		t.Fatal(err)
	}
	// Documented staleness: the first instance still serves its cache.
	msgs, err := fs.Load(ctx, "s")
	if err != nil {
		t.Fatal(err)
	}
	if msgs[0].Text() != "cached" {
		t.Fatalf("expected stale cached value, got %q", msgs[0].Text())
	}
	fs.Invalidate("s")
	msgs, err = fs.Load(ctx, "s")
	if err != nil {
		t.Fatal(err)
	}
	if msgs[0].Text() != "external" {
		t.Fatalf("after Invalidate, load = %q, want external", msgs[0].Text())
	}
}

// seedFileStoreSession builds a session with n messages of ~200 bytes each.
func seedFileStoreSession(tb testing.TB, fs *FileStore, id string, n int) {
	tb.Helper()
	msgs := make([]Message, n)
	body := strings.Repeat("lorem ipsum dolor sit amet ", 8) // ~216 bytes
	for i := range msgs {
		msgs[i] = UserText(body)
	}
	if err := fs.Replace(context.Background(), id, msgs); err != nil {
		tb.Fatal(err)
	}
}

func BenchmarkFileStoreAppendCached(b *testing.B) {
	fs, err := NewFileStore(b.TempDir())
	if err != nil {
		b.Fatal(err)
	}
	seedFileStoreSession(b, fs, "bench", 2000)
	// Warm the cache: one Load so every measured Append is a cache hit.
	if _, err := fs.Load(context.Background(), "bench"); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := fs.Append(context.Background(), "bench", UserText("turn")); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFileStoreAppendCold(b *testing.B) {
	dir := b.TempDir()
	seed, err := NewFileStore(dir)
	if err != nil {
		b.Fatal(err)
	}
	seedFileStoreSession(b, seed, "bench", 2000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// A fresh instance has an empty cache, so every Append pays
		// the disk read + JSON parse of the full history first.
		fs, err := NewFileStore(dir)
		if err != nil {
			b.Fatal(err)
		}
		if err := fs.Append(context.Background(), "bench", UserText("turn")); err != nil {
			b.Fatal(err)
		}
	}
}
