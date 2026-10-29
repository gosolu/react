package tools_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gosolu/react"
	"github.com/gosolu/react/tools"
)

func callTool(t *testing.T, tool react.Tool, args any) (react.ToolResult, error) {
	t.Helper()
	raw, err := json.Marshal(args)
	if err != nil {
		t.Fatal(err)
	}
	return tool.Call(context.Background(), raw)
}

func TestReadFileTool(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "hello.txt")
	if err := os.WriteFile(p, []byte("hello world"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := callTool(t, tools.ReadFileTool(), map[string]any{"path": p})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text() != "hello world" {
		t.Fatalf("read_file = %q, want %q", res.Text(), "hello world")
	}
	if _, err := callTool(t, tools.ReadFileTool(), map[string]any{"path": filepath.Join(dir, "missing.txt")}); err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestWriteFileTool(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "out.txt")
	res, err := callTool(t, tools.WriteFileTool(), map[string]any{"path": p, "content": "written"})
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "written" {
		t.Fatalf("file = %q, want %q", data, "written")
	}
	if !strings.Contains(res.Text(), "wrote 7 bytes") {
		t.Fatalf("unexpected write_file result: %q", res.Text())
	}
}

func TestExecTool(t *testing.T) {
	res, err := callTool(t, tools.ExecTool(), map[string]any{"command": "echo hello"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(res.Text()) != "hello" {
		t.Fatalf("exec = %q, want %q", res.Text(), "hello")
	}
	if _, err := callTool(t, tools.ExecTool(), map[string]any{"command": "exit 3"}); err == nil {
		t.Fatal("expected error for failing command")
	} else if !strings.Contains(err.Error(), "exit status 3") {
		t.Fatalf("unexpected exec error: %v", err)
	}
	if _, err := callTool(t, tools.ExecTool(), map[string]any{"command": ""}); err == nil {
		t.Fatal("expected error for empty command")
	}
}

func TestToolsAnnotations(t *testing.T) {
	read := tools.ReadFileTool().Spec().Annotations
	if !read.ReadOnly || read.Destructive {
		t.Fatalf("read_file annotations = %+v, want ReadOnly only", read)
	}
	write := tools.WriteFileTool().Spec().Annotations
	if !write.Destructive {
		t.Fatalf("write_file annotations = %+v, want Destructive", write)
	}
	ex := tools.ExecTool().Spec().Annotations
	if !ex.Destructive || !ex.Sequential {
		t.Fatalf("exec annotations = %+v, want Destructive + Sequential", ex)
	}
}

func TestAllTools(t *testing.T) {
	all := tools.All()
	if len(all) != 3 {
		t.Fatalf("All() = %d tools, want 3", len(all))
	}
	want := []string{"read_file", "write_file", "exec"}
	for i, tool := range all {
		if tool.Spec().Name != want[i] {
			t.Fatalf("All()[%d] = %q, want %q", i, tool.Spec().Name, want[i])
		}
	}
}
