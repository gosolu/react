package builtin_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gosolu/react/tools/builtin"
)

func TestBuiltinTools(t *testing.T) {
	tmp := t.TempDir()
	ctx := context.Background()

	// 1. Write file
	targetFile := filepath.Join(tmp, "sub", "test.txt")
	writeTool := builtin.WriteFile()
	res, err := writeTool.Call(ctx, json.RawMessage(`{"path":"`+targetFile+`","content":"hello world"}`))
	if err != nil || res.IsError {
		t.Fatalf("write_file failed: %v, text=%s", err, res.Text())
	}

	// 2. Read file
	readTool := builtin.ReadFile()
	res, err = readTool.Call(ctx, json.RawMessage(`{"path":"`+targetFile+`"}`))
	if err != nil || res.IsError || res.Text() != "hello world" {
		t.Fatalf("read_file failed: %v, text=%s", err, res.Text())
	}

	// 3. List dir
	listTool := builtin.ListDir()
	res, err = listTool.Call(ctx, json.RawMessage(`{"path":"`+tmp+`"}`))
	if err != nil || res.IsError || !strings.Contains(res.Text(), "sub") {
		t.Fatalf("list_dir failed: %v, text=%s", err, res.Text())
	}

	// 4. Shell
	shellTool := builtin.Shell()
	res, err = shellTool.Call(ctx, json.RawMessage(`{"command":"echo 'hello shell'"}`))
	if err != nil || res.IsError || !strings.Contains(res.Text(), "hello shell") {
		t.Fatalf("shell failed: %v, text=%s", err, res.Text())
	}

	// 5. Edit file
	editTool := builtin.EditFile()
	res, err = editTool.Call(ctx, json.RawMessage(`{"path":"`+targetFile+`","old_string":"world","new_string":"gopher"}`))
	if err != nil || res.IsError {
		t.Fatalf("edit_file failed: %v, text=%s", err, res.Text())
	}
	res, _ = readTool.Call(ctx, json.RawMessage(`{"path":"`+targetFile+`"}`))
	if res.Text() != "hello gopher" {
		t.Fatalf("expected 'hello gopher', got %q", res.Text())
	}

	// 6. Sandboxing with WithBaseDir
	sandboxedDir := filepath.Join(tmp, "sandbox")
	_ = os.MkdirAll(sandboxedDir, 0755)
	sandboxedTools := builtin.NewFilesystem(sandboxedDir)
	sbRead := sandboxedTools[0]

	// Try reading outside sandbox
	res, err = sbRead.Call(ctx, json.RawMessage(`{"path":"`+targetFile+`"}`))
	if err == nil {
		t.Fatal("expected sandboxed read outside baseDir to fail")
	}
	if !strings.Contains(err.Error(), "outside allowed directory") {
		t.Fatalf("expected 'outside allowed directory' error, got %v", err)
	}

	// Verify annotations
	if !readTool.Spec().Annotations.ReadOnly {
		t.Fatal("read_file should be ReadOnly")
	}
	if !writeTool.Spec().Annotations.Destructive {
		t.Fatal("write_file should be Destructive")
	}
	if !editTool.Spec().Annotations.Destructive {
		t.Fatal("edit_file should be Destructive")
	}
	if !shellTool.Spec().Annotations.Destructive || !shellTool.Spec().Annotations.Sequential {
		t.Fatal("shell should be Destructive and Sequential")
	}

	_ = os.RemoveAll(tmp)
}
