package react

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"testing/fstest"
)

var testFS = fstest.MapFS{
	"skills/pdf/SKILL.md":         {Data: []byte("---\nname: pdf-tools\ndescription: Work with PDFs.\nallowed-tools: [shell]\n---\n# PDF\nUse pdftotext.\n")},
	"skills/pdf/scripts/run.sh":   {Data: []byte("echo hi")},
	"skills/noname/SKILL.md":      {Data: []byte("---\ndescription: dir-named skill\n---\nbody")},
	"skills/bad/SKILL.md":         {Data: []byte("no frontmatter")},
	"skills/notaskill/readme.txt": {Data: []byte("x")},
	"secret.txt":                  {Data: []byte("top secret")},
}

func TestParse(t *testing.T) {
	t.Run("basic", func(t *testing.T) {
		fm, body, err := Parse([]byte("---\r\nname: x\r\ndescription: y\r\n---\r\nhello\r\n"))
		if err != nil || fm.Name != "x" || fm.Description != "y" || body != "hello" {
			t.Fatalf("%+v %q %v", fm, body, err)
		}
	})

	t.Run("rich yaml frontmatter", func(t *testing.T) {
		raw := `---
# Skill definition
name: "data-analyzer"
description: >
  Performs data analysis
  and statistical modeling.
allowed-tools:
  - shell
  - python
  - 'sql-query'
metadata:
  version: 2.1
  category: analytics
  safe_mode: true
author: "Google"
---
# Instructions
Run scripts directly.
`
		fm, body, err := Parse([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if fm.Name != "data-analyzer" {
			t.Fatalf("unexpected name: %s", fm.Name)
		}
		if !strings.Contains(fm.Description, "Performs data analysis and statistical modeling") {
			t.Fatalf("unexpected description: %s", fm.Description)
		}
		if len(fm.AllowedTools) != 3 || fm.AllowedTools[0] != "shell" || fm.AllowedTools[1] != "python" || fm.AllowedTools[2] != "sql-query" {
			t.Fatalf("unexpected allowed tools: %v", fm.AllowedTools)
		}
		if fm.Metadata["version"] != 2.1 || fm.Metadata["category"] != "analytics" || fm.Metadata["safe_mode"] != true {
			t.Fatalf("unexpected metadata: %v", fm.Metadata)
		}
		if fm.Metadata["author"] != "Google" {
			t.Fatalf("expected extra author in metadata, got: %v", fm.Metadata["author"])
		}
		if body != "# Instructions\nRun scripts directly." {
			t.Fatalf("unexpected body: %q", body)
		}
	})
}

func TestRegistryAndTools(t *testing.T) {
	r := NewSkillRegistry()
	err := r.Load(SkillFS(testFS, "skills"))
	if err == nil {
		t.Fatal("expected error for bad skill")
	}
	if r.Len() != 2 {
		t.Fatalf("loaded %d skills", r.Len())
	}
	if _, ok := r.Get("noname"); !ok {
		t.Fatal("dir-name fallback failed")
	}
	cat := r.Catalog()
	if !strings.Contains(cat, "pdf-tools: Work with PDFs.") || !strings.Contains(cat, ActivateToolName) {
		t.Fatal(cat)
	}

	tools := r.Tools()
	activate, readFile := tools[0], tools[1]
	rc := &RunContext{}
	ctx := WithRunContext(context.Background(), rc)

	res, err := activate.Call(ctx, json.RawMessage(`{"name":"pdf-tools"}`))
	if err != nil || res.IsError || !strings.Contains(res.Text(), "Use pdftotext.") || !strings.Contains(res.Text(), "scripts/run.sh") || !strings.Contains(res.Text(), "shell") {
		t.Fatalf("%v %v", res.Text(), err)
	}
	res, _ = activate.Call(ctx, json.RawMessage(`{"name":"pdf-tools"}`))
	if !strings.Contains(res.Text(), "already active") {
		t.Fatal("double activation not detected")
	}
	res, _ = activate.Call(ctx, json.RawMessage(`{"name":"nope"}`))
	if !res.IsError {
		t.Fatal("unknown skill accepted")
	}

	res, _ = readFile.Call(ctx, json.RawMessage(`{"name":"pdf-tools","path":"scripts/run.sh"}`))
	if res.IsError || res.Text() != "echo hi" {
		t.Fatal(res.Text())
	}
	for _, p := range []string{"../../secret.txt", "/../secret.txt", "..", "scripts/../../../secret.txt"} {
		res, _ = readFile.Call(ctx, json.RawMessage(`{"name":"pdf-tools","path":"`+p+`"}`))
		if !res.IsError {
			t.Fatalf("path traversal allowed for %q: %s", p, res.Text())
		}
	}
}
