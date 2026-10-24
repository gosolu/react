package react

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
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

func TestAgentSkillsSpecificationParse(t *testing.T) {
	raw := `---
name: pdf-processing
description: Extract PDF text, fill forms, merge files. Use when handling PDFs.
license: Apache-2.0
compatibility: Designed for agent runtime (or similar products)
metadata:
  author: example-org
  version: "1.0"
allowed-tools: Bash(git:*) Bash(jq:*) Read
---
# PDF Processing Instructions
Follow step-by-step guidance.
`
	fm, body, err := Parse([]byte(raw))
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if fm.Name != "pdf-processing" {
		t.Errorf("expected name 'pdf-processing', got %q", fm.Name)
	}
	if fm.Description != "Extract PDF text, fill forms, merge files. Use when handling PDFs." {
		t.Errorf("unexpected description: %q", fm.Description)
	}
	if fm.License != "Apache-2.0" {
		t.Errorf("expected license 'Apache-2.0', got %q", fm.License)
	}
	if fm.Compatibility != "Designed for agent runtime (or similar products)" {
		t.Errorf("unexpected compatibility: %q", fm.Compatibility)
	}
	if len(fm.AllowedTools) != 3 {
		t.Fatalf("expected 3 allowed tools, got %d: %v", len(fm.AllowedTools), fm.AllowedTools)
	}
	if fm.AllowedTools[0] != "Bash(git:*)" || fm.AllowedTools[1] != "Bash(jq:*)" || fm.AllowedTools[2] != "Read" {
		t.Errorf("unexpected allowed tools: %v", fm.AllowedTools)
	}
	if fm.AllowedToolsString() != "Bash(git:*) Bash(jq:*) Read" {
		t.Errorf("unexpected AllowedToolsString: %q", fm.AllowedToolsString())
	}
	if fm.Metadata["author"] != "example-org" || fm.Metadata["version"] != "1.0" {
		t.Errorf("unexpected metadata: %v", fm.Metadata)
	}
	if !strings.Contains(body, "# PDF Processing Instructions") {
		t.Errorf("unexpected body: %q", body)
	}
	if err := fm.Validate(); err != nil {
		t.Errorf("expected frontmatter to be valid, got: %v", err)
	}
}

func TestValidateSpecification(t *testing.T) {
	validNames := []string{"pdf-processing", "data-analysis", "code-review", "tool1", "a"}
	for _, n := range validNames {
		if err := ValidateSkillName(n); err != nil {
			t.Errorf("expected %q to be valid, got: %v", n, err)
		}
	}

	invalidNames := []string{
		"PDF-Processing",        // uppercase
		"-pdf",                  // starts with hyphen
		"pdf-",                  // ends with hyphen
		"pdf--proc",             // consecutive hyphens
		"pdf_proc",              // underscore not allowed
		"",                      // empty
		strings.Repeat("a", 65), // > 64 chars
	}
	for _, n := range invalidNames {
		if err := ValidateSkillName(n); err == nil {
			t.Errorf("expected %q to be invalid", n)
		}
	}

	// Validate description constraints
	fm := Frontmatter{Name: "valid-name", Description: ""}
	if err := fm.Validate(); err == nil {
		t.Error("expected error for empty description")
	}

	fm.Description = strings.Repeat("x", 1025)
	if err := fm.Validate(); err == nil {
		t.Error("expected error for description > 1024 chars")
	}

	fm.Description = "Valid description"
	fm.Compatibility = strings.Repeat("c", 501)
	if err := fm.Validate(); err == nil {
		t.Error("expected error for compatibility > 500 chars")
	}

	fm.Compatibility = "Valid compatibility"
	if err := fm.Validate(); err != nil {
		t.Errorf("expected valid, got: %v", err)
	}
}

func TestAllowedToolsParsingAndJSON(t *testing.T) {
	// JSON unmarshaling where allowed-tools is a space-separated string
	jsonStr := `{"name":"test-skill","description":"desc","allowed-tools":"Bash(git:*) Read"}`
	var fm Frontmatter
	if err := json.Unmarshal([]byte(jsonStr), &fm); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if len(fm.AllowedTools) != 2 || fm.AllowedTools[0] != "Bash(git:*)" || fm.AllowedTools[1] != "Read" {
		t.Fatalf("unexpected unmarshaled tools from string: %v", fm.AllowedTools)
	}

	// JSON unmarshaling where allowed-tools is an array
	jsonArray := `{"name":"test-skill","description":"desc","allowed-tools":["Bash(git:*)","Read"]}`
	var fm2 Frontmatter
	if err := json.Unmarshal([]byte(jsonArray), &fm2); err != nil {
		t.Fatalf("json.Unmarshal array failed: %v", err)
	}
	if len(fm2.AllowedTools) != 2 || fm2.AllowedTools[0] != "Bash(git:*)" || fm2.AllowedTools[1] != "Read" {
		t.Fatalf("unexpected unmarshaled tools from array: %v", fm2.AllowedTools)
	}
}

func TestAllowedToolsPermissionPreApproval(t *testing.T) {
	skillFS := fstest.MapFS{
		"skills/git-helper/SKILL.md": {Data: []byte(`---
name: git-helper
description: Assists with git workflows.
allowed-tools: GitTool(git:*) SafeHelper DangerousTool
---
# Git Helper
Use git safely.
`)},
	}

	reg := NewSkillRegistry()
	if err := reg.Load(SkillFS(skillFS, "skills")); err != nil {
		t.Fatalf("failed to load skill: %v", err)
	}

	skill, ok := reg.Get("git-helper")
	if !ok {
		t.Fatal("skill git-helper not found")
	}
	if err := skill.Validate(); err != nil {
		t.Fatalf("skill validation failed: %v", err)
	}

	// Non-destructive tools gated by StrictPermissionPolicy
	safeHelperSpec := ToolSpec{
		Name:        "SafeHelper",
		Description: "A non-destructive helper tool",
	}
	gitToolSpec := ToolSpec{
		Name:        "GitTool",
		Description: "Run non-destructive git subcommands",
	}
	// Destructive tool specification
	dangerousSpec := ToolSpec{
		Name:        "DangerousTool",
		Description: "A destructive tool",
		Annotations: ToolAnnotations{Destructive: true},
	}

	policy := StrictPermissionPolicy()
	rc := &RunContext{}
	ctx := WithRunContext(context.Background(), rc)

	// Before activating the skill:
	// All non-ReadOnly tools require approval (PolicyAsk) under StrictPermissionPolicy
	callHelper := ToolCall{ID: "c1", Name: "SafeHelper", Arguments: json.RawMessage(`{}`)}
	if dec := policy.Check(ctx, safeHelperSpec, callHelper); dec != PolicyAsk {
		t.Fatalf("expected PolicyAsk before skill activation, got %s", dec)
	}

	callGit := ToolCall{ID: "c2", Name: "GitTool", Arguments: json.RawMessage(`{"command":"git status"}`)}
	if dec := policy.Check(ctx, gitToolSpec, callGit); dec != PolicyAsk {
		t.Fatalf("expected PolicyAsk for GitTool before skill activation, got %s", dec)
	}

	callDanger := ToolCall{ID: "c3", Name: "DangerousTool", Arguments: json.RawMessage(`{}`)}
	if dec := policy.Check(ctx, dangerousSpec, callDanger); dec != PolicyAsk {
		t.Fatalf("expected PolicyAsk for DangerousTool before skill activation, got %s", dec)
	}

	// Activate the skill via activate_skill tool
	tools := reg.Tools()
	activate := tools[0]
	res, err := activate.Call(ctx, json.RawMessage(`{"name":"git-helper"}`))
	if err != nil || res.IsError {
		t.Fatalf("activate_skill error: %v, res: %s", err, res.Text())
	}

	// After activation: non-destructive SafeHelper is pre-approved (PolicyAllow)
	if dec := policy.Check(ctx, safeHelperSpec, callHelper); dec != PolicyAllow {
		t.Fatalf("expected SafeHelper to be PolicyAllow after skill activation, got %s", dec)
	}

	// GitTool(git:*) is pre-approved for git commands
	if dec := policy.Check(ctx, gitToolSpec, callGit); dec != PolicyAllow {
		t.Fatalf("expected GitTool(git status) to be PolicyAllow after skill activation, got %s", dec)
	}

	// GitTool with non-git command must NOT be pre-approved (still PolicyAsk)
	callOther := ToolCall{ID: "c4", Name: "GitTool", Arguments: json.RawMessage(`{"command":"rm -rf /tmp/foo"}`)}
	if dec := policy.Check(ctx, gitToolSpec, callOther); dec != PolicyAsk {
		t.Fatalf("expected GitTool(rm -rf) to remain PolicyAsk, got %s", dec)
	}

	// CRITICAL (CRIT-01): Destructive tools MUST NEVER be pre-approved by an active skill!
	if dec := policy.Check(ctx, dangerousSpec, callDanger); dec != PolicyAsk {
		t.Fatalf("expected Destructive DangerousTool to remain PolicyAsk even when listed in skill allowed-tools, got %s", dec)
	}
	if dec := DefaultPermissionPolicy().Check(ctx, dangerousSpec, callDanger); dec != PolicyAsk {
		t.Fatalf("expected DefaultPermissionPolicy to keep Destructive DangerousTool as PolicyAsk, got %s", dec)
	}
}

func TestSkillDirectoryHelpers(t *testing.T) {
	fsys := fstest.MapFS{
		"my-skill/SKILL.md":             {Data: []byte("---\nname: my-skill\ndescription: A test skill\n---\nbody")},
		"my-skill/scripts/extract.py":   {Data: []byte("print(1)")},
		"my-skill/scripts/merge.sh":     {Data: []byte("echo 1")},
		"my-skill/references/GUIDE.md":  {Data: []byte("# Guide")},
		"my-skill/references/FORMS.md":  {Data: []byte("# Forms")},
		"my-skill/assets/template.json": {Data: []byte("{}")},
		"my-skill/extra.txt":            {Data: []byte("extra")},
	}

	skill, err := NewSkill(fsys, "my-skill")
	if err != nil {
		t.Fatalf("NewSkill failed: %v", err)
	}

	scripts, err := skill.Scripts()
	if err != nil || len(scripts) != 2 {
		t.Errorf("expected 2 scripts, got %d: %v (err: %v)", len(scripts), scripts, err)
	}

	references, err := skill.References()
	if err != nil || len(references) != 2 {
		t.Errorf("expected 2 references, got %d: %v (err: %v)", len(references), references, err)
	}

	assets, err := skill.Assets()
	if err != nil || len(assets) != 1 {
		t.Errorf("expected 1 asset, got %d: %v (err: %v)", len(assets), assets, err)
	}

	dirs := StandardSkillDirs("/my/project")
	if len(dirs) == 0 || !strings.HasSuffix(dirs[0], path.Join(".agents", "skills")) {
		t.Errorf("unexpected StandardSkillDirs: %v", dirs)
	}
}

func TestAddSkillAllowedToolsConcurrent(t *testing.T) {
	rc := &RunContext{}
	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)
	for i := range n {
		go func(idx int) {
			defer wg.Done()
			AddSkillAllowedTools(rc, fmt.Sprintf("tool_%d", idx))
		}(i)
	}
	wg.Wait()

	ctx := WithRunContext(context.Background(), rc)
	allowed := ActiveSkillAllowedTools(ctx)
	if len(allowed) != n {
		t.Fatalf("expected all %d tools to be retained without race loss, got %d", n, len(allowed))
	}
}

func TestSkillReadFileSymlinkTraversalBlocked(t *testing.T) {
	outerDir := t.TempDir()
	secretFile := filepath.Join(outerDir, "secret.txt")
	if err := os.WriteFile(secretFile, []byte("top-secret-host-file"), 0o600); err != nil {
		t.Fatal(err)
	}

	skillsRoot := filepath.Join(outerDir, "skills")
	skillDir := filepath.Join(skillsRoot, "my-skill")
	if err := os.MkdirAll(skillDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte("---\nname: my-skill\ndescription: test\n---\nbody"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "safe.txt"), []byte("safe-content"), 0o600); err != nil {
		t.Fatal(err)
	}
	// Create a symlink inside skillDir pointing outside to secretFile
	if err := os.Symlink(secretFile, filepath.Join(skillDir, "leak.txt")); err != nil {
		t.Skipf("symlink not supported on this filesystem: %v", err)
	}

	reg := NewSkillRegistry()
	if err := reg.Load(SkillDir(skillsRoot)); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	s, ok := reg.Get("my-skill")
	if !ok {
		t.Fatal("my-skill not found")
	}

	// Reading safe.txt succeeds
	data, err := s.ReadFile("safe.txt")
	if err != nil || string(data) != "safe-content" {
		t.Fatalf("expected safe.txt to be readable, got %q, err %v", string(data), err)
	}

	// Reading symlink leak.txt pointing outside skillDir MUST be blocked
	if _, err := s.ReadFile("leak.txt"); err == nil {
		t.Fatal("expected symlink escaping skill root to be rejected")
	}
}
