package react

import (
	"context"
	"fmt"
	"strings"
	"unicode/utf8"
)

// Tool names registered by the skill system.
const (
	ActivateToolName = "activate_skill"
	ReadFileToolName = "read_skill_file"
)

const maxSkillFileBytes = 256 * 1024

type activateIn struct {
	Name string `json:"name" desc:"Name of the skill to activate"`
}

type readFileIn struct {
	Name string `json:"name" desc:"Name of the skill"`
	Path string `json:"path" desc:"File path relative to the skill directory"`
}

// Tools returns the activate_skill and read_skill_file tools bound to r.
func (r *SkillRegistry) Tools() []Tool {
	activate := NewTool(ActivateToolName,
		"Load the full instructions of a skill listed in the Skills section of the system prompt.",
		func(ctx context.Context, in activateIn) (ToolResult, error) {
			s, ok := r.Get(in.Name)
			if !ok {
				return ErrorResult(fmt.Sprintf("unknown skill %q; available: %s", in.Name, r.names())), nil
			}
			key := "skill.active." + s.Name
			if rc := RunContextFrom(ctx); rc != nil {
				if _, already := rc.Get(key); already {
					return TextResult(fmt.Sprintf("Skill %q is already active; its instructions are above.", s.Name)), nil
				}
				rc.Set(key, true)
				if len(s.AllowedTools) > 0 {
					AddSkillAllowedTools(rc, s.AllowedTools...)
				}
			}
			body, err := s.Body()
			if err != nil {
				return ToolResult{}, err
			}
			var b strings.Builder
			fmt.Fprintf(&b, "<skill name=%q>\n%s\n</skill>\n", s.Name, body)
			if len(s.AllowedTools) > 0 {
				fmt.Fprintf(&b, "\nAllowed tools for this skill: %s\n", strings.Join(s.AllowedTools, ", "))
			}
			if s.Compatibility != "" {
				fmt.Fprintf(&b, "\nCompatibility: %s\n", s.Compatibility)
			}
			if files, err := s.Files(); err == nil && len(files) > 0 {
				fmt.Fprintf(&b, "\nBundled files (read with %s): %s\n", ReadFileToolName, strings.Join(files, ", "))
			}
			res := TextResult(b.String())
			res.Meta = map[string]any{"skill": s.Name}
			return res, nil
		}, ReadOnly())

	readFile := NewTool(ReadFileToolName,
		"Read a file bundled with a skill.",
		func(_ context.Context, in readFileIn) (ToolResult, error) {
			s, ok := r.Get(in.Name)
			if !ok {
				return ErrorResult(fmt.Sprintf("unknown skill %q", in.Name)), nil
			}
			data, err := s.ReadFile(in.Path)
			if err != nil {
				return ErrorResult(err.Error()), nil
			}
			if !utf8.Valid(data) {
				return ErrorResult(fmt.Sprintf("%s is a binary file (%d bytes)", in.Path, len(data))), nil
			}
			if len(data) > maxSkillFileBytes {
				return TextResult(string(data[:maxSkillFileBytes]) + "\n...[truncated]"), nil
			}
			return TextResult(string(data)), nil
		}, ReadOnly())

	return []Tool{activate, readFile}
}

func (r *SkillRegistry) names() string {
	var ns []string
	for _, s := range r.List() {
		ns = append(ns, s.Name)
	}
	return strings.Join(ns, ", ")
}
