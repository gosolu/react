// Skill implements SKILL.md-based skills: folders containing a
// SKILL.md file (YAML frontmatter + markdown instructions) and optional
// resources. Only the name/description catalog is advertised to the model up
// front; the full instructions are loaded on demand via the activate_skill
// tool (progressive disclosure).
package react

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"path"
	"strconv"
	"strings"
)

// FileName is the required skill manifest name.
const FileName = "SKILL.md"

// Frontmatter is the YAML header of a SKILL.md file.
type Frontmatter struct {
	Name          string         `json:"name" yaml:"name"`
	Description   string         `json:"description" yaml:"description"`
	License       string         `json:"license,omitempty" yaml:"license,omitempty"`
	Compatibility string         `json:"compatibility,omitempty" yaml:"compatibility,omitempty"`
	AllowedTools  []string       `json:"allowed-tools,omitempty" yaml:"allowed-tools,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty" yaml:"metadata,omitempty"`
}

// AllowedToolsString returns allowed tools as a space-separated string per the specification.
func (fm Frontmatter) AllowedToolsString() string {
	return strings.Join(fm.AllowedTools, " ")
}

// UnmarshalJSON unmarshals Frontmatter supporting allowed-tools as either a string or string array.
func (fm *Frontmatter) UnmarshalJSON(data []byte) error {
	type rawFrontmatter Frontmatter
	var raw struct {
		rawFrontmatter
		RawAllowedTools any `json:"allowed-tools"`
		AltAllowedTools any `json:"allowed_tools"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*fm = Frontmatter(raw.rawFrontmatter)
	if raw.RawAllowedTools != nil {
		fm.AllowedTools = parseAllowedTools(raw.RawAllowedTools)
	} else if raw.AltAllowedTools != nil {
		fm.AllowedTools = parseAllowedTools(raw.AltAllowedTools)
	}
	return nil
}

// Validate checks if the frontmatter satisfies the Agent Skills specification.
func (fm Frontmatter) Validate() error {
	if err := ValidateSkillName(fm.Name); err != nil {
		return err
	}
	if len(fm.Description) == 0 {
		return errors.New("description is required")
	}
	if len(fm.Description) > 1024 {
		return fmt.Errorf("description exceeds 1024 characters (%d)", len(fm.Description))
	}
	if len(fm.Compatibility) > 500 {
		return fmt.Errorf("compatibility exceeds 500 characters (%d)", len(fm.Compatibility))
	}
	return nil
}

// ValidateSkillName checks if name satisfies the Agent Skills specification constraints:
// 1-64 characters, lowercase alphanumeric and hyphens, no leading/trailing hyphens, no consecutive hyphens.
func ValidateSkillName(name string) error {
	if len(name) < 1 || len(name) > 64 {
		return fmt.Errorf("skill name must be 1-64 characters, got %d", len(name))
	}
	if strings.HasPrefix(name, "-") || strings.HasSuffix(name, "-") {
		return errors.New("skill name must not start or end with a hyphen")
	}
	if strings.Contains(name, "--") {
		return errors.New("skill name must not contain consecutive hyphens")
	}
	for _, r := range name {
		if (r < 'a' || r > 'z') && (r < '0' || r > '9') && r != '-' {
			return fmt.Errorf("skill name must only contain lowercase alphanumeric characters and hyphens, invalid character: %c", r)
		}
	}
	return nil
}

// Skill is a discovered skill. Its body is read lazily.
type Skill struct {
	Frontmatter
	fsys fs.FS
	dir  string // directory within fsys ("." for root)
}

// New constructs a skill rooted at dir within fsys, parsing its SKILL.md.
func NewSkill(fsys fs.FS, dir string) (*Skill, error) {
	data, err := fs.ReadFile(fsys, path.Join(dir, FileName))
	if err != nil {
		return nil, err
	}
	fm, _, err := Parse(data)
	if err != nil {
		return nil, fmt.Errorf("skill %s: %w", dir, err)
	}
	if fm.Name == "" {
		fm.Name = path.Base(dir)
	}
	if fm.Description == "" {
		return nil, fmt.Errorf("skill %s: description is required", fm.Name)
	}
	return &Skill{Frontmatter: fm, fsys: fsys, dir: dir}, nil
}

// Parse splits a SKILL.md document into frontmatter and body.
func Parse(data []byte) (Frontmatter, string, error) {
	var fm Frontmatter
	data = bytes.TrimPrefix(data, []byte("\ufeff"))
	s := strings.ReplaceAll(string(data), "\r\n", "\n")
	if !strings.HasPrefix(s, "---\n") {
		return fm, strings.TrimSpace(s), errors.New("missing YAML frontmatter")
	}
	rest := s[4:]
	header, body, ok := strings.Cut(rest, "\n---")
	if !ok {
		return fm, "", errors.New("unterminated YAML frontmatter")
	}
	if i := strings.IndexByte(body, '\n'); i >= 0 {
		body = body[i+1:]
	} else {
		body = ""
	}
	parsed, err := parseYAMLHeader(header)
	if err != nil {
		return fm, "", fmt.Errorf("invalid frontmatter: %w", err)
	}
	return parsed, strings.TrimSpace(body), nil
}

// Body returns the instructions (SKILL.md without frontmatter).
func (s *Skill) Body() (string, error) {
	data, err := fs.ReadFile(s.fsys, path.Join(s.dir, FileName))
	if err != nil {
		return "", err
	}
	_, body, err := Parse(data)
	return body, err
}

// Files lists bundled resource files (relative paths, excluding SKILL.md).
func (s *Skill) Files() ([]string, error) {
	var out []string
	root := s.dir
	err := fs.WalkDir(s.fsys, root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if p != root && strings.HasPrefix(d.Name(), ".") {
				return fs.SkipDir
			}
			return nil
		}
		rel := strings.TrimPrefix(strings.TrimPrefix(p, root), "/")
		if root == "." {
			rel = p
		}
		if rel != FileName {
			out = append(out, rel)
		}
		return nil
	})
	return out, err
}

// ReadFile reads a resource relative to the skill directory. Paths that
// escape the skill directory are rejected.
func (s *Skill) ReadFile(rel string) ([]byte, error) {
	clean := path.Clean(strings.TrimPrefix(rel, "/"))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") || !fs.ValidPath(clean) {
		return nil, fmt.Errorf("invalid path %q", rel)
	}
	return fs.ReadFile(s.fsys, path.Join(s.dir, clean))
}

// Location returns a human-readable location for diagnostics.
func (s *Skill) Location() string { return s.dir }

// Validate checks if the skill satisfies the Agent Skills specification.
func (s *Skill) Validate() error {
	if err := s.Frontmatter.Validate(); err != nil {
		return err
	}
	if s.dir != "" && s.dir != "." {
		expected := path.Base(s.dir)
		if s.Name != expected {
			return fmt.Errorf("skill name %q does not match parent directory name %q", s.Name, expected)
		}
	}
	return nil
}

// Scripts lists bundled executable scripts (files under scripts/).
func (s *Skill) Scripts() ([]string, error) {
	return s.filesUnder("scripts")
}

// References lists bundled reference documentation (files under references/).
func (s *Skill) References() ([]string, error) {
	return s.filesUnder("references")
}

// Assets lists bundled assets or templates (files under assets/).
func (s *Skill) Assets() ([]string, error) {
	return s.filesUnder("assets")
}

func (s *Skill) filesUnder(prefix string) ([]string, error) {
	files, err := s.Files()
	if err != nil {
		return nil, err
	}
	var out []string
	prefixWithSlash := prefix + "/"
	for _, f := range files {
		if strings.HasPrefix(f, prefixWithSlash) {
			out = append(out, f)
		}
	}
	return out, nil
}

// --- Zero-dependency YAML Frontmatter Parser ---

type yamlLine struct {
	indent int
	text   string
}

func parseYAMLHeader(header string) (Frontmatter, error) {
	rawLines := strings.Split(header, "\n")
	var lines []yamlLine
	for _, raw := range rawLines {
		stripped := stripYAMLComment(raw)
		trimmed := strings.TrimSpace(stripped)
		if trimmed == "" {
			continue
		}
		// Count leading spaces (tabs count as 2 spaces)
		indent := 0
	countIndent:
		for _, ch := range raw {
			switch ch {
			case ' ':
				indent++
			case '\t':
				indent += 2
			default:
				break countIndent
			}
		}
		lines = append(lines, yamlLine{indent: indent, text: trimmed})
	}

	if len(lines) == 0 {
		return Frontmatter{}, nil
	}

	parsed, _ := parseYAMLBlock(lines, 0, lines[0].indent)
	m, ok := parsed.(map[string]any)
	if !ok {
		return Frontmatter{}, nil
	}
	return mapToFrontmatter(m), nil
}

func parseYAMLBlock(lines []yamlLine, start int, baseIndent int) (any, int) {
	if start >= len(lines) {
		return nil, start
	}
	if strings.HasPrefix(lines[start].text, "- ") || lines[start].text == "-" {
		return parseYAMLSequence(lines, start, baseIndent)
	}
	return parseYAMLMapping(lines, start, baseIndent)
}

func parseYAMLSequence(lines []yamlLine, start int, baseIndent int) ([]any, int) {
	var list []any
	i := start
	for i < len(lines) {
		if lines[i].indent < baseIndent {
			break
		}
		if lines[i].indent > baseIndent {
			i++
			continue
		}
		if !strings.HasPrefix(lines[i].text, "- ") && lines[i].text != "-" {
			break
		}
		itemText := strings.TrimSpace(strings.TrimPrefix(lines[i].text, "-"))
		nextIdx := i + 1

		if nextIdx >= len(lines) || lines[nextIdx].indent <= lines[i].indent {
			list = append(list, parseYAMLScalar(itemText))
			i++
			continue
		}

		if itemText == "" {
			child, nextI := parseYAMLBlock(lines, nextIdx, lines[nextIdx].indent)
			list = append(list, child)
			i = nextI
			continue
		}

		if k, v, ok := splitYAMLKeyVal(itemText); ok {
			childMap := make(map[string]any)
			if v != "" {
				childMap[k] = parseYAMLScalar(v)
			}
			sub, nextI := parseYAMLBlock(lines, nextIdx, lines[nextIdx].indent)
			if sm, ok := sub.(map[string]any); ok {
				maps.Copy(childMap, sm)
			}
			list = append(list, childMap)
			i = nextI
			continue
		}

		list = append(list, parseYAMLScalar(itemText))
		i++
	}
	return list, i
}

func parseYAMLMapping(lines []yamlLine, start int, baseIndent int) (map[string]any, int) {
	mapping := make(map[string]any)
	i := start
	for i < len(lines) {
		if lines[i].indent < baseIndent {
			break
		}
		if lines[i].indent > baseIndent {
			i++
			continue
		}
		k, v, ok := splitYAMLKeyVal(lines[i].text)
		if !ok {
			i++
			continue
		}
		k = unquoteYAML(k)
		nextIdx := i + 1

		// Multiline folded (>) or literal (|)
		if v == "|" || v == ">" {
			isFolded := v == ">"
			var multiline []string
			for nextIdx < len(lines) && lines[nextIdx].indent > lines[i].indent {
				multiline = append(multiline, lines[nextIdx].text)
				nextIdx++
			}
			if isFolded {
				mapping[k] = strings.Join(multiline, " ")
			} else {
				mapping[k] = strings.Join(multiline, "\n")
			}
			i = nextIdx
			continue
		}

		if v != "" {
			mapping[k] = parseYAMLScalar(v)
			i++
			continue
		}

		// Empty v: check if subsequent lines are indented child block
		if nextIdx < len(lines) && lines[nextIdx].indent > lines[i].indent {
			child, nextI := parseYAMLBlock(lines, nextIdx, lines[nextIdx].indent)
			mapping[k] = child
			i = nextI
			continue
		}

		mapping[k] = ""
		i++
	}
	return mapping, i
}

func splitYAMLKeyVal(s string) (string, string, bool) {
	inSingle := false
	inDouble := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '\'':
			if !inDouble {
				inSingle = !inSingle
			}
		case '"':
			if !inSingle {
				inDouble = !inDouble
			}
		case ':':
			if !inSingle && !inDouble {
				if i+1 == len(s) || s[i+1] == ' ' || s[i+1] == '\t' {
					return strings.TrimSpace(s[:i]), strings.TrimSpace(s[i+1:]), true
				}
			}
		}
	}
	return "", "", false
}

func stripYAMLComment(s string) string {
	inSingle := false
	inDouble := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '\'':
			if !inDouble {
				inSingle = !inSingle
			}
		case '"':
			if !inSingle {
				inDouble = !inDouble
			}
		case '#':
			if !inSingle && !inDouble {
				return s[:i]
			}
		}
	}
	return s
}

func parseYAMLScalar(s string) any {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	// Double quoted string
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		var unquoted string
		if err := json.Unmarshal([]byte(s), &unquoted); err == nil {
			return unquoted
		}
		return s[1 : len(s)-1]
	}
	// Single quoted string
	if len(s) >= 2 && s[0] == '\'' && s[len(s)-1] == '\'' {
		return strings.ReplaceAll(s[1:len(s)-1], "''", "'")
	}
	// Flow sequence: [a, b, c]
	if strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]") {
		inner := strings.TrimSpace(s[1 : len(s)-1])
		if inner == "" {
			return []any{}
		}
		items := parseFlowSequence(inner)
		res := make([]any, len(items))
		for i, it := range items {
			res[i] = parseYAMLScalar(it)
		}
		return res
	}
	// Booleans
	lower := strings.ToLower(s)
	if lower == "true" || lower == "yes" || lower == "on" {
		return true
	}
	if lower == "false" || lower == "no" || lower == "off" {
		return false
	}
	if lower == "null" || lower == "~" {
		return nil
	}
	// Integers
	if n, err := strconv.ParseInt(s, 10, 64); err == nil {
		return int(n)
	}
	// Floats
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f
	}
	return s
}

func parseFlowSequence(s string) []string {
	var items []string
	var cur strings.Builder
	inSingle := false
	inDouble := false
	bracketDepth := 0
	for i := 0; i < len(s); i++ {
		ch := s[i]
		switch ch {
		case '\'':
			if !inDouble {
				inSingle = !inSingle
			}
			cur.WriteByte(ch)
		case '"':
			if !inSingle {
				inDouble = !inDouble
			}
			cur.WriteByte(ch)
		case '[':
			if !inSingle && !inDouble {
				bracketDepth++
			}
			cur.WriteByte(ch)
		case ']':
			if !inSingle && !inDouble && bracketDepth > 0 {
				bracketDepth--
			}
			cur.WriteByte(ch)
		case ',':
			if !inSingle && !inDouble && bracketDepth == 0 {
				item := strings.TrimSpace(cur.String())
				if item != "" {
					items = append(items, item)
				}
				cur.Reset()
				continue
			}
			cur.WriteByte(ch)
		default:
			cur.WriteByte(ch)
		}
	}
	if cur.Len() > 0 {
		item := strings.TrimSpace(cur.String())
		if item != "" {
			items = append(items, item)
		}
	}
	return items
}

func unquoteYAML(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && ((s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'')) {
		return s[1 : len(s)-1]
	}
	return s
}

func mapToFrontmatter(m map[string]any) Frontmatter {
	var fm Frontmatter
	if v, ok := m["name"].(string); ok {
		fm.Name = strings.TrimSpace(v)
	}
	if v, ok := m["description"].(string); ok {
		fm.Description = strings.TrimSpace(v)
	}
	if v, ok := m["license"].(string); ok {
		fm.License = strings.TrimSpace(v)
	}
	if v, ok := m["compatibility"].(string); ok {
		fm.Compatibility = strings.TrimSpace(v)
	}
	if v, ok := m["allowed-tools"]; ok {
		fm.AllowedTools = parseAllowedTools(v)
	} else if v, ok := m["allowed_tools"]; ok {
		fm.AllowedTools = parseAllowedTools(v)
	}
	if v, ok := m["metadata"].(map[string]any); ok {
		fm.Metadata = v
	}

	for k, v := range m {
		if k == "name" || k == "description" || k == "license" || k == "compatibility" ||
			k == "allowed-tools" || k == "allowed_tools" || k == "metadata" {
			continue
		}
		if fm.Metadata == nil {
			fm.Metadata = make(map[string]any)
		}
		fm.Metadata[k] = v
	}
	return fm
}

// parseAllowedTools converts an allowed-tools value into a string slice.
// Per Agent Skills spec, allowed-tools is a space-separated string, but YAML/JSON sequences
// are also supported for compatibility.
func parseAllowedTools(val any) []string {
	switch v := val.(type) {
	case []string:
		var out []string
		for _, item := range v {
			out = append(out, splitAllowedToolsString(item)...)
		}
		return out
	case []any:
		var out []string
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, splitAllowedToolsString(s)...)
			} else if item != nil {
				out = append(out, splitAllowedToolsString(fmt.Sprint(item))...)
			}
		}
		return out
	case string:
		return splitAllowedToolsString(v)
	default:
		return nil
	}
}

// splitAllowedToolsString splits a space-separated string of tool patterns (e.g. "Bash(git:*) Bash(jq:*) Read")
// while respecting parentheses and quotes.
func splitAllowedToolsString(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	var tokens []string
	var cur strings.Builder
	parenDepth := 0
	inSingle := false
	inDouble := false

	flush := func() {
		tok := strings.TrimSpace(cur.String())
		tok = strings.Trim(tok, ",")
		tok = unquoteYAML(tok)
		if tok != "" {
			tokens = append(tokens, tok)
		}
		cur.Reset()
	}

	for i := 0; i < len(s); i++ {
		ch := s[i]
		switch ch {
		case '\'':
			if !inDouble {
				inSingle = !inSingle
			}
			cur.WriteByte(ch)
		case '"':
			if !inSingle {
				inDouble = !inDouble
			}
			cur.WriteByte(ch)
		case '(':
			if !inSingle && !inDouble {
				parenDepth++
			}
			cur.WriteByte(ch)
		case ')':
			if !inSingle && !inDouble && parenDepth > 0 {
				parenDepth--
			}
			cur.WriteByte(ch)
		case ' ', '\t', '\r', '\n':
			if !inSingle && !inDouble && parenDepth == 0 {
				flush()
				continue
			}
			cur.WriteByte(ch)
		case ',':
			if !inSingle && !inDouble && parenDepth == 0 {
				flush()
				continue
			}
			cur.WriteByte(ch)
		default:
			cur.WriteByte(ch)
		}
	}
	flush()
	return tokens
}

// SkillAllowedToolsKey is the RunContext key used to store tools pre-approved by active skills.
const SkillAllowedToolsKey = "skill.active_allowed_tools"

// AddSkillAllowedTools registers pre-approved tool patterns in rc for the active run.
func AddSkillAllowedTools(rc *RunContext, tools ...string) {
	if rc == nil || len(tools) == 0 {
		return
	}
	var existing []string
	if v, ok := rc.Get(SkillAllowedToolsKey); ok {
		if list, ok := v.([]string); ok {
			existing = append([]string(nil), list...)
		}
	}
	seen := make(map[string]bool, len(existing)+len(tools))
	for _, t := range existing {
		seen[t] = true
	}
	for _, t := range tools {
		t = strings.TrimSpace(t)
		if t != "" && !seen[t] {
			seen[t] = true
			existing = append(existing, t)
		}
	}
	rc.Set(SkillAllowedToolsKey, existing)
}

// ActiveSkillAllowedTools returns the tool patterns pre-approved by currently active skills.
func ActiveSkillAllowedTools(ctx context.Context) []string {
	if rc := RunContextFrom(ctx); rc != nil {
		if v, ok := rc.Get(SkillAllowedToolsKey); ok {
			if list, ok := v.([]string); ok {
				return list
			}
		}
	}
	return nil
}
