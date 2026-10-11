// Skill implements SKILL.md-based skills: folders containing a
// SKILL.md file (YAML frontmatter + markdown instructions) and optional
// resources. Only the name/description catalog is advertised to the model up
// front; the full instructions are loaded on demand via the activate_skill
// tool (progressive disclosure).
package react

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"path"
	"strings"

	"gopkg.in/yaml.v3"
)

// FileName is the required skill manifest name.
const FileName = "SKILL.md"

// Frontmatter is the YAML header of a SKILL.md file.
type Frontmatter struct {
	Name         string         `yaml:"name"`
	Description  string         `yaml:"description"`
	AllowedTools []string       `yaml:"allowed-tools"`
	Metadata     map[string]any `yaml:"metadata"`
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
	if err := yaml.Unmarshal([]byte(header), &fm); err != nil {
		return fm, "", fmt.Errorf("invalid frontmatter: %w", err)
	}
	return fm, strings.TrimSpace(body), nil
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
