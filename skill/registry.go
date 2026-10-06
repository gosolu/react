package skill

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"sort"
	"strings"
	"sync"
)

// Source provides skills.
type Source interface {
	Skills() ([]*Skill, error)
}

// SourceFunc adapts a function to Source.
type SourceFunc func() ([]*Skill, error)

func (f SourceFunc) Skills() ([]*Skill, error) { return f() }

// FS discovers skills under root within fsys: root itself if it contains a
// SKILL.md, otherwise each immediate subdirectory containing one. Use this
// with //go:embed to ship skills inside plugins.
func FS(fsys fs.FS, root string) Source {
	if root == "" {
		root = "."
	}
	return SourceFunc(func() ([]*Skill, error) {
		if _, err := fs.Stat(fsys, path.Join(root, FileName)); err == nil {
			s, err := New(fsys, root)
			if err != nil {
				return nil, err
			}
			return []*Skill{s}, nil
		}
		entries, err := fs.ReadDir(fsys, root)
		if err != nil {
			return nil, err
		}
		var out []*Skill
		var errs []error
		for _, e := range entries {
			if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
				continue
			}
			dir := path.Join(root, e.Name())
			if _, err := fs.Stat(fsys, path.Join(dir, FileName)); err != nil {
				continue
			}
			s, err := New(fsys, dir)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			out = append(out, s)
		}
		return out, errors.Join(errs...)
	})
}

// Dir discovers skills in an OS directory. A missing directory yields no skills.
func Dir(dir string) Source {
	return SourceFunc(func() ([]*Skill, error) {
		if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return FS(os.DirFS(dir), ".").Skills()
	})
}

// Registry holds loaded skills. It is safe for concurrent use.
type Registry struct {
	mu     sync.RWMutex
	skills map[string]*Skill
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return &Registry{skills: map[string]*Skill{}} }

// Load adds all skills from the sources. Later sources do not override
// earlier ones; duplicates are reported as errors. Valid skills are added
// even if some fail to load.
func (r *Registry) Load(sources ...Source) error {
	var errs []error
	for _, src := range sources {
		skills, err := src.Skills()
		if err != nil {
			errs = append(errs, err)
		}
		for _, s := range skills {
			if err := r.Add(s); err != nil {
				errs = append(errs, err)
			}
		}
	}
	return errors.Join(errs...)
}

// Add registers a single skill.
func (r *Registry) Add(s *Skill) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.skills == nil {
		r.skills = map[string]*Skill{}
	}
	if _, dup := r.skills[s.Name]; dup {
		return fmt.Errorf("skill: duplicate name %q", s.Name)
	}
	r.skills[s.Name] = s
	return nil
}

// Get returns a skill by name.
func (r *Registry) Get(name string) (*Skill, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	s, ok := r.skills[name]
	return s, ok
}

// List returns skills sorted by name.
func (r *Registry) List() []*Skill {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Skill, 0, len(r.skills))
	for _, s := range r.skills {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Len returns the number of skills.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.skills)
}

// Catalog renders the system-prompt section advertising available skills.
// It returns "" when there are no skills.
func (r *Registry) Catalog() string {
	skills := r.List()
	if len(skills) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("## Skills\n")
	b.WriteString("You have access to the following skills. When a task matches a skill's description, ")
	b.WriteString("call the `" + ActivateToolName + "` tool with its name to load the full instructions before proceeding. ")
	b.WriteString("Use `" + ReadFileToolName + "` to read files bundled with a skill.\n\n")
	for _, s := range skills {
		fmt.Fprintf(&b, "- %s: %s\n", s.Name, strings.TrimSpace(s.Description))
	}
	return b.String()
}
