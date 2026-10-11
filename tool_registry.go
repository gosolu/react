package react

import (
	"fmt"
	"regexp"
	"sync"
)

var validName = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// Registry holds tools by name, preserving registration order.
// It is safe for concurrent use.
type ToolRegistry struct {
	mu    sync.RWMutex
	tools map[string]Tool
	order []string
}

// NewRegistry creates a registry, panicking on invalid or duplicate tools.
func NewToolRegistry(tools ...Tool) *ToolRegistry {
	r := &ToolRegistry{tools: map[string]Tool{}}
	for _, t := range tools {
		if err := r.Register(t); err != nil {
			panic(err)
		}
	}
	return r
}

// Register adds a tool. Names must be unique and match [a-zA-Z0-9_-]{1,64}.
func (r *ToolRegistry) Register(t Tool) error {
	name := t.Spec().Name
	if !validName.MatchString(name) {
		return fmt.Errorf("tool: invalid name %q", name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.tools == nil {
		r.tools = map[string]Tool{}
	}
	if _, dup := r.tools[name]; dup {
		return fmt.Errorf("tool: duplicate name %q", name)
	}
	r.tools[name] = t
	r.order = append(r.order, name)
	return nil
}

// Get looks up a tool by name.
func (r *ToolRegistry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	return t, ok
}

// List returns tools in registration order.
func (r *ToolRegistry) List() []Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Tool, 0, len(r.order))
	for _, n := range r.order {
		out = append(out, r.tools[n])
	}
	return out
}

// Specs returns tool specs in registration order.
func (r *ToolRegistry) Specs() []ToolSpec {
	ts := r.List()
	out := make([]ToolSpec, len(ts))
	for i, t := range ts {
		out[i] = t.Spec()
	}
	return out
}

// Len returns the number of tools.
func (r *ToolRegistry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.order)
}
