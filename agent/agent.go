// Package agent implements the agent core: a ReAct loop that alternates
// between model calls and tool execution, with streaming and non-streaming
// modes sharing a single implementation.
package agent

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gosolu/react/contextmgr"
	"github.com/gosolu/react/hook"
	"github.com/gosolu/react/memory"
	"github.com/gosolu/react/permission"
	"github.com/gosolu/react/plugin"
	"github.com/gosolu/react/skill"
	"github.com/gosolu/react/tool"
)

// Errors returned by the agent.
var (
	ErrMaxTurns    = errors.New("agent: max turns exceeded")
	ErrMaxDepth    = errors.New("agent: max sub-agent depth exceeded")
	ErrNoProvider  = errors.New("agent: provider is required")
	ErrAgentClosed = errors.New("agent: closed")
)

// Agent is an immutable, concurrency-safe agent. Create with New.
type Agent struct {
	cfg        config
	registry   *tool.Registry
	skills     *skill.Registry
	system     string
	middleware []tool.Middleware
	hooks      hook.Chain
	plugins    []plugin.Plugin

	approvalMu sync.Mutex
	closeOnce  sync.Once
}

// New builds an agent, initialising plugins and loading skills.
func New(opts ...Option) (*Agent, error) {
	cfg := config{
		name:        "agent",
		maxTurns:    20,
		maxParallel: 4,
		toolTimeout: 2 * time.Minute,
		maxDepth:    3,
	}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.provider == nil {
		return nil, ErrNoProvider
	}
	if cfg.store == nil {
		cfg.store = memory.NewInMemory()
	}
	if cfg.ctxManager == nil {
		cfg.ctxManager = contextmgr.Window{}
	}
	if cfg.policy == nil {
		cfg.policy = permission.Default()
	}

	a := &Agent{
		cfg:        cfg,
		registry:   tool.NewRegistry(),
		skills:     skill.NewRegistry(),
		middleware: append([]tool.Middleware(nil), cfg.middleware...),
		hooks:      append(hook.Chain(nil), cfg.hooks...),
	}
	for _, t := range cfg.tools {
		if err := a.registry.Register(t); err != nil {
			return nil, err
		}
	}

	reg := &registrar{a: a, skillSources: append([]skill.Source(nil), cfg.skillSources...)}
	for _, p := range cfg.plugins {
		reg.current = p.Name()
		if err := p.Init(context.Background(), reg); err != nil {
			a.Close()
			return nil, fmt.Errorf("agent: plugin %q: %w", p.Name(), err)
		}
		a.plugins = append(a.plugins, p)
	}

	if err := a.skills.Load(reg.skillSources...); err != nil && cfg.strictSkills {
		a.Close()
		return nil, fmt.Errorf("agent: load skills: %w", err)
	}
	if a.skills.Len() > 0 {
		for _, t := range a.skills.Tools() {
			if err := a.registry.Register(t); err != nil {
				a.Close()
				return nil, err
			}
		}
	}

	var parts []string
	if s := strings.TrimSpace(cfg.system); s != "" {
		parts = append(parts, s)
	}
	parts = append(parts, reg.prompts...)
	if cat := a.skills.Catalog(); cat != "" {
		parts = append(parts, cat)
	}
	a.system = strings.Join(parts, "\n\n")
	return a, nil
}

// Name returns the agent's name.
func (a *Agent) Name() string { return a.cfg.name }

// SystemPrompt returns the fully assembled system prompt.
func (a *Agent) SystemPrompt() string { return a.system }

// Tools returns the specs of all registered tools.
func (a *Agent) Tools() []tool.Spec { return a.registry.Specs() }

// Skills returns all loaded skills.
func (a *Agent) Skills() []*skill.Skill { return a.skills.List() }

// Memory returns the session store.
func (a *Agent) Memory() memory.Store { return a.cfg.store }

// Close releases plugin resources (in reverse order). It is idempotent.
func (a *Agent) Close() error {
	var errs []error
	a.closeOnce.Do(func() {
		for _, v := range slices.Backward(a.plugins) {
			if c, ok := v.(plugin.Closer); ok {
				if err := c.Close(); err != nil {
					errs = append(errs, fmt.Errorf("plugin %q: %w", v.Name(), err))
				}
			}
		}
	})
	return errors.Join(errs...)
}

// registrar implements plugin.Registrar during New.
type registrar struct {
	a            *Agent
	current      string
	prompts      []string
	skillSources []skill.Source
}

func (r *registrar) AddTool(t tool.Tool) error {
	if err := r.a.registry.Register(t); err != nil {
		return fmt.Errorf("plugin %q: %w", r.current, err)
	}
	return nil
}

func (r *registrar) AddToolMiddleware(m tool.Middleware) { r.a.middleware = append(r.a.middleware, m) }

func (r *registrar) AddSkillSource(src skill.Source) { r.skillSources = append(r.skillSources, src) }
func (r *registrar) AddHooks(h hook.Hooks)           { r.a.hooks = append(r.a.hooks, h) }
func (r *registrar) AddSystemPrompt(s string) {
	if s = strings.TrimSpace(s); s != "" {
		r.prompts = append(r.prompts, s)
	}
}

func (r *registrar) Config(key string) (any, bool) {
	v, ok := r.a.cfg.pluginConfig[key]
	return v, ok
}
