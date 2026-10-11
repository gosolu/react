// Plugin defines compiled-in agent plugins. A plugin bundles tools,
// skills, hooks, middleware and system-prompt fragments and registers them
// with an agent at construction time.
package react

import (
	"context"
)

// Plugin extends an agent.
type Plugin interface {
	Name() string
	Init(ctx context.Context, r Registrar) error
}

// Closer is optionally implemented by plugins that hold resources.
// It is called by agent.Close in reverse registration order.
type Closer interface {
	Close() error
}

// Registrar is the surface a plugin uses to extend the agent.
type Registrar interface {
	AddTool(t Tool) error
	AddToolMiddleware(m ToolMiddleware)
	AddSkillSource(src SkillSource)
	AddHooks(h Hooks)
	AddSystemPrompt(fragment string)
	// Config returns plugin configuration supplied via agent.WithPluginConfig.
	Config(key string) (any, bool)
}

// Func adapts a function into a Plugin.
func Func(name string, init func(ctx context.Context, r Registrar) error) Plugin {
	return funcPlugin{name: name, init: init}
}

type funcPlugin struct {
	name string
	init func(context.Context, Registrar) error
}

func (p funcPlugin) Name() string                                { return p.name }
func (p funcPlugin) Init(ctx context.Context, r Registrar) error { return p.init(ctx, r) }
