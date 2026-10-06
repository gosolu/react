package agent

import (
	"time"

	"github.com/gosolu/react/contextmgr"
	"github.com/gosolu/react/hook"
	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/memory"
	"github.com/gosolu/react/permission"
	"github.com/gosolu/react/plugin"
	"github.com/gosolu/react/skill"
	"github.com/gosolu/react/tool"
)

// Option configures an Agent.
type Option func(*config)

type config struct {
	name          string
	provider      llm.Provider
	model         string
	system        string
	tools         []tool.Tool
	middleware    []tool.Middleware
	skillSources  []skill.Source
	plugins       []plugin.Plugin
	pluginConfig  map[string]any
	hooks         hook.Chain
	store         memory.Store
	ctxManager    contextmgr.Manager
	contextBudget int
	policy        permission.Policy
	approver      permission.Approver
	maxTurns      int
	maxParallel   int
	toolTimeout   time.Duration
	maxTokens     int
	temperature   *float64
	maxDepth      int
	strictSkills  bool
}

// WithName sets the agent name (shown in events; default "agent").
func WithName(name string) Option { return func(c *config) { c.name = name } }

// WithProvider sets the model provider (required).
func WithProvider(p llm.Provider) Option { return func(c *config) { c.provider = p } }

// WithModel sets the model identifier.
func WithModel(m string) Option { return func(c *config) { c.model = m } }

// WithSystemPrompt sets the base system prompt.
func WithSystemPrompt(s string) Option { return func(c *config) { c.system = s } }

// WithTools registers tools.
func WithTools(ts ...tool.Tool) Option { return func(c *config) { c.tools = append(c.tools, ts...) } }

// WithToolMiddleware wraps every tool call; the first middleware is outermost.
func WithToolMiddleware(ms ...tool.Middleware) Option {
	return func(c *config) { c.middleware = append(c.middleware, ms...) }
}

// WithSkillDirs loads SKILL.md skills from OS directories (missing dirs are ignored).
func WithSkillDirs(dirs ...string) Option {
	return func(c *config) {
		for _, d := range dirs {
			c.skillSources = append(c.skillSources, skill.Dir(d))
		}
	}
}

// WithSkillSources loads skills from arbitrary sources (e.g. embedded FS).
func WithSkillSources(srcs ...skill.Source) Option {
	return func(c *config) { c.skillSources = append(c.skillSources, srcs...) }
}

// WithStrictSkills makes New fail if any skill fails to load (default: skip invalid skills).
func WithStrictSkills() Option { return func(c *config) { c.strictSkills = true } }

// WithPlugins registers plugins, initialised in order during New.
func WithPlugins(ps ...plugin.Plugin) Option {
	return func(c *config) { c.plugins = append(c.plugins, ps...) }
}

// WithPluginConfig supplies a value readable by plugins via Registrar.Config.
func WithPluginConfig(key string, v any) Option {
	return func(c *config) {
		if c.pluginConfig == nil {
			c.pluginConfig = map[string]any{}
		}
		c.pluginConfig[key] = v
	}
}

// WithHooks adds lifecycle hooks.
func WithHooks(hs ...hook.Hooks) Option { return func(c *config) { c.hooks = append(c.hooks, hs...) } }

// WithMemory sets the session store (default: in-memory).
func WithMemory(s memory.Store) Option { return func(c *config) { c.store = s } }

// WithContextManager sets the context manager and token budget for history.
func WithContextManager(m contextmgr.Manager, budget int) Option {
	return func(c *config) { c.ctxManager, c.contextBudget = m, budget }
}

// WithPermission sets the tool permission policy and approver (default:
// permission.Default with no approver, so "ask" decisions are denied).
func WithPermission(p permission.Policy, a permission.Approver) Option {
	return func(c *config) { c.policy, c.approver = p, a }
}

// WithMaxTurns bounds model calls per run (default 20).
func WithMaxTurns(n int) Option { return func(c *config) { c.maxTurns = n } }

// WithMaxParallelTools bounds concurrent tool calls (default 4; 1 = sequential).
func WithMaxParallelTools(n int) Option { return func(c *config) { c.maxParallel = n } }

// WithToolTimeout bounds each tool call (default 2m; 0 disables).
func WithToolTimeout(d time.Duration) Option { return func(c *config) { c.toolTimeout = d } }

// WithMaxTokens sets the per-response output token limit.
func WithMaxTokens(n int) Option { return func(c *config) { c.maxTokens = n } }

// WithTemperature sets the sampling temperature.
func WithTemperature(t float64) Option { return func(c *config) { c.temperature = &t } }

// WithMaxDepth bounds sub-agent nesting (default 3).
func WithMaxDepth(n int) Option { return func(c *config) { c.maxDepth = n } }

// RunOption customises a single run.
type RunOption func(*runConfig)

type runConfig struct {
	parts      []llm.Part
	history    []llm.Message
	model      string
	toolChoice string
	onEvent    func(Event)
}

// WithInputParts appends extra content (e.g. images) to the user message.
func WithInputParts(parts ...llm.Part) RunOption {
	return func(r *runConfig) { r.parts = append(r.parts, parts...) }
}

// WithHistory supplies prior messages for a stateless run (empty session ID).
// It is ignored when a session ID is given.
func WithHistory(msgs []llm.Message) RunOption { return func(r *runConfig) { r.history = msgs } }

// WithRunModel overrides the model for this run.
func WithRunModel(m string) RunOption { return func(r *runConfig) { r.model = m } }

// WithToolChoice sets the tool choice for this run (auto, none, required, or a tool name).
func WithToolChoice(choice string) RunOption { return func(r *runConfig) { r.toolChoice = choice } }

// WithEventHandler observes events during Run (non-stream mode). Calls are
// serialized. It is also honoured by Stream, in addition to the iterator.
func WithEventHandler(fn func(Event)) RunOption { return func(r *runConfig) { r.onEvent = fn } }
