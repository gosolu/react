package react

import (
	"time"
)

// Option configures an Agent.
type Option func(*config)

type config struct {
	name          string
	provider      Provider
	model         string
	system        string
	tools         []Tool
	middleware    []ToolMiddleware
	skillSources  []SkillSource
	plugins       []Plugin
	pluginConfig  map[string]any
	hooks         HookChain
	store         Store
	ctxManager    ContextManager
	contextBudget int
	policy        PermissionPolicy
	approver      PermissionApprover
	maxTurns      int
	maxParallel   int
	toolTimeout   time.Duration
	maxTokens     int
	temperature   *float64
	maxDepth          int
	strictSkills      bool
	maxToolOutput     int
	persistCompaction bool
}

// WithName sets the agent name (shown in events; default "agent").
func WithName(name string) Option { return func(c *config) { c.name = name } }

// WithProvider sets the model provider (required).
func WithProvider(p Provider) Option { return func(c *config) { c.provider = p } }

// WithModel sets the model identifier.
func WithModel(m string) Option { return func(c *config) { c.model = m } }

// WithSystemPrompt sets the base system prompt.
func WithSystemPrompt(s string) Option { return func(c *config) { c.system = s } }

// WithTools registers tools.
func WithTools(ts ...Tool) Option { return func(c *config) { c.tools = append(c.tools, ts...) } }

// WithToolMiddleware wraps every tool call; the first middleware is outermost.
func WithToolMiddleware(ms ...ToolMiddleware) Option {
	return func(c *config) { c.middleware = append(c.middleware, ms...) }
}

// WithSkillDirs loads SKILL.md skills from OS directories (missing dirs are ignored).
func WithSkillDirs(dirs ...string) Option {
	return func(c *config) {
		for _, d := range dirs {
			c.skillSources = append(c.skillSources, SkillDir(d))
		}
	}
}

// WithSkillSources loads skills from arbitrary sources (e.g. embedded FS).
func WithSkillSources(srcs ...SkillSource) Option {
	return func(c *config) { c.skillSources = append(c.skillSources, srcs...) }
}

// WithStrictSkills makes New fail if any skill fails to load (default: skip invalid skills).
func WithStrictSkills() Option { return func(c *config) { c.strictSkills = true } }

// WithPlugins registers plugins, initialised in order during New.
func WithPlugins(ps ...Plugin) Option {
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
func WithHooks(hs ...Hooks) Option { return func(c *config) { c.hooks = append(c.hooks, hs...) } }

// WithStore sets the session store (default: in-memory).
func WithStore(s Store) Option { return func(c *config) { c.store = s } }

// WithMemory sets the session store (alias for WithStore).
func WithMemory(s Store) Option { return func(c *config) { c.store = s } }

// WithContextManager sets the context manager and token budget for history.
func WithContextManager(m ContextManager, budget int) Option {
	return func(c *config) { c.ctxManager, c.contextBudget = m, budget }
}

// WithPermission sets the tool permission policy and approver (default:
// DefaultPermissionPolicy with no approver, so "ask" decisions are denied).
func WithPermission(p PermissionPolicy, a PermissionApprover) Option {
	return func(c *config) { c.policy, c.approver = p, a }
}

// WithPolicy sets the tool permission policy without changing the approver.
func WithPolicy(p PermissionPolicy) Option { return func(c *config) { c.policy = p } }

// WithApprover sets the human-in-the-loop approver for "ask" decisions.
func WithApprover(a PermissionApprover) Option { return func(c *config) { c.approver = a } }

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

// WithMaxToolOutput bounds the maximum character/byte length of tool results (default 64KB, 0 disables).
func WithMaxToolOutput(bytes int) Option { return func(c *config) { c.maxToolOutput = bytes } }

// WithPersistCompaction enables saving compacted history back to Store when Fit shrinks the history.
func WithPersistCompaction() Option { return func(c *config) { c.persistCompaction = true } }

// RunOption customises a single run.
type RunOption func(*runConfig)

type runConfig struct {
	parts          []Part
	history        []Message
	model          string
	toolChoice     string
	tools          []Tool
	responseFormat *ResponseFormat
	onEvent        func(Event)
}

// WithInputParts appends extra content (e.g. images) to the user message.
func WithInputParts(parts ...Part) RunOption {
	return func(r *runConfig) { r.parts = append(r.parts, parts...) }
}

// WithHistory supplies prior messages for a stateless run (empty session ID).
// It is ignored when a session ID is given.
func WithHistory(msgs []Message) RunOption { return func(r *runConfig) { r.history = msgs } }

// WithRunModel overrides the model for this run.
func WithRunModel(m string) RunOption { return func(r *runConfig) { r.model = m } }

// WithToolChoice sets the tool choice for this run (auto, none, required, or a tool name).
func WithToolChoice(choice string) RunOption { return func(r *runConfig) { r.toolChoice = choice } }

// WithRunTools injects additional tools for this specific run.
func WithRunTools(ts ...Tool) RunOption {
	return func(r *runConfig) { r.tools = append(r.tools, ts...) }
}

// WithResponseFormat sets the desired response format (e.g. JSON schema) for this run.
func WithResponseFormat(rf ResponseFormat) RunOption {
	return func(r *runConfig) { r.responseFormat = &rf }
}

// WithEventHandler observes events during Run (non-stream mode). Calls are
// serialized. It is also honoured by Stream, in addition to the iterator.
func WithEventHandler(fn func(Event)) RunOption { return func(r *runConfig) { r.onEvent = fn } }
