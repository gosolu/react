# ReAct: Go Agent Core

A modular, minimal-dependency AI agent core written from scratch in Go.

Features a unified ReAct (Reasoning + Acting) loop with support for both streaming and non-streaming modes, typed tools with automatic JSON Schema generation, progressive disclosure skills, in-process plugins, lifecycle hooks, session memory, permission policies, and sub-agents.

---

## Features

- **Unified Top-Level Package:** Core primitives (`Agent`, `Tool`, `Skill`, `Policy`, `Store`, `Hook`, `Event`, `Plugin`, `ContextManager`) are all directly exported by `github.com/gosolu/react`.
- **Unified ReAct Engine:** A single loop powers both non-streaming (`Run`) and streaming (`Stream`) execution.
- **Go 1.23+ Iterators:** Streaming exposes range-over-func iterators (`iter.Seq2[Event, error]`) with cancellation on break.
- **Zero External Dependencies:** Built 100% on the Go standard library (no third-party dependencies required). Native HTTP clients, SSE parsers, and a built-in YAML frontmatter parser for skills.
- **Provider Abstraction:**
  - `provider/deepseek`: Native DeepSeek API client with vision (`deepseek-flash`), reasoning (`deepseek-reasoner` / R1), supporting both Chat Completions and Responses API (`/responses`).
  - `provider/opencode`: OpenCode Go curated model gateway with client User-Agent and session routing (`x-opencode-session`), supporting both Chat Completions and Responses API.
  - `provider/openai`: Works with OpenAI and OpenAI-compatible endpoints, supporting both standard Chat Completions and the new Responses API (`/responses`).
  - `provider/anthropic`: Anthropic Messages API.
  - `provider/mock`: Scripted provider for deterministic testing and offline development.
- **Type-safe Tools:**
  - Generics-based tool definitions (`react.NewTool[In, Out](...)`).
  - Automatic JSON Schema generation via reflection with support for struct tags (`desc`, `enum`, `omitempty`, `required`).
  - Parallel tool execution with semaphore bounds, per-tool timeouts, and panic recovery.
- **Progressive Disclosure Skills:**
  - `SKILL.md` folders with YAML frontmatter.
  - Only the catalog is advertised up-front; full instructions are lazily loaded on demand via `activate_skill`.
  - Directory-traversal protected `read_skill_file` tool.
- **In-Process Plugins:**
  - Modular plugins can register tools, middleware, skills, hooks, and system prompt fragments.
- **Comprehensive Lifecycle Hooks (Claude Code Architecture):**
  - Prompt & Turn: `BeforePrompt` (block/transform input), `BeforeStop` (block stopping to enforce verification), `OnStopFailure`.
  - Tool Execution: `BeforeTool` (allow, deny, override result, rewrite arguments, inject context), `AfterTool`, `OnToolFailure`, `AfterToolBatch`.
  - Matcher Patterns: `MatchTool`, `MatchToolAfter` glob matching.
  - Model & Context: `BeforeModel`, `AfterModel`, `PreCompact`, `PostCompact`, and dynamic `<system-reminder>` context injection.
  - Permissions & Delegation: `OnPermissionRequest` (automated approval), `OnPermissionDenied`, `SubagentStart`, `SubagentStop`.
- **Pluggable Session Memory:**
  - Built-in concurrency-safe `InMemory` store.
  - Open `react.Store` interface to connect custom backends (Redis, PostgreSQL, etc.).
- **Context Window Management:**
  - Sliding window (`react.ContextWindow`) that preserves tool-call / tool-result message integrity.
  - LLM-based summarizer (`react.ContextSummarizer`) with caching.
- **Permission & Safety:**
  - Rule-based policy (`ReadOnly` auto-allowed, `Destructive` requires approval).
  - Human-in-the-loop approval interface (`react.PermissionApprover`).
- **Sub-Agents (`agent.AsTool`):**
  - Expose any agent as a tool to parent agents with depth tracking and forwarded event streams.
- **CLI REPL (`cmd/react`):**
  - Ready-to-use terminal interface with live streaming, slash commands (`/tools`, `/skills`, `/reset`, `/exit`), and interactive `[y/N]` approval prompts for destructive tools.

---

## Directory Structure

```
react/
├── agent.go, loop.go, run.go, subagent.go, options.go, event.go  # Core ReAct loop, events, options, sub-agents
├── tool.go, tool_*.go                                            # Generic tools, JSON schema, executor, middleware
├── skill.go, skill_*.go                                          # SKILL.md parser, registry, activation tools
├── hook.go                                                       # RunContext and lifecycle hooks
├── memory.go                                                     # Session store interface and in-memory implementation
├── contextmgr.go, contextmgr_summarize.go                        # Context window sliding window & summarization
├── permission.go                                                 # Permission policies and approver interfaces
├── plugin.go                                                     # Plugin interface & Registrar
├── tools/builtin/                                                # Standard tools: read_file, write_file, list_dir, shell
├── cmd/react/                                                    # Interactive CLI REPL application
├── examples/
│   ├── basic/                                                    # Minimal non-streaming example
│   ├── plugin/                                                   # In-process plugin & skill folder example
│   └── streaming/                                                # Streaming example using iter.Seq2
├── provider.go, message.go, stream.go                          # Messages, streaming accumulator, provider interface
└── provider/                                                          # Provider abstraction and HTTP clients
    ├── anthropic/                                                # Anthropic Messages API client
    ├── deepseek/                                                 # DeepSeek API client (vision, chat, reasoner)
    ├── mock/                                                     # Scripted mock provider for testing
    ├── opencode/                                                 # OpenCode Go gateway client (session & UA headers)
    └── openai/                                                   # OpenAI-compatible API client
```

---

## Installation

Requires **Go 1.23+** (tested with Go 1.27):

```bash
go get github.com/gosolu/react
```

---

## Quick Start

### 1. Basic Agent (Streaming)

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider/openai"
	"github.com/gosolu/react/tools/builtin"
)

func main() {
	provider := openai.New(openai.Config{
		APIKey: os.Getenv("OPENAI_API_KEY"),
	})

	a, err := react.New(
		react.WithProvider(provider),
		react.WithModel("gpt-4o"),
		react.WithTools(builtin.ReadFile(), builtin.ListDir()),
	)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	for ev, err := range a.Stream(ctx, "session-1", "List files in the current directory") {
		if err != nil {
			log.Fatal(err)
		}
		switch ev.Type {
		case react.EventTextDelta:
			fmt.Print(ev.Text)
		case react.EventToolCallStart:
			fmt.Printf("\n[Calling tool: %s]\n", ev.ToolCall.Name)
		case react.EventToolResult:
			fmt.Printf("[Tool result: %s]\n", ev.ToolResult.Text())
		}
	}
	fmt.Println()
}
```

### 2. Defining Custom Typed Tools

Define input parameters with Go structs. JSON Schema is derived automatically via reflection:

```go
type CalculatorInput struct {
	Operation string  `json:"operation" enum:"add,sub,mul,div" desc:"The arithmetic operation"`
	A         float64 `json:"a" desc:"First operand"`
	B         float64 `json:"b" desc:"Second operand"`
}

calcTool := react.NewTool("calc", "Performs basic arithmetic", func(ctx context.Context, in CalculatorInput) (float64, error) {
	switch in.Operation {
	case "add":
		return in.A + in.B, nil
	case "sub":
		return in.A - in.B, nil
	case "mul":
		return in.A * in.B, nil
	case "div":
		if in.B == 0 {
			return 0, fmt.Errorf("division by zero")
		}
		return in.A / in.B, nil
	default:
		return 0, fmt.Errorf("unsupported operation: %s", in.Operation)
	}
}, react.ReadOnly())
```

### 3. Non-Streaming Execution (`Run`)

```go
res, err := a.Run(context.Background(), "session-1", "Calculate 42 * 2")
if err != nil {
    log.Fatal(err)
}
fmt.Println("Answer:", res.Output)
fmt.Println("Turns taken:", res.Turns)
fmt.Println("Tokens used:", res.Usage.Total())
```

---

## Skills (`SKILL.md`)

Organize domain instructions into folders with a `SKILL.md` file:

```markdown
---
name: code-reviewer
description: Review code changes for security, performance, and style issues.
allowed-tools: [read_file, list_dir]
---
# Code Reviewer Instructions

When reviewing code:
1. Examine the file changes thoroughly.
2. Flag potential null/nil pointer dereferences and concurrency races.
3. Check for proper error handling.
```

Load skills into the agent:

```go
a, err := react.New(
    react.WithProvider(provider),
    react.WithSkillDirs("./skills"),
)
```

The model automatically receives catalog summaries in its system prompt and calls `activate_skill("code-reviewer")` when relevant.

---

## Writing an In-Process Plugin

Plugins implement `react.Plugin`:

```go
type MyPlugin struct{}

func (p *MyPlugin) Name() string { return "my_plugin" }

func (p *MyPlugin) Init(ctx context.Context, r react.Registrar) error {
    // Add tools
    r.AddTool(myCustomTool)

    // Add lifecycle hooks
    r.AddHooks(react.Hooks{
        BeforeTool: func(ctx context.Context, rc *react.RunContext, call *react.ToolCall) (react.HookDecision, error) {
            log.Printf("Executing tool %s in turn %d", call.Name, rc.Turn)
            return react.HookDecision{Action: react.HookContinue}, nil
        },
    })

    // Add system prompt fragments
    r.AddSystemPrompt("Always prioritize concise answers.")
    return nil
}
```

---

## Lifecycle Hooks (Claude Code Architecture)

Attach granular lifecycle hooks via `react.WithHooks(...)` to control prompts, tool calls, model requests, and termination:

```go
agent, _ := react.New(
    react.WithProvider(provider),
    react.WithModel("claude-3-7-sonnet"),
    react.WithTools(tools...),
    react.WithHooks(react.Hooks{
        // 1. Inspect or sanitize user prompts before agent processes them
        BeforePrompt: func(ctx context.Context, rc *react.RunContext, prompt string) (react.HookPromptDecision, error) {
            if strings.Contains(prompt, "drop database") {
                return react.HookPromptDecision{Action: react.HookDeny, Reason: "disallowed command"}, nil
            }
            return react.HookPromptDecision{Action: react.HookContinue}, nil
        },

        // 2. Filter tool execution with MatchTool (glob pattern matching)
        BeforeTool: react.MatchTool("bash*", func(ctx context.Context, rc *react.RunContext, call *react.ToolCall) (react.HookDecision, error) {
            // Can Deny, Override with synthetic result, or rewrite UpdatedArguments
            return react.HookDecision{Action: react.HookContinue, AdditionalContext: "Running in sandbox"}, nil
        }),

        // 3. Inspect batches of parallel tool calls after they complete
        AfterToolBatch: func(ctx context.Context, rc *react.RunContext, results []react.ToolCallResult) error {
            log.Printf("Turn %d completed %d tool calls", rc.Turn, len(results))
            return nil
        },

        // 4. BeforeStop: Prevent agent from stopping prematurely if tasks are unfinished
        BeforeStop: func(ctx context.Context, rc *react.RunContext, output string) (react.HookDecision, error) {
            if !testsPassed() {
                // Keep agent working: injects <system-reminder> feedback into next turn
                return react.HookDecision{
                    Action: react.HookContinueWorking,
                    Reason: "Tests have not passed yet. Please run tests and fix errors before finishing.",
                }, nil
            }
            return react.HookDecision{Action: react.HookContinue}, nil
        },
    }),
)
```

---

## Structured Output

Constrain model outputs to a typed Go struct using standard schema generation:

```go
type Analysis struct {
    Summary    string   `json:"summary" desc:"Brief summary of findings"`
    RiskScore  int      `json:"risk_score" desc:"Risk score from 1 to 10"`
    Actionable []string `json:"actionable" desc:"Recommended next steps"`
}

analysis, res, err := react.RunStructured[Analysis](ctx, agent, "session-id", "Analyze recent incident report")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Risk: %d/10 - %s\n", analysis.RiskScore, analysis.Summary)
```

---

## Persistent Storage & Confinement

### File-Based Session Storage

Persist conversation histories across process restarts using atomic writes:

```go
store, err := react.NewFileStore("./sessions")
agent, err := react.New(
    react.WithProvider(provider),
    react.WithStore(store),
)
```

### Confined Filesystem & Surgical Edits

Restrict filesystem access to a target directory and perform exact surgical replacements:

```go
// Confine tools to a sandbox workspace
tools := builtin.NewFilesystem("/path/to/workspace")

// Or add surgical string replace tool
editTool := builtin.EditFile(builtin.WithBaseDir("/path/to/workspace"))
```

---

## Sub-Agents (Agent-as-a-Tool)

Delegate complex tasks to specialized child agents:

```go
researcher, _ := react.New(
    react.WithName("researcher"),
    react.WithProvider(provider),
    react.WithTools(builtin.ReadFile(), builtin.ListDir()),
)

mainAgent, _ := react.New(
    react.WithName("main"),
    react.WithProvider(provider),
    react.WithTools(
        researcher.AsTool("research_codebase", "Delegate codebase research tasks to a specialist agent"),
    ),
)
```

---

## Running the REPL CLI

The built-in CLI supports OpenAI, Anthropic, DeepSeek, OpenCode, and Mock providers with session memory:

```bash
# Using OpenAI (or compatible API)
export OPENAI_API_KEY="your-api-key"
go run ./cmd/react --provider openai --model gpt-4o

# Using DeepSeek (Chat or Reasoner with thinking display)
export DEEPSEEK_API_KEY="your-api-key"
go run ./cmd/react --provider deepseek --model deepseek-reasoner

# Using OpenCode
export OPENCODE_API_KEY="your-api-key"
go run ./cmd/react --provider opencode --model deepseek-v4-pro

# Using Anthropic
export ANTHROPIC_API_KEY="your-api-key"
go run ./cmd/react --provider anthropic --model claude-3-7-sonnet-20250219

# Persisting sessions across CLI runs
go run ./cmd/react --provider mock --store-dir ./data/sessions

# Offline testing with mock provider
go run ./cmd/react --provider mock
```

Interactive features:
- Interactive permissions: Destructive tools prompt with `[y/N/a]`, where `a` remembers approval for the remainder of the session.
- `/tools`: List registered tools
- `/skills`: List available skills
- `/reset`: Clear the active session history
- `/exit`: Quit the REPL

---

## Testing

Run unit and race detector tests across the entire codebase:

```bash
go test -race ./...
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
