# ReAct: Go Agent Core

A modular, minimal-dependency AI agent core written from scratch in Go.

Features a unified ReAct (Reasoning + Acting) loop with support for both streaming and non-streaming modes, typed tools with automatic JSON Schema generation, progressive disclosure skills, in-process plugins, lifecycle hooks, session memory, permission policies, and sub-agents.

---

## Features

- **Unified ReAct Engine:** A single loop powers both non-streaming (`Run`) and streaming (`Stream`) execution.
- **Go 1.23+ Iterators:** Streaming exposes range-over-func iterators (`iter.Seq2[Event, error]`) with cancellation on break.
- **Zero Heavy SDK Dependencies:** Standard library HTTP clients and SSE parsers for OpenAI and Anthropic. Only one external dependency (`gopkg.in/yaml.v3`) for parsing skill frontmatter.
- **Provider Abstraction:**
  - `llm/openai`: Works with OpenAI and OpenAI-compatible endpoints (Ollama, vLLM, DeepSeek, LocalAI, etc.).
  - `llm/anthropic`: Anthropic Messages API.
  - `llm/mock`: Scripted provider for deterministic testing and offline development.
- **Type-safe Tools (`tool`):**
  - Generics-based tool definitions (`tool.New[In, Out](...)`).
  - Automatic JSON Schema generation via reflection with support for struct tags (`desc`, `enum`, `omitempty`, `required`).
  - Parallel tool execution with semaphore bounds, per-tool timeouts, and panic recovery.
- **Progressive Disclosure Skills (`skill`):**
  - `SKILL.md` folders with YAML frontmatter.
  - Only the catalog is advertised up-front; full instructions are lazily loaded on demand via `activate_skill`.
  - Directory-traversal protected `read_skill_file` tool.
- **In-Process Plugins (`plugin`):**
  - Modular plugins can register tools, middleware, skills, hooks, and system prompt fragments.
- **Lifecycle Hooks (`hook`):**
  - Granular hooks for `BeforeModel`, `AfterModel`, `BeforeTool`, `AfterTool`, `OnEvent`, and run lifecycle.
- **Pluggable Session Memory (`memory`):**
  - Built-in concurrency-safe `InMemory` store.
  - Open `memory.Store` interface to connect custom backends (Redis, PostgreSQL, etc.). Includes `memory/storetest` conformance test suite.
- **Context Window Management (`contextmgr`):**
  - Sliding window (`Window`) that preserves tool-call / tool-result message integrity.
  - LLM-based summarizer (`Summarize`) with caching.
- **Permission & Safety (`permission`):**
  - Rule-based policy (`ReadOnly` auto-allowed, `Destructive` requires approval).
  - Human-in-the-loop approval interface (`permission.Approver`).
- **Sub-Agents (`agent.AsTool`):**
  - Expose any agent as a tool to parent agents with depth tracking and forwarded event streams.
- **CLI REPL (`cmd/react`):**
  - Ready-to-use terminal interface with live streaming, slash commands (`/tools`, `/skills`, `/reset`, `/exit`), and interactive `[y/N]` approval prompts for destructive tools.

---

## Directory Structure

```
react/
├── agent/            # Core ReAct loop, events, options, sub-agent tool
├── cmd/react/        # Interactive CLI REPL application
├── contextmgr/       # Context window sliding window & summarization
├── examples/
│   ├── basic/        # Minimal non-streaming example
│   ├── plugin/       # In-process plugin & skill folder example
│   └── streaming/    # Streaming example using iter.Seq2
├── hook/             # RunContext and lifecycle hooks
├── llm/              # Core message types, Provider interface, SSE reader, accumulator
│   ├── anthropic/    # Anthropic Messages API client
│   ├── mock/         # Scripted mock provider for testing
│   └── openai/       # OpenAI-compatible API client
├── memory/           # Session store interface and in-memory implementation
│   └── storetest/    # Conformance test suite for custom store implementations
├── permission/       # Permission policies and human approver interfaces
├── plugin/           # Plugin interface & Registrar
├── skill/            # SKILL.md parser, registry, and activation tools
├── tool/             # Generic tools, reflection JSON schema, executor, middleware
└── tools/builtin/    # Standard tools: read_file, write_file, list_dir, shell
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

	"github.com/gosolu/react/agent"
	"github.com/gosolu/react/llm/openai"
	"github.com/gosolu/react/tools/builtin"
)

func main() {
	provider := openai.New(openai.Config{
		APIKey: os.Getenv("OPENAI_API_KEY"),
	})

	a, err := agent.New(
		agent.WithProvider(provider),
		agent.WithModel("gpt-4o"),
		agent.WithTools(builtin.ReadFile(), builtin.ListDir()),
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
		case agent.EventTextDelta:
			fmt.Print(ev.Text)
		case agent.EventToolCallStart:
			fmt.Printf("\n[Calling tool: %s]\n", ev.ToolCall.Name)
		case agent.EventToolResult:
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

calcTool := tool.New("calc", "Performs basic arithmetic", func(ctx context.Context, in CalculatorInput) (float64, error) {
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
}, tool.ReadOnly())
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
a, err := agent.New(
    agent.WithProvider(provider),
    agent.WithSkillDirs("./skills"),
)
```

The model automatically receives catalog summaries in its system prompt and calls `activate_skill("code-reviewer")` when relevant.

---

## Writing an In-Process Plugin

Plugins implement `plugin.Plugin`:

```go
type MyPlugin struct{}

func (p *MyPlugin) Name() string { return "my_plugin" }

func (p *MyPlugin) Init(ctx context.Context, r plugin.Registrar) error {
    // Add tools
    r.AddTool(myCustomTool)

    // Add lifecycle hooks
    r.AddHooks(hook.Hooks{
        BeforeTool: func(ctx context.Context, rc *hook.RunContext, call *llm.ToolCall) (hook.ToolDecision, error) {
            log.Printf("Executing tool %s in turn %d", call.Name, rc.Turn)
            return hook.ToolDecision{Action: hook.Continue}, nil
        },
    })

    // Add system prompt fragments
    r.AddSystemPrompt("Always prioritize concise answers.")
    return nil
}
```

---

## Sub-Agents (Agent-as-a-Tool)

Delegate complex tasks to specialized child agents:

```go
researcher, _ := agent.New(
    agent.WithName("researcher"),
    agent.WithProvider(provider),
    agent.WithTools(builtin.ReadFile(), builtin.ListDir()),
)

mainAgent, _ := agent.New(
    agent.WithName("main"),
    agent.WithProvider(provider),
    agent.WithTools(
        researcher.AsTool("research_codebase", "Delegate codebase research tasks to a specialist agent"),
    ),
)
```

---

## Running the REPL CLI

The built-in CLI supports OpenAI, Anthropic, and Mock providers:

```bash
# Using OpenAI (or compatible API)
export OPENAI_API_KEY="your-api-key"
go run ./cmd/react --provider openai --model gpt-4o

# Using Ollama / LocalAI (OpenAI-compatible)
go run ./cmd/react --provider openai --model llama3 --base-url http://localhost:11434/v1

# Using Anthropic
export ANTHROPIC_API_KEY="your-api-key"
go run ./cmd/react --provider anthropic --model claude-3-5-sonnet-20241022

# Offline testing with mock provider
go run ./cmd/react --provider mock
```

Interactive CLI commands:
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
