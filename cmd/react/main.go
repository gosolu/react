package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/gosolu/react/agent"
	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/llm/anthropic"
	"github.com/gosolu/react/llm/mock"
	"github.com/gosolu/react/llm/openai"
	"github.com/gosolu/react/permission"
	"github.com/gosolu/react/tools/builtin"
)

func main() {
	providerFlag := flag.String("provider", "mock", "LLM provider: openai | anthropic | mock")
	modelFlag := flag.String("model", "", "Model identifier (defaults to provider standard)")
	baseURLFlag := flag.String("base-url", "", "Custom BaseURL (e.g. for Ollama / vLLM / local AI)")
	streamFlag := flag.Bool("stream", true, "Enable streaming responses")
	skillsDirFlag := flag.String("skills", "", "Path to skills directory")
	sessionFlag := flag.String("session", "default-repl", "Session ID")
	systemPromptFlag := flag.String("system", "You are a helpful and capable agent assistant equipped with system tools.", "Base system prompt")
	flag.Parse()

	var p llm.Provider
	model := *modelFlag

	switch strings.ToLower(*providerFlag) {
	case "openai":
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			fmt.Println("Warning: OPENAI_API_KEY environment variable is not set.")
		}
		if model == "" {
			model = "gpt-4o"
		}
		p = openai.New(openai.Config{
			APIKey:  apiKey,
			BaseURL: *baseURLFlag,
		})
	case "anthropic":
		apiKey := os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			fmt.Println("Warning: ANTHROPIC_API_KEY environment variable is not set.")
		}
		if model == "" {
			model = "claude-3-5-sonnet-20241022"
		}
		p = anthropic.New(anthropic.Config{
			APIKey:  apiKey,
			BaseURL: *baseURLFlag,
		})
	case "mock":
		if model == "" {
			model = "mock-model"
		}
		p = mock.New(
			mock.Text("Hello! I am running in mock offline mode. Configure --provider openai or anthropic for live models."),
		)
	default:
		fmt.Fprintf(os.Stderr, "Unknown provider: %s\n", *providerFlag)
		os.Exit(1)
	}

	reader := bufio.NewReader(os.Stdin)

	// Interactive CLI approval for destructive tools
	approver := permission.ApproverFunc(func(ctx context.Context, req permission.Request) (bool, error) {
		fmt.Printf("\n⚠️  [APPROVAL NEEDED] Tool: %s\nArguments: %s\nApprove execution? [y/N]: ", req.Call.Name, string(req.Call.Arguments))
		line, err := reader.ReadString('\n')
		if err != nil {
			return false, err
		}
		line = strings.TrimSpace(strings.ToLower(line))
		return line == "y" || line == "yes", nil
	})

	opts := []agent.Option{
		agent.WithName("react-cli"),
		agent.WithProvider(p),
		agent.WithModel(model),
		agent.WithSystemPrompt(*systemPromptFlag),
		agent.WithTools(builtin.All()...),
		agent.WithPermission(permission.Default(), approver),
	}

	if *skillsDirFlag != "" {
		opts = append(opts, agent.WithSkillDirs(*skillsDirFlag))
	}

	a, err := agent.New(opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create agent: %v\n", err)
		os.Exit(1)
	}
	defer a.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Println("==================================================")
	fmt.Printf(" ReAct Agent REPL (Provider: %s, Model: %s, Stream: %v)\n", p.Name(), model, *streamFlag)
	fmt.Println(" Commands: /tools, /skills, /reset, /exit")
	fmt.Println("==================================================")

	sessionID := *sessionFlag

	for {
		if ctx.Err() != nil {
			break
		}
		fmt.Printf("\n(%s) > ", sessionID)
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		input := strings.TrimSpace(line)
		if input == "" {
			continue
		}

		switch input {
		case "/exit", "exit", "quit":
			fmt.Println("Bye!")
			return
		case "/tools":
			fmt.Println("Available tools:")
			for _, t := range a.Tools() {
				fmt.Printf(" - %-15s: %s\n", t.Name, t.Description)
			}
			continue
		case "/skills":
			fmt.Println("Available skills:")
			skills := a.Skills()
			if len(skills) == 0 {
				fmt.Println(" (none loaded)")
			}
			for _, s := range skills {
				fmt.Printf(" - %-15s: %s\n", s.Name, s.Description)
			}
			continue
		case "/reset":
			_ = a.Memory().Delete(ctx, sessionID)
			fmt.Printf("Session %q reset.\n", sessionID)
			continue
		}

		if *streamFlag {
			for ev, err := range a.Stream(ctx, sessionID, input) {
				if err != nil {
					fmt.Printf("\n[Error: %v]\n", err)
					break
				}
				switch ev.Type {
				case agent.EventTextDelta:
					fmt.Print(ev.Text)
				case agent.EventToolCallStart:
					fmt.Printf("\n⚙️  Calling [%s]...\n", ev.ToolCall.Name)
				case agent.EventToolResult:
					status := "✓"
					if ev.ToolResult.IsError {
						status = "✗ (error)"
					}
					fmt.Printf("   %s [%s]: %s\n", status, ev.ToolCall.Name, summarize(ev.ToolResult.Text()))
				case agent.EventRunEnd:
					fmt.Println()
				}
			}
		} else {
			res, err := a.Run(ctx, sessionID, input)
			if err != nil {
				fmt.Printf("\n[Error: %v]\n", err)
			} else {
				fmt.Printf("\n%s\n", res.Output)
			}
		}
	}
}

func summarize(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 100 {
		return s[:97] + "..."
	}
	return s
}
