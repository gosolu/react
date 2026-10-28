package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"

	"github.com/gosolu/react"
	"github.com/gosolu/react/pkg/utf8util"
	"github.com/gosolu/react/provider"
	"github.com/gosolu/react/provider/anthropic"
	"github.com/gosolu/react/provider/deepseek"
	"github.com/gosolu/react/provider/gemini"
	"github.com/gosolu/react/provider/mock"
	"github.com/gosolu/react/provider/openai"
	"github.com/gosolu/react/provider/opencode"
)

func main() {
	providerFlag := flag.String("provider", "mock", "LLM provider: openai | anthropic | deepseek | opencode | gemini | mock")
	apiFlag := flag.String("api", "completions", "API protocol: completions | responses")
	modelFlag := flag.String("model", "", "Model identifier (defaults to provider standard)")
	baseURLFlag := flag.String("base-url", "", "Custom BaseURL (e.g. for Ollama / vLLM / local AI)")
	streamFlag := flag.Bool("stream", true, "Enable streaming responses")
	skillsDirFlag := flag.String("skills", "", "Path to skills directory")
	sessionFlag := flag.String("session", "default-repl", "Session ID")
	storeDirFlag := flag.String("store-dir", "", "Directory to persist conversation sessions across CLI restarts")
	systemPromptFlag := flag.String("system", "You are a helpful and capable agent assistant.", "Base system prompt")
	flag.Parse()

	var p react.Provider
	model := *modelFlag

	apiType := provider.APITypeChatCompletions
	if strings.EqualFold(*apiFlag, "responses") {
		apiType = provider.APITypeResponses
	} else if strings.EqualFold(*apiFlag, "interactions") || strings.EqualFold(*apiFlag, "interaction") {
		apiType = provider.APITypeInteractions
	} else if strings.EqualFold(*apiFlag, "generate_content") || strings.EqualFold(*apiFlag, "generateContent") {
		apiType = provider.APITypeGenerateContent
	}

	cfg := provider.Config{
		BaseURL: *baseURLFlag,
		APIType: apiType,
	}

	switch strings.ToLower(*providerFlag) {
	case "openai":
		cfg.APIKey = os.Getenv("OPENAI_API_KEY")
		if cfg.APIKey == "" {
			fmt.Println("Warning: OPENAI_API_KEY environment variable is not set.")
		}
		if model == "" {
			model = "gpt-4o"
		}
		p = openai.New(cfg)
	case "anthropic":
		cfg.APIKey = os.Getenv("ANTHROPIC_API_KEY")
		if cfg.APIKey == "" {
			fmt.Println("Warning: ANTHROPIC_API_KEY environment variable is not set.")
		}
		if model == "" {
			model = "claude-3-5-sonnet-20241022"
		}
		p = anthropic.New(cfg)
	case "deepseek":
		cfg.APIKey = os.Getenv("DEEPSEEK_API_KEY")
		if cfg.APIKey == "" {
			fmt.Println("Warning: DEEPSEEK_API_KEY environment variable is not set.")
		}
		if model == "" {
			model = deepseek.ModelChat
		}
		p = deepseek.New(cfg)
	case "opencode", "opencode-go":
		cfg.APIKey = os.Getenv("OPENCODE_API_KEY")
		if cfg.APIKey == "" {
			cfg.APIKey = os.Getenv("OPENCODE_GO_API_KEY")
		}
		if cfg.APIKey == "" {
			fmt.Println("Warning: OPENCODE_API_KEY environment variable is not set.")
		}
		if model == "" {
			model = opencode.ModelDeepSeekV4Pro
		}
		p = opencode.New(cfg)
	case "gemini":
		cfg.APIKey = os.Getenv("GEMINI_API_KEY")
		if cfg.APIKey == "" {
			cfg.APIKey = os.Getenv("GOOGLE_API_KEY")
		}
		if cfg.APIKey == "" {
			fmt.Println("Warning: GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is not set.")
		}
		if model == "" {
			model = gemini.ModelGemini38Flash
		}
		p = gemini.New(cfg)
	case "mock":
		if model == "" {
			model = "mock-model"
		}
		p = mock.New(
			mock.Text("Hello! I am running in mock offline mode. Configure --provider openai, anthropic, deepseek, opencode, or gemini for live models."),
		)
	default:
		fmt.Fprintf(os.Stderr, "Unknown provider: %s\n", *providerFlag)
		os.Exit(1)
	}

	reader := bufio.NewReader(os.Stdin)
	sessionID := *sessionFlag

	// Interactive CLI approval for destructive tools with session memory
	var (
		remember   *react.SessionRememberApprover
		approvalMu sync.Mutex
	)
	baseApprover := react.PermissionApproverFunc(func(ctx context.Context, req react.ApprovalRequest) (bool, error) {
		approvalMu.Lock()
		defer approvalMu.Unlock()
		if err := ctx.Err(); err != nil {
			return false, err
		}
		fmt.Printf("\n⚠️  [APPROVAL NEEDED] Tool: %s\nArguments: %s\nApprove execution? [y/N/a (always for session)]: ", req.Call.Name, string(req.Call.Arguments))
		line, err := reader.ReadString('\n')
		if err != nil {
			return false, err
		}
		line = strings.TrimSpace(strings.ToLower(line))
		if line == "a" || line == "always" {
			if remember != nil {
				remember.RememberForSession(sessionID, req.Call.Name)
			}
			return true, nil
		}
		return line == "y" || line == "yes", nil
	})
	remember = react.NewSessionRememberApprover(baseApprover)

	opts := []react.Option{
		react.WithName("react-cli"),
		react.WithProvider(p),
		react.WithModel(model),
		react.WithSystemPrompt(*systemPromptFlag),
		react.WithPermission(react.DefaultPermissionPolicy(), remember),
	}

	if *storeDirFlag != "" {
		fs, err := react.NewFileStore(*storeDirFlag)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to initialize store directory %q: %v\n", *storeDirFlag, err)
			os.Exit(1)
		}
		opts = append(opts, react.WithStore(fs))
	}

	if *skillsDirFlag != "" {
		opts = append(opts, react.WithSkillDirs(*skillsDirFlag))
	}

	a, err := react.New(opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create agent: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = a.Close() }()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Println("==================================================")
	fmt.Printf(" ReAct Agent REPL (Provider: %s, Model: %s, Stream: %v)\n", p.Name(), model, *streamFlag)
	fmt.Println(" Commands: /tools, /skills, /reset, /exit")
	fmt.Println("==================================================")

	for ctx.Err() == nil {
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
				case react.EventTextDelta:
					fmt.Print(ev.Text)
				case react.EventToolCallStart:
					fmt.Printf("\n⚙️  Calling [%s]...\n", ev.ToolCall.Name)
				case react.EventToolResult:
					status := "✓"
					if ev.ToolResult.IsError {
						status = "✗ (error)"
					}
					fmt.Printf("   %s [%s]: %s\n", status, ev.ToolCall.Name, summarize(ev.ToolResult.Text()))
				case react.EventRunEnd:
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
		return utf8util.TruncateString(s, 97) + "..."
	}
	return s
}
