package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gosolu/react"
	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/llm/mock"
)

// AnalyticsPlugin is an in-process plugin registering tools, hooks, and skill sources.
type AnalyticsPlugin struct{}

func (p *AnalyticsPlugin) Name() string { return "analytics" }

func (p *AnalyticsPlugin) Init(ctx context.Context, r react.Registrar) error {
	// Register custom tool
	calcTool := react.NewTool("multiply", "Multiplies two numbers", func(ctx context.Context, in struct{ A, B int }) (int, error) {
		return in.A * in.B, nil
	})
	if err := r.AddTool(calcTool); err != nil {
		return err
	}

	// Register lifecycle hook
	r.AddHooks(react.Hooks{
		BeforeTool: func(ctx context.Context, rc *react.RunContext, call *llm.ToolCall) (react.HookDecision, error) {
			fmt.Printf("[Plugin Hook] Preparing to run tool: %s\n", call.Name)
			return react.HookDecision{Action: react.HookContinue}, nil
		},
	})

	// Register skill directory
	r.AddSkillSource(react.SkillDir("./examples/plugin/skills"))
	return nil
}

func main() {
	p := mock.New(
		mock.ToolCalls("Using tool...", mock.Call("multiply", `{"A": 6, "B": 7}`)),
		mock.Text("6 * 7 = 42."),
	)

	a, err := react.New(
		react.WithName("plugin-agent"),
		react.WithProvider(p),
		react.WithPlugins(&AnalyticsPlugin{}),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Available skills in agent:")
	for _, s := range a.Skills() {
		fmt.Printf(" - %s: %s\n", s.Name, s.Description)
	}

	res, err := a.Run(context.Background(), "s1", "calculate 6*7")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Agent output:", res.Output)
}
