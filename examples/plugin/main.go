package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gosolu/react/agent"
	"github.com/gosolu/react/hook"
	"github.com/gosolu/react/llm"
	"github.com/gosolu/react/llm/mock"
	"github.com/gosolu/react/plugin"
	"github.com/gosolu/react/skill"
	"github.com/gosolu/react/tool"
)

// AnalyticsPlugin is an in-process plugin registering tools, hooks, and skill sources.
type AnalyticsPlugin struct{}

func (p *AnalyticsPlugin) Name() string { return "analytics" }

func (p *AnalyticsPlugin) Init(ctx context.Context, r plugin.Registrar) error {
	// Register custom tool
	calcTool := tool.New("multiply", "Multiplies two numbers", func(ctx context.Context, in struct{ A, B int }) (int, error) {
		return in.A * in.B, nil
	})
	if err := r.AddTool(calcTool); err != nil {
		return err
	}

	// Register lifecycle hook
	r.AddHooks(hook.Hooks{
		BeforeTool: func(ctx context.Context, rc *hook.RunContext, call *llm.ToolCall) (hook.ToolDecision, error) {
			fmt.Printf("[Plugin Hook] Preparing to run tool: %s\n", call.Name)
			return hook.ToolDecision{Action: hook.Continue}, nil
		},
	})

	// Register skill directory
	r.AddSkillSource(skill.Dir("./examples/plugin/skills"))
	return nil
}

func main() {
	p := mock.New(
		mock.ToolCalls("Using tool...", mock.Call("multiply", `{"A": 6, "B": 7}`)),
		mock.Text("6 * 7 = 42."),
	)

	a, err := agent.New(
		agent.WithName("plugin-agent"),
		agent.WithProvider(p),
		agent.WithPlugins(&AnalyticsPlugin{}),
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
