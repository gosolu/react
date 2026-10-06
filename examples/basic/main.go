package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gosolu/react/agent"
	"github.com/gosolu/react/llm/mock"
	"github.com/gosolu/react/tool"
)

type addArgs struct {
	A int `json:"a"`
	B int `json:"b"`
}

func main() {
	// Simple typed tool
	add := tool.New("add", "Add two numbers", func(ctx context.Context, in addArgs) (int, error) {
		return in.A + in.B, nil
	})

	// Scripted mock provider
	p := mock.New(
		mock.ToolCalls("Thinking...", mock.Call("add", `{"a": 10, "b": 25}`)),
		mock.Text("The sum of 10 and 25 is 35."),
	)

	a, err := agent.New(
		agent.WithName("basic-agent"),
		agent.WithProvider(p),
		agent.WithTools(add),
	)
	if err != nil {
		log.Fatal(err)
	}

	res, err := a.Run(context.Background(), "session-1", "What is 10 + 25?")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Result:", res.Output)
}
