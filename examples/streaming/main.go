package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gosolu/react/agent"
	"github.com/gosolu/react/llm/mock"
	"github.com/gosolu/react/tool"
)

type weatherArgs struct {
	City string `json:"city"`
}

func main() {
	weather := tool.New("get_weather", "Check weather in city", func(ctx context.Context, in weatherArgs) (string, error) {
		return fmt.Sprintf("The weather in %s is 22°C and sunny.", in.City), nil
	})

	p := mock.New(
		mock.ToolCalls("Checking...", mock.Call("get_weather", `{"city": "Tokyo"}`)),
		mock.Text("Tokyo is currently 22°C and sunny!"),
	)
	p.ChunkSize = 2 // stream in small chunks

	a, err := agent.New(
		agent.WithName("stream-agent"),
		agent.WithProvider(p),
		agent.WithTools(weather),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Starting stream...")
	for ev, err := range a.Stream(context.Background(), "session-stream", "Tokyo weather?") {
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
		case agent.EventRunEnd:
			fmt.Println("\n\nFinished in turns:", ev.Result.Turns)
		}
	}
}
