package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gosolu/react"
	"github.com/gosolu/react/llm/mock"
)

type weatherArgs struct {
	City string `json:"city"`
}

func main() {
	weather := react.NewTool("get_weather", "Check weather in city", func(ctx context.Context, in weatherArgs) (string, error) {
		return fmt.Sprintf("The weather in %s is 22°C and sunny.", in.City), nil
	})

	p := mock.New(
		mock.ToolCalls("Checking...", mock.Call("get_weather", `{"city": "Tokyo"}`)),
		mock.Text("Tokyo is currently 22°C and sunny!"),
	)
	p.ChunkSize = 2 // stream in small chunks

	a, err := react.New(
		react.WithName("stream-agent"),
		react.WithProvider(p),
		react.WithTools(weather),
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
		case react.EventTextDelta:
			fmt.Print(ev.Text)
		case react.EventToolCallStart:
			fmt.Printf("\n[Calling tool: %s]\n", ev.ToolCall.Name)
		case react.EventToolResult:
			fmt.Printf("[Tool result: %s]\n", ev.ToolResult.Text())
		case react.EventRunEnd:
			fmt.Println("\n\nFinished in turns:", ev.Result.Turns)
		}
	}
}
