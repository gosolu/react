package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider/mock"
)

type addArgs struct {
	A int `json:"a"`
	B int `json:"b"`
}

func main() {
	// Simple typed tool
	add := react.NewTool("add", "Add two numbers", func(ctx context.Context, in addArgs) (int, error) {
		return in.A + in.B, nil
	})

	// Scripted mock provider
	p := mock.New(
		mock.ToolCalls("Thinking...", mock.Call("add", `{"a": 10, "b": 25}`)),
		mock.Text("The sum of 10 and 25 is 35."),
	)

	a, err := react.New(
		react.WithName("basic-agent"),
		react.WithProvider(p),
		react.WithTools(add),
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
