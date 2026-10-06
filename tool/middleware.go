package tool

import (
	"context"
	"fmt"
	"time"

	"github.com/gosolu/react/llm"
)

// Handler executes a tool call.
type Handler func(ctx context.Context, call llm.ToolCall) (Result, error)

// Middleware wraps a Handler.
type Middleware func(next Handler) Handler

// Chain wraps h with mws; the first middleware is the outermost.
func Chain(h Handler, mws ...Middleware) Handler {
	for i := len(mws) - 1; i >= 0; i-- {
		h = mws[i](h)
	}
	return h
}

// TimeoutMiddleware bounds every call to d.
func TimeoutMiddleware(d time.Duration) Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, call llm.ToolCall) (Result, error) {
			ctx, cancel := context.WithTimeout(ctx, d)
			defer cancel()
			return next(ctx, call)
		}
	}
}

// TruncateMiddleware caps each text part of a result at maxBytes.
func TruncateMiddleware(maxBytes int) Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, call llm.ToolCall) (Result, error) {
			res, err := next(ctx, call)
			for i, p := range res.Content {
				if p.Type == llm.PartText && len(p.Text) > maxBytes {
					res.Content[i].Text = fmt.Sprintf("%s\n...[truncated %d bytes]", p.Text[:maxBytes], len(p.Text)-maxBytes)
				}
			}
			return res, err
		}
	}
}
