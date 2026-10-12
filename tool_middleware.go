package react

import (
	"context"
	"fmt"
	"slices"
	"time"
)

// ToolHandler executes a tool call.
type ToolHandler func(ctx context.Context, call ToolCall) (ToolResult, error)

// ToolMiddleware wraps a ToolHandler.
type ToolMiddleware func(next ToolHandler) ToolHandler

// ChainToolMiddleware wraps h with mws; the first middleware is the outermost.
func ChainToolMiddleware(h ToolHandler, mws ...ToolMiddleware) ToolHandler {
	for _, mw := range slices.Backward(mws) {
		h = mw(h)
	}
	return h
}

// TimeoutMiddleware bounds every call to d.
func TimeoutMiddleware(d time.Duration) ToolMiddleware {
	return func(next ToolHandler) ToolHandler {
		return func(ctx context.Context, call ToolCall) (ToolResult, error) {
			ctx, cancel := context.WithTimeout(ctx, d)
			defer cancel()
			return next(ctx, call)
		}
	}
}

// TruncateMiddleware caps each text part of a result at maxBytes.
func TruncateMiddleware(maxBytes int) ToolMiddleware {
	return func(next ToolHandler) ToolHandler {
		return func(ctx context.Context, call ToolCall) (ToolResult, error) {
			res, err := next(ctx, call)
			for i, p := range res.Content {
				if p.Type == PartText && len(p.Text) > maxBytes {
					res.Content[i].Text = fmt.Sprintf("%s\n...[truncated %d bytes]", p.Text[:maxBytes], len(p.Text)-maxBytes)
				}
			}
			return res, err
		}
	}
}
