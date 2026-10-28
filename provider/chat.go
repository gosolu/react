package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"maps"
	"strings"

	"github.com/gosolu/react"
)

// BuildChatCompletionsRequestBody constructs a standard OpenAI-compatible
// /chat/completions request body.
func BuildChatCompletionsRequestBody(req *react.Request, stream bool) map[string]any {
	body := make(map[string]any)
	maps.Copy(body, req.Extra)

	body["model"] = req.Model
	body["stream"] = stream
	if stream {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	if req.MaxTokens > 0 {
		body["max_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}

	var messages []map[string]any
	if req.System != "" {
		messages = append(messages, map[string]any{
			"role":    "system",
			"content": req.System,
		})
	}

	for _, m := range req.Messages {
		switch m.Role {
		case react.RoleSystem:
			messages = append(messages, map[string]any{
				"role":    "system",
				"content": m.Text(),
			})
		case react.RoleUser:
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": buildChatContent(m),
			})
		case react.RoleAssistant:
			msgObj := map[string]any{
				"role":    "assistant",
				"content": m.Text(),
			}
			if m.Reasoning != "" {
				msgObj["reasoning_content"] = m.Reasoning
			}
			if len(m.ToolCalls) > 0 {
				tcs := make([]map[string]any, 0, len(m.ToolCalls))
				for _, tc := range m.ToolCalls {
					tcMap := map[string]any{
						"id":   tc.ID,
						"type": "function",
						"function": map[string]any{
							"name":      tc.Name,
							"arguments": string(tc.Arguments),
						},
					}
					if tc.ThoughtSignature != "" {
						tcMap["extra_content"] = map[string]any{
							"google": map[string]any{
								"thought_signature": tc.ThoughtSignature,
							},
						}
					}
					tcs = append(tcs, tcMap)
				}
				msgObj["tool_calls"] = tcs
			}
			messages = append(messages, msgObj)
		case react.RoleTool:
			toolMsg := map[string]any{
				"role":         "tool",
				"tool_call_id": m.ToolCallID,
				"content":      buildChatContent(m),
			}
			if m.Name != "" {
				toolMsg["name"] = m.Name
			}
			messages = append(messages, toolMsg)
		}
	}
	body["messages"] = messages

	if len(req.Tools) > 0 {
		tools := make([]map[string]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  t.InputSchema,
				},
			})
		}
		body["tools"] = tools

		switch req.ToolChoice {
		case react.ToolChoiceAuto, react.ToolChoiceNone, react.ToolChoiceRequired:
			body["tool_choice"] = req.ToolChoice
		case "":
			// omit, defaults to auto
		default:
			body["tool_choice"] = map[string]any{
				"type":     "function",
				"function": map[string]any{"name": req.ToolChoice},
			}
		}
	}

	return body
}

func buildChatContent(m react.Message) any {
	if !HasImageParts(m.Content) {
		return m.Text()
	}

	parts := make([]map[string]any, 0, len(m.Content))
	for _, p := range m.Content {
		switch p.Type {
		case react.PartText:
			parts = append(parts, map[string]any{"type": "text", "text": p.Text})
		case react.PartImage:
			parts = append(parts, map[string]any{
				"type":      "image_url",
				"image_url": map[string]string{"url": ResolveImageURL(p)},
			})
		}
	}
	return parts
}

// MapFinishReason maps an OpenAI-compatible finish reason string to react.StopReason.
func MapFinishReason(r string) react.StopReason {
	switch r {
	case "stop":
		return react.StopEnd
	case "tool_calls", "function_call":
		return react.StopToolCall
	case "length":
		return react.StopLength
	case "content_filter":
		return react.StopContentFilter
	default:
		return react.StopOther
	}
}

type chatExtraContentWire struct {
	Google *struct {
		ThoughtSignature string `json:"thought_signature"`
	} `json:"google"`
}

func (e *chatExtraContentWire) thoughtSignature() string {
	if e != nil && e.Google != nil {
		return e.Google.ThoughtSignature
	}
	return ""
}

type chatUsageWire struct {
	PromptTokens          int `json:"prompt_tokens"`
	CompletionTokens      int `json:"completion_tokens"`
	PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
	PromptTokensDetails   *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func (u *chatUsageWire) toUsage() react.Usage {
	if u == nil {
		return react.Usage{}
	}
	cached := u.PromptCacheHitTokens
	if cached == 0 && u.PromptTokensDetails != nil {
		cached = u.PromptTokensDetails.CachedTokens
	}
	return react.Usage{
		InputTokens:      u.PromptTokens,
		OutputTokens:     u.CompletionTokens,
		CacheReadTokens:  cached,
		CacheWriteTokens: u.PromptCacheMissTokens,
	}
}

// ChatCompletionResponse represents an OpenAI-compatible /chat/completions response body.
type ChatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			ToolCalls        []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
				ExtraContent *chatExtraContentWire `json:"extra_content"`
			} `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage chatUsageWire `json:"usage"`
}

// ChatCompletionChunk represents a single SSE chunk from /chat/completions.
type ChatCompletionChunk struct {
	Choices []struct {
		Delta struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			ToolCalls        []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
				ExtraContent *chatExtraContentWire `json:"extra_content"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *chatUsageWire `json:"usage"`
}

// ParseChatCompletionsResponse parses a non-streaming /chat/completions response.
func ParseChatCompletionsResponse(r io.Reader) (*react.Response, error) {
	var chatResp ChatCompletionResponse
	if err := json.NewDecoder(r).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("chat completions: decode response: %w", err)
	}
	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("chat completions: empty choices returned")
	}

	choice := chatResp.Choices[0]
	msg := react.Message{
		Role:      react.RoleAssistant,
		Reasoning: choice.Message.ReasoningContent,
	}
	if choice.Message.Content != "" {
		msg.Content = []react.Part{react.TextPart(choice.Message.Content)}
	}
	for _, tc := range choice.Message.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, react.ToolCall{
			ID:               tc.ID,
			Name:             tc.Function.Name,
			Arguments:        json.RawMessage(tc.Function.Arguments),
			ThoughtSignature: tc.ExtraContent.thoughtSignature(),
		})
	}

	stopReason := MapFinishReason(choice.FinishReason)
	if len(msg.ToolCalls) > 0 && stopReason == react.StopEnd {
		stopReason = react.StopToolCall
	}

	return &react.Response{
		Message:    msg,
		Reasoning:  choice.Message.ReasoningContent,
		StopReason: stopReason,
		Usage:      chatResp.Usage.toUsage(),
	}, nil
}

// StreamChatCompletions parses an SSE stream of /chat/completions chunks and yields react.StreamEvents.
func StreamChatCompletions(ctx context.Context, r io.Reader) iter.Seq2[react.StreamEvent, error] {
	return StreamChatCompletionsEvents(ctx, ReadSSE(r))
}

// StreamChatCompletionsEvents parses a sequence of SSEEvents from /chat/completions and yields react.StreamEvents.
func StreamChatCompletionsEvents(ctx context.Context, events iter.Seq2[SSEEvent, error]) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		seenTools := make(map[int]bool)
		var finishedCleanly bool

		for sse, err := range events {
			if err != nil {
				if err != io.EOF && ctx.Err() == nil {
					yield(react.StreamEvent{}, err)
				}
				return
			}
			if sse.Data == "[DONE]" {
				return
			}
			if err := CheckSSEError(sse, "provider stream error"); err != nil {
				yield(react.StreamEvent{}, err)
				return
			}

			var chunk ChatCompletionChunk
			if err := json.Unmarshal([]byte(sse.Data), &chunk); err != nil {
				if s := strings.TrimSpace(sse.Data); s != "" && !strings.HasPrefix(s, ":") {
					yield(react.StreamEvent{}, fmt.Errorf("provider stream invalid json: %w: %s", err, s))
					return
				}
				continue
			}
			if chunk.Usage != nil {
				u := chunk.Usage.toUsage()
				if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &u}, nil) {
					return
				}
			}
			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			if choice.Delta.ReasoningContent != "" {
				if !yield(react.StreamEvent{Type: react.StreamReasoningDelta, Text: choice.Delta.ReasoningContent}, nil) {
					return
				}
			}
			if choice.Delta.Content != "" {
				if !yield(react.StreamEvent{Type: react.StreamTextDelta, Text: choice.Delta.Content}, nil) {
					return
				}
			}
			for _, tc := range choice.Delta.ToolCalls {
				idx := tc.Index
				sig := tc.ExtraContent.thoughtSignature()
				if !seenTools[idx] {
					seenTools[idx] = true
					if !yield(react.StreamEvent{
						Type:       react.StreamToolCallStart,
						Index:      idx,
						ToolCallID: tc.ID,
						ToolName:   tc.Function.Name,
						Signature:  sig,
						ArgsDelta:  tc.Function.Arguments,
					}, nil) {
						return
					}
				} else if tc.Function.Arguments != "" || sig != "" {
					if !yield(react.StreamEvent{
						Type:      react.StreamToolCallDelta,
						Index:     idx,
						Signature: sig,
						ArgsDelta: tc.Function.Arguments,
					}, nil) {
						return
					}
				}
			}
			if choice.FinishReason != "" {
				finishedCleanly = true
				stopReason := MapFinishReason(choice.FinishReason)
				if len(seenTools) > 0 && stopReason == react.StopEnd {
					stopReason = react.StopToolCall
				}
				if !yield(react.StreamEvent{
					Type:       react.StreamDone,
					StopReason: stopReason,
				}, nil) {
					return
				}
			}
		}

		if !finishedCleanly && ctx.Err() == nil {
			yield(react.StreamEvent{}, io.ErrUnexpectedEOF)
		}
	}
}
