package provider

import (
	"context"
	"encoding/base64"
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
			hasImages := false
			for _, p := range m.Content {
				if p.Type == react.PartImage {
					hasImages = true
					break
				}
			}
			var content any = m.Text()
			if hasImages {
				var parts []map[string]any
				for _, p := range m.Content {
					switch p.Type {
					case react.PartText:
						parts = append(parts, map[string]any{"type": "text", "text": p.Text})
					case react.PartImage:
						url := p.ImageURL
						if url == "" && len(p.Data) > 0 {
							mime := p.MIMEType
							if mime == "" {
								mime = "image/jpeg"
							}
							url = fmt.Sprintf("data:%s;base64,%s", mime, base64.StdEncoding.EncodeToString(p.Data))
						}
						parts = append(parts, map[string]any{
							"type": "image_url",
							"image_url": map[string]string{
								"url": url,
							},
						})
					}
				}
				content = parts
			}
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": content,
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
				var tcs []map[string]any
				for _, tc := range m.ToolCalls {
					tcs = append(tcs, map[string]any{
						"id":   tc.ID,
						"type": "function",
						"function": map[string]any{
							"name":      tc.Name,
							"arguments": string(tc.Arguments),
						},
					})
				}
				msgObj["tool_calls"] = tcs
			}
			messages = append(messages, msgObj)
		case react.RoleTool:
			var toolContent any = m.Text()
			var hasImages bool
			for _, p := range m.Content {
				if p.Type == react.PartImage {
					hasImages = true
					break
				}
			}
			if hasImages {
				var parts []map[string]any
				for _, p := range m.Content {
					switch p.Type {
					case react.PartText:
						parts = append(parts, map[string]any{"type": "text", "text": p.Text})
					case react.PartImage:
						url := p.ImageURL
						if url == "" && len(p.Data) > 0 {
							mime := p.MIMEType
							if mime == "" {
								mime = "image/jpeg"
							}
							url = fmt.Sprintf("data:%s;base64,%s", mime, base64.StdEncoding.EncodeToString(p.Data))
						}
						parts = append(parts, map[string]any{
							"type": "image_url",
							"image_url": map[string]string{
								"url": url,
							},
						})
					}
				}
				toolContent = parts
			}
			messages = append(messages, map[string]any{
				"role":         "tool",
				"tool_call_id": m.ToolCallID,
				"content":      toolContent,
			})
		}
	}
	body["messages"] = messages

	if len(req.Tools) > 0 {
		var tools []map[string]any
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
				"type": "function",
				"function": map[string]any{
					"name": req.ToolChoice,
				},
			}
		}
	}

	return body
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
			} `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens          int `json:"prompt_tokens"`
		CompletionTokens      int `json:"completion_tokens"`
		PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
		PromptTokensDetails   struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
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
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens          int `json:"prompt_tokens"`
		CompletionTokens      int `json:"completion_tokens"`
		PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
		PromptTokensDetails   *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
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
		Role: react.RoleAssistant,
	}
	if choice.Message.Content != "" {
		msg.Content = []react.Part{react.TextPart(choice.Message.Content)}
	}
	for _, tc := range choice.Message.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, react.ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: json.RawMessage(tc.Function.Arguments),
		})
	}
	if choice.Message.ReasoningContent != "" {
		msg.Reasoning = choice.Message.ReasoningContent
	}

	cached := chatResp.Usage.PromptCacheHitTokens
	if cached == 0 {
		cached = chatResp.Usage.PromptTokensDetails.CachedTokens
	}

	return &react.Response{
		Message:    msg,
		Reasoning:  choice.Message.ReasoningContent,
		StopReason: MapFinishReason(choice.FinishReason),
		Usage: react.Usage{
			InputTokens:      chatResp.Usage.PromptTokens,
			OutputTokens:     chatResp.Usage.CompletionTokens,
			CacheReadTokens:  cached,
			CacheWriteTokens: chatResp.Usage.PromptCacheMissTokens,
		},
	}, nil
}

// StreamChatCompletions parses an SSE stream of /chat/completions chunks and yields react.StreamEvents.
func StreamChatCompletions(ctx context.Context, r io.Reader) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		seenTools := make(map[int]bool)
		var finishedCleanly bool

		for sse, err := range ReadSSE(r) {
			if err != nil {
				if err != io.EOF && ctx.Err() == nil {
					yield(react.StreamEvent{}, err)
				}
				return
			}
			if sse.Data == "[DONE]" {
				return
			}

			// Check for mid-stream error events or error JSON frames
			if sse.Event == "error" || strings.Contains(sse.Data, `"error"`) {
				var errPayload struct {
					Error *struct {
						Message string `json:"message"`
						Type    string `json:"type"`
						Code    any    `json:"code"`
					} `json:"error"`
				}
				if json.Unmarshal([]byte(sse.Data), &errPayload) == nil && errPayload.Error != nil && errPayload.Error.Message != "" {
					msg := errPayload.Error.Message
					if errPayload.Error.Type != "" {
						msg = fmt.Sprintf("%s (%s)", msg, errPayload.Error.Type)
					}
					yield(react.StreamEvent{}, fmt.Errorf("provider stream error: %s", msg))
					return
				}
				if sse.Event == "error" {
					yield(react.StreamEvent{}, fmt.Errorf("provider stream error: %s", sse.Data))
					return
				}
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
				cached := chunk.Usage.PromptCacheHitTokens
				if cached == 0 && chunk.Usage.PromptTokensDetails != nil {
					cached = chunk.Usage.PromptTokensDetails.CachedTokens
				}
				u := react.Usage{
					InputTokens:      chunk.Usage.PromptTokens,
					OutputTokens:     chunk.Usage.CompletionTokens,
					CacheReadTokens:  cached,
					CacheWriteTokens: chunk.Usage.PromptCacheMissTokens,
				}
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
				if !seenTools[idx] {
					seenTools[idx] = true
					if !yield(react.StreamEvent{
						Type:       react.StreamToolCallStart,
						Index:      idx,
						ToolCallID: tc.ID,
						ToolName:   tc.Function.Name,
						ArgsDelta:  tc.Function.Arguments,
					}, nil) {
						return
					}
				} else if tc.Function.Arguments != "" {
					if !yield(react.StreamEvent{
						Type:      react.StreamToolCallDelta,
						Index:     idx,
						ArgsDelta: tc.Function.Arguments,
					}, nil) {
						return
					}
				}
			}
			if choice.FinishReason != "" {
				finishedCleanly = true
				if !yield(react.StreamEvent{
					Type:       react.StreamDone,
					StopReason: MapFinishReason(choice.FinishReason),
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
