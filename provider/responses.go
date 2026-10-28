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

// BuildResponsesRequestBody creates a request payload for the Responses API.
// See: https://developers.openai.com/api/reference/resources/responses/methods/create
func BuildResponsesRequestBody(req *react.Request, stream bool) map[string]any {
	body := make(map[string]any)
	maps.Copy(body, req.Extra)

	body["model"] = req.Model
	if stream {
		body["stream"] = true
	}
	if req.MaxTokens > 0 {
		body["max_output_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.System != "" {
		body["instructions"] = req.System
	}

	var input []any
	for _, m := range req.Messages {
		switch m.Role {
		case react.RoleSystem:
			input = append(input, map[string]any{
				"role":    "system",
				"content": m.Text(),
			})
		case react.RoleUser:
			input = append(input, map[string]any{
				"role":    "user",
				"content": buildResponsesPartsContent(m),
			})
		case react.RoleAssistant:
			if m.Reasoning != "" {
				input = append(input, map[string]any{
					"type": "reasoning",
					"content": []map[string]any{
						{"type": "reasoning_text", "text": m.Reasoning},
					},
				})
			}
			if m.Text() != "" || len(m.ToolCalls) == 0 {
				input = append(input, map[string]any{
					"role":    "assistant",
					"content": m.Text(),
				})
			}
			for _, tc := range m.ToolCalls {
				input = append(input, map[string]any{
					"type":      "function_call",
					"call_id":   tc.ID,
					"name":      tc.Name,
					"arguments": string(tc.Arguments),
				})
			}
		case react.RoleTool:
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": m.ToolCallID,
				"output":  buildResponsesPartsContent(m),
			})
		}
	}
	body["input"] = input

	if len(req.Tools) > 0 {
		tools := make([]map[string]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, map[string]any{
				"type":        "function",
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.InputSchema,
			})
		}
		body["tools"] = tools

		if req.ToolChoice != "" {
			switch req.ToolChoice {
			case react.ToolChoiceAuto, react.ToolChoiceNone, react.ToolChoiceRequired:
				body["tool_choice"] = req.ToolChoice
			default:
				body["tool_choice"] = map[string]any{
					"type": "function",
					"name": req.ToolChoice,
				}
			}
		}
	}

	return body
}

func buildResponsesPartsContent(m react.Message) any {
	if !HasImageParts(m.Content) {
		return m.Text()
	}

	parts := make([]map[string]any, 0, len(m.Content))
	for _, part := range m.Content {
		switch part.Type {
		case react.PartText:
			parts = append(parts, map[string]any{
				"type": "input_text",
				"text": part.Text,
			})
		case react.PartImage:
			parts = append(parts, map[string]any{
				"type":      "input_image",
				"image_url": ResolveImageURL(part),
			})
		}
	}
	return parts
}

// ResponsesResponseEnvelope represents the top-level Response object from /responses.
type ResponsesResponseEnvelope struct {
	ID                string                      `json:"id"`
	Status            string                      `json:"status"`
	Output            []ResponsesOutputItem       `json:"output"`
	OutputText        string                      `json:"output_text"`
	IncompleteDetails *ResponsesIncompleteDetails `json:"incomplete_details"`
	Usage             *ResponsesUsage             `json:"usage"`
}

// ResponsesOutputItem is an item within Response.Output.
type ResponsesOutputItem struct {
	Type    string `json:"type"`
	ID      string `json:"id"`
	Role    string `json:"role"`
	Status  string `json:"status"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	// For type == "function_call"
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	// For type == "reasoning"
	Summary []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"summary"`
}

// ResponsesIncompleteDetails describes why a response did not complete normally.
type ResponsesIncompleteDetails struct {
	Reason string `json:"reason"`
}

// ResponsesUsage contains token usage details.
type ResponsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func mapResponsesStopReason(hasToolCalls bool, inc *ResponsesIncompleteDetails) react.StopReason {
	if hasToolCalls {
		return react.StopToolCall
	}
	if inc != nil {
		switch inc.Reason {
		case "max_output_tokens", "length":
			return react.StopLength
		case "content_filter":
			return react.StopContentFilter
		default:
			return react.StopOther
		}
	}
	return react.StopEnd
}

// ParseResponsesResponse decodes a JSON response body into a *react.Response.
func ParseResponsesResponse(r io.Reader) (*react.Response, error) {
	var env ResponsesResponseEnvelope
	if err := json.NewDecoder(r).Decode(&env); err != nil {
		return nil, fmt.Errorf("responses: decode response: %w", err)
	}

	msg := react.Message{Role: react.RoleAssistant}
	var textBuilder, reasoningBuilder strings.Builder

	for _, item := range env.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" || part.Text != "" {
					textBuilder.WriteString(part.Text)
				}
			}
		case "function_call":
			callID := item.CallID
			if callID == "" {
				callID = item.ID
			}
			msg.ToolCalls = append(msg.ToolCalls, react.ToolCall{
				ID:        callID,
				Name:      item.Name,
				Arguments: json.RawMessage(item.Arguments),
			})
		case "reasoning":
			for _, part := range item.Content {
				if part.Type == "reasoning_text" || part.Text != "" {
					reasoningBuilder.WriteString(part.Text)
				}
			}
			if reasoningBuilder.Len() == 0 {
				for _, part := range item.Summary {
					if part.Type == "summary_text" || part.Text != "" {
						reasoningBuilder.WriteString(part.Text)
					}
				}
			}
		}
	}

	text := textBuilder.String()
	if text == "" && env.OutputText != "" {
		text = env.OutputText
	}
	if text != "" {
		msg.Content = []react.Part{react.TextPart(text)}
	}
	msg.Reasoning = reasoningBuilder.String()

	var usage react.Usage
	if env.Usage != nil {
		usage.InputTokens = env.Usage.InputTokens
		usage.OutputTokens = env.Usage.OutputTokens
	}

	return &react.Response{
		Message:    msg,
		Reasoning:  msg.Reasoning,
		StopReason: mapResponsesStopReason(len(msg.ToolCalls) > 0, env.IncompleteDetails),
		Usage:      usage,
	}, nil
}

// StreamResponses parses an SSE stream from the Responses API and yields react.StreamEvents.
func StreamResponses(ctx context.Context, r io.Reader) iter.Seq2[react.StreamEvent, error] {
	return StreamResponsesEvents(ctx, ReadSSE(r))
}

// StreamResponsesEvents parses a sequence of SSEEvents from the Responses API and yields react.StreamEvents.
func StreamResponsesEvents(ctx context.Context, events iter.Seq2[SSEEvent, error]) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		var toolCallSeen, finishedCleanly bool

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

			var base struct {
				Type        string `json:"type"`
				OutputIndex int    `json:"output_index"`
				Delta       string `json:"delta"`
				Item        *struct {
					ID        string `json:"id"`
					Type      string `json:"type"`
					CallID    string `json:"call_id"`
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"item"`
				Response *ResponsesResponseEnvelope `json:"response"`
			}
			if err := json.Unmarshal([]byte(sse.Data), &base); err != nil {
				if s := strings.TrimSpace(sse.Data); s != "" && !strings.HasPrefix(s, ":") {
					yield(react.StreamEvent{}, fmt.Errorf("provider stream invalid json: %w: %s", err, s))
					return
				}
				continue
			}

			eventType := sse.Event
			if eventType == "" {
				eventType = base.Type
			}

			switch eventType {
			case "error", "response.failed":
				yield(react.StreamEvent{}, fmt.Errorf("provider stream error: %s", sse.Data))
				return

			case "response.output_text.delta":
				if base.Delta != "" {
					if !yield(react.StreamEvent{Type: react.StreamTextDelta, Text: base.Delta}, nil) {
						return
					}
				}

			case "response.reasoning_text.delta", "response.reasoning.delta", "response.reasoning_summary_text.delta":
				if base.Delta != "" {
					if !yield(react.StreamEvent{Type: react.StreamReasoningDelta, Text: base.Delta}, nil) {
						return
					}
				}

			case "response.output_item.added":
				if base.Item != nil && base.Item.Type == "function_call" {
					toolCallSeen = true
					callID := base.Item.CallID
					if callID == "" {
						callID = base.Item.ID
					}
					if !yield(react.StreamEvent{
						Type:       react.StreamToolCallStart,
						Index:      base.OutputIndex,
						ToolCallID: callID,
						ToolName:   base.Item.Name,
					}, nil) {
						return
					}
				}

			case "response.function_call_arguments.delta":
				toolCallSeen = true
				if base.Delta != "" {
					if !yield(react.StreamEvent{
						Type:      react.StreamToolCallDelta,
						Index:     base.OutputIndex,
						ArgsDelta: base.Delta,
					}, nil) {
						return
					}
				}

			case "response.completed", "response.done":
				finishedCleanly = true
				if base.Response != nil {
					if base.Response.Usage != nil {
						u := react.Usage{
							InputTokens:  base.Response.Usage.InputTokens,
							OutputTokens: base.Response.Usage.OutputTokens,
						}
						if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &u}, nil) {
							return
						}
					}
					stopReason := mapResponsesStopReason(toolCallSeen, base.Response.IncompleteDetails)
					if !yield(react.StreamEvent{Type: react.StreamDone, StopReason: stopReason}, nil) {
						return
					}
				}
			}
		}

		if !finishedCleanly && ctx.Err() == nil {
			yield(react.StreamEvent{}, io.ErrUnexpectedEOF)
		}
	}
}
