// Package anthropic implements an Anthropic Messages API react.Provider.
package anthropic

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
	"github.com/gosolu/react/provider"
)

// DefaultBaseURL is Anthropic's official API base URL.
const DefaultBaseURL = "https://api.anthropic.com/v1"

// DefaultVersion is the Anthropic API version.
const DefaultVersion = "2023-06-01"

// Provider implements react.Provider for the Anthropic Messages API.
type Provider struct {
	cfg provider.Config
}

// New creates a new Anthropic provider.
func New(cfg provider.Config) *Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	return &Provider{cfg: cfg}
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "anthropic"
}

// Generate implements react.Provider (non-streaming).
func (p *Provider) Generate(ctx context.Context, req *react.Request) (*react.Response, error) {
	rc, err := p.cfg.PostJSON(
		ctx,
		p.Name(),
		p.cfg.BaseURL+"/messages",
		p.headers(),
		p.buildRequestBody(req, false),
	)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rc.Close() }()

	var msgResp messagesResponse
	if err := json.NewDecoder(rc).Decode(&msgResp); err != nil {
		return nil, fmt.Errorf("anthropic: decode response: %w", err)
	}

	msg := react.Message{Role: react.RoleAssistant}
	for _, block := range msgResp.Content {
		switch block.Type {
		case "thinking":
			msg.Reasoning = block.Thinking
			msg.ReasoningSignature = block.Signature
		case "text":
			msg.Content = append(msg.Content, react.TextPart(block.Text))
		case "tool_use":
			msg.ToolCalls = append(msg.ToolCalls, react.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: block.Input,
			})
		}
	}

	return &react.Response{
		Message:    msg,
		Reasoning:  msg.Reasoning,
		StopReason: mapStopReason(msgResp.StopReason),
		Usage: react.Usage{
			InputTokens:      msgResp.Usage.InputTokens,
			OutputTokens:     msgResp.Usage.OutputTokens,
			CacheReadTokens:  msgResp.Usage.CacheReadInputTokens,
			CacheWriteTokens: msgResp.Usage.CacheCreationInputTokens,
		},
	}, nil
}

// Stream implements react.Provider (server-sent events).
func (p *Provider) Stream(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		var currentUsage react.Usage
		var finishedCleanly bool

		events := p.cfg.StreamJSON(
			ctx,
			p.Name(),
			p.cfg.BaseURL+"/messages",
			p.headers(),
			p.buildRequestBody(req, true),
		)

		for sse, err := range events {
			if err != nil {
				if err != io.EOF && ctx.Err() == nil {
					yield(react.StreamEvent{}, err)
				}
				return
			}
			if err := provider.CheckSSEError(sse, "anthropic stream error"); err != nil {
				yield(react.StreamEvent{}, err)
				return
			}

			switch sse.Event {
			case "message_start":
				var start messageStartEvent
				if json.Unmarshal([]byte(sse.Data), &start) == nil {
					currentUsage.InputTokens = start.Message.Usage.InputTokens
					currentUsage.OutputTokens = start.Message.Usage.OutputTokens
					currentUsage.CacheReadTokens = start.Message.Usage.CacheReadInputTokens
					currentUsage.CacheWriteTokens = start.Message.Usage.CacheCreationInputTokens
					if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &currentUsage}, nil) {
						return
					}
				}
			case "content_block_start":
				var blockStart contentBlockStartEvent
				if json.Unmarshal([]byte(sse.Data), &blockStart) == nil && blockStart.ContentBlock.Type == "tool_use" {
					if !yield(react.StreamEvent{
						Type:       react.StreamToolCallStart,
						Index:      blockStart.Index,
						ToolCallID: blockStart.ContentBlock.ID,
						ToolName:   blockStart.ContentBlock.Name,
					}, nil) {
						return
					}
				}
			case "content_block_delta":
				var delta contentBlockDeltaEvent
				if json.Unmarshal([]byte(sse.Data), &delta) == nil {
					switch delta.Delta.Type {
					case "text_delta":
						if !yield(react.StreamEvent{Type: react.StreamTextDelta, Text: delta.Delta.Text}, nil) {
							return
						}
					case "thinking_delta":
						if !yield(react.StreamEvent{Type: react.StreamReasoningDelta, Text: delta.Delta.Thinking}, nil) {
							return
						}
					case "signature_delta":
						if !yield(react.StreamEvent{Type: react.StreamReasoningDelta, Signature: delta.Delta.Signature}, nil) {
							return
						}
					case "input_json_delta":
						if !yield(react.StreamEvent{Type: react.StreamToolCallDelta, Index: delta.Index, ArgsDelta: delta.Delta.PartialJSON}, nil) {
							return
						}
					}
				}
			case "message_delta":
				var msgDelta messageDeltaEvent
				if json.Unmarshal([]byte(sse.Data), &msgDelta) == nil {
					currentUsage.OutputTokens = msgDelta.Usage.OutputTokens
					if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &currentUsage}, nil) {
						return
					}
					if msgDelta.Delta.StopReason != "" {
						finishedCleanly = true
						if !yield(react.StreamEvent{
							Type:       react.StreamDone,
							StopReason: mapStopReason(msgDelta.Delta.StopReason),
						}, nil) {
							return
						}
					}
				}
			case "message_stop":
				return
			}
		}

		if !finishedCleanly && ctx.Err() == nil {
			yield(react.StreamEvent{}, io.ErrUnexpectedEOF)
		}
	}
}

func (p *Provider) headers() map[string]string {
	h := make(map[string]string, len(p.cfg.ExtraHeader)+3)
	h["Content-Type"] = "application/json"
	h["anthropic-version"] = DefaultVersion
	if p.cfg.APIKey != "" {
		h["x-api-key"] = p.cfg.APIKey
	}
	maps.Copy(h, p.cfg.ExtraHeader)
	return h
}

func (p *Provider) buildRequestBody(req *react.Request, stream bool) map[string]any {
	body := make(map[string]any)
	maps.Copy(body, req.Extra)

	body["model"] = req.Model
	body["stream"] = stream

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096 // Anthropic requires max_tokens
	}
	body["max_tokens"] = maxTokens

	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}

	var systemParts []string
	if req.System != "" {
		systemParts = append(systemParts, req.System)
	}

	var messages []map[string]any
	for _, m := range req.Messages {
		switch m.Role {
		case react.RoleSystem:
			systemParts = append(systemParts, m.Text())
		case react.RoleUser:
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": buildAnthropicContentBlocks(m.Content),
			})
		case react.RoleAssistant:
			var content []map[string]any
			if m.Reasoning != "" {
				thinkingBlock := map[string]any{
					"type":     "thinking",
					"thinking": m.Reasoning,
				}
				if m.ReasoningSignature != "" {
					thinkingBlock["signature"] = m.ReasoningSignature
				}
				content = append(content, thinkingBlock)
			}
			for _, part := range m.Content {
				if part.Type == react.PartText && part.Text != "" {
					content = append(content, map[string]any{"type": "text", "text": part.Text})
				}
			}
			for _, tc := range m.ToolCalls {
				var parsedArgs any
				if err := json.Unmarshal(tc.Arguments, &parsedArgs); err != nil {
					parsedArgs = map[string]any{}
				}
				content = append(content, map[string]any{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Name,
					"input": parsedArgs,
				})
			}
			messages = append(messages, map[string]any{
				"role":    "assistant",
				"content": content,
			})
		case react.RoleTool:
			// In Anthropic, tool results go into a "user" turn with type="tool_result"
			var resBody any = m.Text()
			if provider.HasImageParts(m.Content) {
				resBody = buildAnthropicContentBlocks(m.Content)
			}
			resContent := map[string]any{
				"type":        "tool_result",
				"tool_use_id": m.ToolCallID,
				"content":     resBody,
				"is_error":    m.IsError,
			}
			// Collapse with previous user turn if consecutive
			if n := len(messages); n > 0 && messages[n-1]["role"] == "user" {
				messages[n-1]["content"] = append(messages[n-1]["content"].([]map[string]any), resContent)
			} else {
				messages = append(messages, map[string]any{
					"role":    "user",
					"content": []map[string]any{resContent},
				})
			}
		}
	}

	if len(systemParts) > 0 && body["system"] == nil {
		body["system"] = strings.Join(systemParts, "\n\n")
	}
	body["messages"] = messages

	if len(req.Tools) > 0 {
		tools := make([]map[string]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			var schema map[string]any
			_ = json.Unmarshal(t.InputSchema, &schema)
			tools = append(tools, map[string]any{
				"name":         t.Name,
				"description":  t.Description,
				"input_schema": schema,
			})
		}
		body["tools"] = tools

		switch req.ToolChoice {
		case react.ToolChoiceAuto:
			body["tool_choice"] = map[string]string{"type": "auto"}
		case react.ToolChoiceRequired:
			body["tool_choice"] = map[string]string{"type": "any"}
		case "", react.ToolChoiceNone:
			// omit
		default:
			body["tool_choice"] = map[string]string{
				"type": "tool",
				"name": req.ToolChoice,
			}
		}
	}

	return body
}

func buildAnthropicContentBlocks(parts []react.Part) []map[string]any {
	out := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case react.PartText:
			out = append(out, map[string]any{"type": "text", "text": p.Text})
		case react.PartImage:
			out = append(out, buildAnthropicImageBlock(p))
		}
	}
	return out
}

func buildAnthropicImageBlock(p react.Part) map[string]any {
	if p.ImageURL != "" {
		return map[string]any{
			"type":   "image",
			"source": map[string]any{"type": "url", "url": p.ImageURL},
		}
	}
	mediaType := p.MIMEType
	if mediaType == "" {
		mediaType = "image/jpeg"
	}
	var b64Data string
	if len(p.Data) > 0 {
		b64Data = base64.StdEncoding.EncodeToString(p.Data)
	}
	return map[string]any{
		"type": "image",
		"source": map[string]any{
			"type":       "base64",
			"media_type": mediaType,
			"data":       b64Data,
		},
	}
}

func mapStopReason(r string) react.StopReason {
	switch r {
	case "end_turn", "stop_sequence":
		return react.StopEnd
	case "tool_use":
		return react.StopToolCall
	case "max_tokens":
		return react.StopLength
	default:
		return react.StopOther
	}
}

// Anthropic Wire format structs

type messagesResponse struct {
	ID         string         `json:"id"`
	Type       string         `json:"type"`
	Role       string         `json:"role"`
	Content    []contentBlock `json:"content"`
	StopReason string         `json:"stop_reason"`
	Usage      struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	} `json:"usage"`
}

type contentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
}

type messageStartEvent struct {
	Message messagesResponse `json:"message"`
}

type contentBlockStartEvent struct {
	Index        int          `json:"index"`
	ContentBlock contentBlock `json:"content_block"`
}

type contentBlockDeltaEvent struct {
	Index int `json:"index"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text,omitempty"`
		Thinking    string `json:"thinking,omitempty"`
		Signature   string `json:"signature,omitempty"`
		PartialJSON string `json:"partial_json,omitempty"`
	} `json:"delta"`
}

type messageDeltaEvent struct {
	Delta struct {
		StopReason string `json:"stop_reason"`
	} `json:"delta"`
	Usage struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}
