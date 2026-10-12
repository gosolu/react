// Package anthropic implements an Anthropic Messages API react.Provider.
package anthropic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider"
)

// DefaultBaseURL is Anthropic's official API base URL.
const DefaultBaseURL = "https://api.anthropic.com/v1"

// DefaultVersion is the Anthropic API version.
const DefaultVersion = "2023-06-01"

// Config configures the Anthropic provider.
type Config struct {
	APIKey      string
	BaseURL     string       // defaults to DefaultBaseURL
	Version     string       // defaults to DefaultVersion
	HTTPClient  *http.Client // defaults to http.DefaultClient
	RetryPolicy provider.RetryPolicy
	ExtraHeader map[string]string
}

// Provider implements react.Provider for the Anthropic Messages API.
type Provider struct {
	cfg Config
}

// New creates a new Anthropic provider.
func New(cfg Config) *Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.Version == "" {
		cfg.Version = DefaultVersion
	}
	return &Provider{cfg: cfg}
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "anthropic"
}

// Generate implements react.Provider (non-streaming).
func (p *Provider) Generate(ctx context.Context, req *react.Request) (*react.Response, error) {
	bodyMap := p.buildRequestBody(req, false)
	data, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.cfg.BaseURL+"/messages", bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		p.setHeaders(httpReq)
		return httpReq, nil
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var msgResp messagesResponse
	if err := json.NewDecoder(resp.Body).Decode(&msgResp); err != nil {
		return nil, fmt.Errorf("anthropic: decode response: %w", err)
	}

	msg := react.Message{
		Role: react.RoleAssistant,
	}
	for _, block := range msgResp.Content {
		switch block.Type {
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
		StopReason: mapStopReason(msgResp.StopReason),
		Usage: react.Usage{
			InputTokens:  msgResp.Usage.InputTokens,
			OutputTokens: msgResp.Usage.OutputTokens,
		},
	}, nil
}

// Stream implements react.Provider (server-sent events).
func (p *Provider) Stream(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		bodyMap := p.buildRequestBody(req, true)
		data, err := json.Marshal(bodyMap)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("anthropic: marshal request: %w", err))
			return
		}

		resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
			httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.cfg.BaseURL+"/messages", bytes.NewReader(data))
			if err != nil {
				return nil, err
			}
			p.setHeaders(httpReq)
			httpReq.Header.Set("Accept", "text/event-stream")
			return httpReq, nil
		})
		if err != nil {
			yield(react.StreamEvent{}, err)
			return
		}
		defer resp.Body.Close()

		var currentUsage react.Usage

		for sse, err := range provider.ReadSSE(resp.Body) {
			if err != nil {
				if err != io.EOF && ctx.Err() == nil {
					yield(react.StreamEvent{}, err)
				}
				return
			}
			switch sse.Event {
			case "message_start":
				var start messageStartEvent
				if json.Unmarshal([]byte(sse.Data), &start) == nil {
					currentUsage.InputTokens = start.Message.Usage.InputTokens
					currentUsage.OutputTokens = start.Message.Usage.OutputTokens
					if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &currentUsage}, nil) {
						return
					}
				}
			case "content_block_start":
				var blockStart contentBlockStartEvent
				if json.Unmarshal([]byte(sse.Data), &blockStart) == nil {
					if blockStart.ContentBlock.Type == "tool_use" {
						if !yield(react.StreamEvent{
							Type:       react.StreamToolCallStart,
							Index:      blockStart.Index,
							ToolCallID: blockStart.ContentBlock.ID,
							ToolName:   blockStart.ContentBlock.Name,
						}, nil) {
							return
						}
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
	}
}

func (p *Provider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if p.cfg.APIKey != "" {
		req.Header.Set("x-api-key", p.cfg.APIKey)
	}
	req.Header.Set("anthropic-version", p.cfg.Version)
	for k, v := range p.cfg.ExtraHeader {
		req.Header.Set(k, v)
	}
}

func (p *Provider) buildRequestBody(req *react.Request, stream bool) map[string]any {
	body := make(map[string]any)
	for k, v := range req.Extra {
		body[k] = v
	}

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

	// System prompt in Anthropic is a top-level field
	systemParts := []string{}
	if req.System != "" {
		systemParts = append(systemParts, req.System)
	}

	var messages []map[string]any
	for _, m := range req.Messages {
		if m.Role == react.RoleSystem {
			systemParts = append(systemParts, m.Text())
			continue
		}

		switch m.Role {
		case react.RoleUser:
			var content []map[string]any
			for _, part := range m.Content {
				if part.Type == react.PartText {
					content = append(content, map[string]any{"type": "text", "text": part.Text})
				} else if part.Type == react.PartImage {
					mediaType := part.MIMEType
					if mediaType == "" {
						mediaType = "image/jpeg"
					}
					var b64Data string
					if len(part.Data) > 0 {
						b64Data = base64.StdEncoding.EncodeToString(part.Data)
					}
					content = append(content, map[string]any{
						"type": "image",
						"source": map[string]any{
							"type":       "base64",
							"media_type": mediaType,
							"data":       b64Data,
						},
					})
				}
			}
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": content,
			})
		case react.RoleAssistant:
			var content []map[string]any
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
			resContent := map[string]any{
				"type":        "tool_result",
				"tool_use_id": m.ToolCallID,
				"content":     m.Text(),
				"is_error":    m.IsError,
			}
			// Append or collapse with previous user turn if consecutive
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": []map[string]any{resContent},
			})
		}
	}

	if len(systemParts) > 0 {
		body["system"] = strings.Join(systemParts, "\n\n")
	}
	body["messages"] = messages

	if len(req.Tools) > 0 {
		var tools []map[string]any
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

		if req.ToolChoice != "" {
			switch req.ToolChoice {
			case react.ToolChoiceAuto:
				body["tool_choice"] = map[string]string{"type": "auto"}
			case react.ToolChoiceRequired:
				body["tool_choice"] = map[string]string{"type": "any"}
			case react.ToolChoiceNone:
				// no tool choice
			default:
				body["tool_choice"] = map[string]string{
					"type": "tool",
					"name": req.ToolChoice,
				}
			}
		}
	}

	return body
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
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type contentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
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
