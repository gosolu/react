// Package openai implements an OpenAI-compatible react.Provider.
// It supports standard OpenAI endpoints as well as compatible APIs
// like vLLM, Ollama, DeepSeek, LocalAI, etc.
package openai

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

// DefaultBaseURL is OpenAI's official API base URL.
const DefaultBaseURL = "https://api.openai.com/v1"

// Config configures the OpenAI provider.
type Config struct {
	APIKey      string
	BaseURL     string       // defaults to DefaultBaseURL
	HTTPClient  *http.Client // defaults to http.DefaultClient
	RetryPolicy provider.RetryPolicy
	ExtraHeader map[string]string
}

// Provider implements react.Provider for OpenAI and compatible APIs.
type Provider struct {
	cfg Config
}

// New creates a new OpenAI provider.
func New(cfg Config) *Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	return &Provider{cfg: cfg}
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "openai"
}

// Generate implements react.Provider (non-streaming).
func (p *Provider) Generate(ctx context.Context, req *react.Request) (*react.Response, error) {
	bodyMap := p.buildRequestBody(req, false)
	data, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.cfg.BaseURL+"/chat/completions", bytes.NewReader(data))
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

	var chatResp chatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("openai: empty choices returned")
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

	return &react.Response{
		Message:    msg,
		StopReason: mapFinishReason(choice.FinishReason),
		Usage: react.Usage{
			InputTokens:  chatResp.Usage.PromptTokens,
			OutputTokens: chatResp.Usage.CompletionTokens,
		},
	}, nil
}

// Stream implements react.Provider (server-sent events).
func (p *Provider) Stream(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		bodyMap := p.buildRequestBody(req, true)
		data, err := json.Marshal(bodyMap)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("openai: marshal request: %w", err))
			return
		}

		resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
			httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.cfg.BaseURL+"/chat/completions", bytes.NewReader(data))
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

		seenTools := make(map[int]bool)

		for sse, err := range provider.ReadSSE(resp.Body) {
			if err != nil {
				if err != io.EOF && ctx.Err() == nil {
					yield(react.StreamEvent{}, err)
				}
				return
			}
			if sse.Data == "[DONE]" {
				return
			}
			var chunk chatCompletionChunk
			if err := json.Unmarshal([]byte(sse.Data), &chunk); err != nil {
				continue
			}
			if chunk.Usage != nil {
				u := react.Usage{
					InputTokens:  chunk.Usage.PromptTokens,
					OutputTokens: chunk.Usage.CompletionTokens,
				}
				if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &u}, nil) {
					return
				}
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
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
				if !yield(react.StreamEvent{
					Type:       react.StreamDone,
					StopReason: mapFinishReason(choice.FinishReason),
				}, nil) {
					return
				}
			}
		}
	}
}

func (p *Provider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if p.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	}
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
			var content any = m.Text()
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
					if p.Type == react.PartText {
						parts = append(parts, map[string]any{"type": "text", "text": p.Text})
					} else if p.Type == react.PartImage {
						url := p.ImageURL
						if url == "" && len(p.Data) > 0 {
							url = fmt.Sprintf("data:%s;base64,%s", p.MIMEType, base64.StdEncoding.EncodeToString(p.Data))
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
			messages = append(messages, map[string]any{
				"role":         "tool",
				"tool_call_id": m.ToolCallID,
				"content":      m.Text(),
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

		if req.ToolChoice != "" {
			switch req.ToolChoice {
			case react.ToolChoiceAuto, react.ToolChoiceNone, react.ToolChoiceRequired:
				body["tool_choice"] = req.ToolChoice
			default:
				body["tool_choice"] = map[string]any{
					"type": "function",
					"function": map[string]string{
						"name": req.ToolChoice,
					},
				}
			}
		}
	}

	return body
}

func mapFinishReason(r string) react.StopReason {
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

// Wire structures for OpenAI JSON API

type chatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			ToolCalls []struct {
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
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

type chatCompletionChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
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
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}
