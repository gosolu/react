// Package deepseek implements a DeepSeek-compatible react.Provider.
// It supports DeepSeek chat, reasoning (DeepSeek-R1 / deepseek-reasoner),
// and vision (deepseek-flash) models.
//
// Reference: https://api-docs.deepseek.com/guides/vision
package deepseek

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"maps"
	"net/http"
	"strings"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider"
)

// DefaultBaseURL is DeepSeek's official API base URL.
const DefaultBaseURL = "https://api.deepseek.com"

// Common DeepSeek model identifiers.
const (
	ModelChat     = "deepseek-chat"     // DeepSeek-V3
	ModelReasoner = "deepseek-reasoner" // DeepSeek-R1 reasoning model
	ModelFlash    = "deepseek-flash"    // Multimodal vision & fast text model
)

// Config configures the DeepSeek provider.
type Config struct {
	APIKey      string
	BaseURL     string           // defaults to DefaultBaseURL
	APIType     provider.APIType // defaults to provider.APITypeChatCompletions
	HTTPClient  *http.Client     // defaults to http.DefaultClient
	RetryPolicy provider.RetryPolicy
	ExtraHeader map[string]string
}

// Provider implements react.Provider for the DeepSeek API.
type Provider struct {
	cfg Config
}

// New creates a new DeepSeek provider.
func New(cfg Config) *Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	return &Provider{cfg: cfg}
}

// NewResponses creates a new DeepSeek provider configured to use the Responses API.
func NewResponses(cfg Config) *Provider {
	cfg.APIType = provider.APITypeResponses
	return New(cfg)
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "deepseek"
}

// Generate implements react.Provider (non-streaming).
func (p *Provider) Generate(ctx context.Context, req *react.Request) (*react.Response, error) {
	if p.cfg.APIType == provider.APITypeResponses {
		return p.generateResponses(ctx, req)
	}
	return p.generateChatCompletions(ctx, req)
}

func (p *Provider) generateResponses(ctx context.Context, req *react.Request) (*react.Response, error) {
	bodyMap := provider.BuildResponsesRequestBody(req, false)
	data, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, fmt.Errorf("deepseek: marshal responses request: %w", err)
	}

	resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpoint(), bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		p.setHeaders(httpReq)
		return httpReq, nil
	})
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	return provider.ParseResponsesResponse(resp.Body)
}

func (p *Provider) generateChatCompletions(ctx context.Context, req *react.Request) (*react.Response, error) {
	bodyMap := p.buildRequestBody(req, false)
	data, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, fmt.Errorf("deepseek: marshal request: %w", err)
	}

	resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpoint(), bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		p.setHeaders(httpReq)
		return httpReq, nil
	})
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	var chatResp chatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("deepseek: decode response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("deepseek: empty choices returned")
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
		Reasoning:  choice.Message.ReasoningContent,
		StopReason: mapFinishReason(choice.FinishReason),
		Usage: react.Usage{
			InputTokens:  chatResp.Usage.PromptTokens,
			OutputTokens: chatResp.Usage.CompletionTokens,
		},
	}, nil
}

// Stream implements react.Provider (server-sent events).
func (p *Provider) Stream(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	if p.cfg.APIType == provider.APITypeResponses {
		return p.streamResponses(ctx, req)
	}
	return p.streamChatCompletions(ctx, req)
}

func (p *Provider) streamResponses(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		bodyMap := provider.BuildResponsesRequestBody(req, true)
		data, err := json.Marshal(bodyMap)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("deepseek: marshal responses request: %w", err))
			return
		}

		resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
			httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpoint(), bytes.NewReader(data))
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
		defer func() { _ = resp.Body.Close() }()

		for ev, err := range provider.StreamResponses(ctx, resp.Body) {
			if !yield(ev, err) {
				return
			}
		}
	}
}

func (p *Provider) streamChatCompletions(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		bodyMap := p.buildRequestBody(req, true)
		data, err := json.Marshal(bodyMap)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("deepseek: marshal request: %w", err))
			return
		}

		resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
			httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpoint(), bytes.NewReader(data))
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
		defer func() { _ = resp.Body.Close() }()

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
				if !seenTools[tc.Index] {
					seenTools[tc.Index] = true
					if !yield(react.StreamEvent{
						Type:       react.StreamToolCallStart,
						Index:      tc.Index,
						ToolCallID: tc.ID,
						ToolName:   tc.Function.Name,
					}, nil) {
						return
					}
				}
				if tc.Function.Arguments != "" {
					if !yield(react.StreamEvent{
						Type:      react.StreamToolCallDelta,
						Index:     tc.Index,
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

func (p *Provider) endpoint() string {
	if p.cfg.APIType == provider.APITypeResponses {
		baseURL := strings.TrimSuffix(p.cfg.BaseURL, "/chat/completions")
		if strings.HasSuffix(baseURL, "/responses") {
			return baseURL
		}
		return baseURL + "/responses"
	}
	baseURL := strings.TrimSuffix(p.cfg.BaseURL, "/responses")
	if strings.HasSuffix(baseURL, "/chat/completions") {
		return baseURL
	}
	return baseURL + "/chat/completions"
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
			item := map[string]any{
				"role":    "assistant",
				"content": m.Text(),
			}
			if len(m.ToolCalls) > 0 {
				var calls []map[string]any
				for _, tc := range m.ToolCalls {
					calls = append(calls, map[string]any{
						"id":   tc.ID,
						"type": "function",
						"function": map[string]any{
							"name":      tc.Name,
							"arguments": string(tc.Arguments),
						},
					})
				}
				item["tool_calls"] = calls
			}
			messages = append(messages, item)
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

// Wire structures for DeepSeek JSON API

type chatCompletionResponse struct {
	Choices []struct {
		Message struct {
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
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

type chatCompletionChunk struct {
	Choices []struct {
		Delta struct {
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
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}
