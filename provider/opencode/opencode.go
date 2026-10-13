// Package opencode implements a react.Provider for OpenCode Go.
// OpenCode Go is a curated model gateway for coding agents.
// It requires specific client identification (User-Agent) and
// conversation routing session tracking (x-opencode-session).
//
// See: https://opencode.ai/v2/docs/console/go#where-can-i-use-it
package opencode

import (
	"bytes"
	"context"
	"crypto/rand"
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

// DefaultBaseURL is OpenCode Go's official API base URL.
const DefaultBaseURL = "https://opencode.ai/zen/go/v1"

// DefaultUserAgent is the client identifier sent to OpenCode Go.
// OpenCode Go requires coding clients to identify themselves with
// their own User-Agent rather than a generic SDK or HTTP library.
const DefaultUserAgent = "react-agent/1.0"

// HeaderSession is the required header for OpenCode Go session persistence
// and prompt-cache routing optimization.
const HeaderSession = "x-opencode-session"

// Popular OpenCode Go model identifiers.
const (
	ModelDeepSeekV4Pro         = "deepseek-v4-pro"
	ModelDeepSeekV41Flash      = "deepseek-v4.1-flash"
	ModelDeepSeekV4Flash       = "deepseek-v4-flash"
	ModelDeepSeekV4FlashVision = "deepseek-v4-flash-vision-exp"
	ModelKimiK3                = "kimi-k3"
	ModelKimiK27Code           = "kimi-k2.7-code"
	ModelKimiK26               = "kimi-k2.6"
	ModelGLM53Flash            = "glm-5.3-flash"
	ModelGLM53                 = "glm-5.3"
	ModelGLM52                 = "glm-5.2"
	ModelMiMoV26Flash          = "mimo-v2.6-flash"
	ModelMiMoV26Pro            = "mimo-v2.6-pro"
	ModelMiMoV25               = "mimo-v2.5"
	ModelMiMoV25Pro            = "mimo-v2.5-pro"
	ModelLongCat20             = "longcat-2.0"
	ModelLongCat25Preview      = "longcat-2.5-preview-free"
	ModelStep5Preview          = "step-5-preview-free"
	ModelHy4Preview            = "hy4-preview"
	ModelHy3                   = "hy3"
	ModelSpaceBunny            = "space-bunny"
)

// Config configures the OpenCode Go provider.
type Config struct {
	APIKey        string
	BaseURL       string           // defaults to DefaultBaseURL ("https://opencode.ai/zen/go/v1")
	APIType       provider.APIType // defaults to provider.APITypeChatCompletions
	UserAgent     string           // defaults to DefaultUserAgent ("react-agent/1.0")
	SessionID     string           // static fallback session ID for x-opencode-session
	SessionIDFunc func(ctx context.Context, req *react.Request) string
	HTTPClient    *http.Client // defaults to http.DefaultClient
	RetryPolicy   provider.RetryPolicy
	ExtraHeader   map[string]string
}

// Provider implements react.Provider for OpenCode Go.
type Provider struct {
	cfg              Config
	defaultSessionID string
}

// New creates a new OpenCode Go provider.
func New(cfg Config) *Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")

	if cfg.UserAgent == "" {
		cfg.UserAgent = DefaultUserAgent
	}

	return &Provider{
		cfg:              cfg,
		defaultSessionID: generateUUID(),
	}
}

// NewResponses creates a new OpenCode Go provider configured to use the Responses API.
func NewResponses(cfg Config) *Provider {
	cfg.APIType = provider.APITypeResponses
	return New(cfg)
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "opencode-go"
}

// endpointURL returns the chat completions or responses URL.
func (p *Provider) endpointURL() string {
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

// resolveSessionID determines the session ID to pass in the x-opencode-session header.
// Precedence: SessionIDFunc > react.RunContextFrom(ctx) > Config.SessionID > generated fallback.
func (p *Provider) resolveSessionID(ctx context.Context, req *react.Request) string {
	if p.cfg.SessionIDFunc != nil {
		if sid := p.cfg.SessionIDFunc(ctx, req); sid != "" {
			return sid
		}
	}
	if rc := react.RunContextFrom(ctx); rc != nil && rc.SessionID != "" {
		return rc.SessionID
	}
	if p.cfg.SessionID != "" {
		return p.cfg.SessionID
	}
	return p.defaultSessionID
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
		return nil, fmt.Errorf("opencode: marshal responses request: %w", err)
	}

	sessionID := p.resolveSessionID(ctx, req)
	resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpointURL(), bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		p.setHeaders(httpReq, sessionID)
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
		return nil, fmt.Errorf("opencode: marshal request: %w", err)
	}

	sessionID := p.resolveSessionID(ctx, req)
	resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpointURL(), bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		p.setHeaders(httpReq, sessionID)
		return httpReq, nil
	})
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	var chatResp chatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("opencode: decode response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("opencode: empty choices returned")
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
			yield(react.StreamEvent{}, fmt.Errorf("opencode: marshal responses request: %w", err))
			return
		}

		sessionID := p.resolveSessionID(ctx, req)
		resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
			httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpointURL(), bytes.NewReader(data))
			if err != nil {
				return nil, err
			}
			p.setHeaders(httpReq, sessionID)
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
			yield(react.StreamEvent{}, fmt.Errorf("opencode: marshal request: %w", err))
			return
		}

		sessionID := p.resolveSessionID(ctx, req)
		resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
			httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpointURL(), bytes.NewReader(data))
			if err != nil {
				return nil, err
			}
			p.setHeaders(httpReq, sessionID)
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

func (p *Provider) setHeaders(req *http.Request, sessionID string) {
	req.Header.Set("Content-Type", "application/json")
	if p.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	}

	ua := p.cfg.UserAgent
	if ua == "" {
		ua = DefaultUserAgent
	}
	req.Header.Set("User-Agent", ua)

	if sessionID != "" {
		req.Header.Set(HeaderSession, sessionID)
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
			var content any = m.Text()
			var hasImages bool
			for _, part := range m.Content {
				if part.Type == react.PartImage {
					hasImages = true
					break
				}
			}
			if hasImages {
				var parts []map[string]any
				for _, part := range m.Content {
					switch part.Type {
					case react.PartText:
						parts = append(parts, map[string]any{
							"type": "text",
							"text": part.Text,
						})
					case react.PartImage:
						url := part.ImageURL
						if url == "" && len(part.Data) > 0 {
							url = fmt.Sprintf("data:%s;base64,%s", part.MIMEType, base64.StdEncoding.EncodeToString(part.Data))
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

func generateUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// Wire structures for OpenAI-compatible JSON API.

type chatCompletionResponse struct {
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
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

type chatCompletionChunk struct {
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
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}
