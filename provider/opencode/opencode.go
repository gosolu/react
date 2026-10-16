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
	"encoding/json"
	"fmt"
	"iter"
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
	bodyMap := provider.BuildChatCompletionsRequestBody(req, false)
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

	return provider.ParseChatCompletionsResponse(resp.Body)
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
		bodyMap := provider.BuildChatCompletionsRequestBody(req, true)
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

		for ev, err := range provider.StreamChatCompletions(ctx, resp.Body) {
			if !yield(ev, err) {
				return
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

func generateUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

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

func mapFinishReason(r string) react.StopReason { return provider.MapFinishReason(r) }
