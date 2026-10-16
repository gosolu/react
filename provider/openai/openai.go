// Package openai implements an OpenAI-compatible react.Provider.
// It supports standard OpenAI endpoints as well as compatible APIs
// like vLLM, Ollama, DeepSeek, LocalAI, etc.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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
	BaseURL     string           // defaults to DefaultBaseURL
	APIType     provider.APIType // defaults to provider.APITypeChatCompletions
	HTTPClient  *http.Client     // defaults to http.DefaultClient
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

// NewResponses creates a new OpenAI provider configured to use the Responses API.
func NewResponses(cfg Config) *Provider {
	cfg.APIType = provider.APITypeResponses
	return New(cfg)
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "openai"
}

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
		return nil, fmt.Errorf("openai: marshal responses request: %w", err)
	}

	resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpointURL(), bytes.NewReader(data))
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
	bodyMap := provider.BuildChatCompletionsRequestBody(req, false)
	data, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpointURL(), bytes.NewReader(data))
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
			yield(react.StreamEvent{}, fmt.Errorf("openai: marshal responses request: %w", err))
			return
		}

		resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
			httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpointURL(), bytes.NewReader(data))
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
		bodyMap := provider.BuildChatCompletionsRequestBody(req, true)
		data, err := json.Marshal(bodyMap)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("openai: marshal request: %w", err))
			return
		}

		resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
			httpReq, err := http.NewRequestWithContext(c, http.MethodPost, p.endpointURL(), bytes.NewReader(data))
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

		for ev, err := range provider.StreamChatCompletions(ctx, resp.Body) {
			if !yield(ev, err) {
				return
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
