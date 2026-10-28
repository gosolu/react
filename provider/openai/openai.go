// Package openai implements an OpenAI-compatible react.Provider.
// It supports standard OpenAI endpoints as well as compatible APIs
// like vLLM, Ollama, DeepSeek, LocalAI, etc.
package openai

import (
	"context"
	"iter"
	"strings"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider"
)

// DefaultBaseURL is OpenAI's official API base URL.
const DefaultBaseURL = "https://api.openai.com/v1"

// Provider implements react.Provider for OpenAI and compatible APIs.
type Provider struct {
	cfg provider.Config
}

// New creates a new OpenAI provider.
func New(cfg provider.Config) *Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	return &Provider{cfg: cfg}
}

// NewResponses creates a new OpenAI provider configured to use the Responses API.
func NewResponses(cfg provider.Config) *Provider {
	cfg.APIType = provider.APITypeResponses
	return New(cfg)
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "openai"
}

// Generate implements react.Provider (non-streaming).
func (p *Provider) Generate(ctx context.Context, req *react.Request) (*react.Response, error) {
	return p.cfg.GenerateOpenAICompat(ctx, p.Name(), req, p.cfg.BearerHeaders(nil))
}

// Stream implements react.Provider (server-sent events).
func (p *Provider) Stream(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return p.cfg.StreamOpenAICompat(ctx, p.Name(), req, p.cfg.BearerHeaders(nil))
}
