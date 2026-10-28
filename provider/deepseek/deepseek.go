// Package deepseek implements a DeepSeek-compatible react.Provider.
// It supports DeepSeek chat, reasoning (DeepSeek-R1 / deepseek-reasoner),
// and vision (deepseek-flash) models.
//
// Reference: https://api-docs.deepseek.com/guides/vision
package deepseek

import (
	"context"
	"iter"
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

// Provider implements react.Provider for the DeepSeek API.
type Provider struct {
	cfg provider.Config
}

// New creates a new DeepSeek provider.
func New(cfg provider.Config) *Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	return &Provider{cfg: cfg}
}

// NewResponses creates a new DeepSeek provider configured to use the Responses API.
func NewResponses(cfg provider.Config) *Provider {
	cfg.APIType = provider.APITypeResponses
	return New(cfg)
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "deepseek"
}

// Generate implements react.Provider (non-streaming).
func (p *Provider) Generate(ctx context.Context, req *react.Request) (*react.Response, error) {
	return p.cfg.GenerateOpenAICompat(ctx, p.Name(), req, p.cfg.BearerHeaders(nil))
}

// Stream implements react.Provider (server-sent events).
func (p *Provider) Stream(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return p.cfg.StreamOpenAICompat(ctx, p.Name(), req, p.cfg.BearerHeaders(nil))
}
