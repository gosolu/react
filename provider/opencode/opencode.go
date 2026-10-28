// Package opencode implements a react.Provider for OpenCode Go.
// OpenCode Go is a curated model gateway for coding agents.
// It requires specific client identification (User-Agent) and
// conversation routing session tracking (x-opencode-session).
//
// See: https://opencode.ai/v2/docs/console/go#where-can-i-use-it
package opencode

import (
	"context"
	"crypto/rand"
	"fmt"
	"iter"
	"strings"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider"
)

// DefaultBaseURL is OpenCode Go's official API base URL.
const DefaultBaseURL = "https://opencode.ai/zen/go/v1"

// DefaultUserAgent is the client identifier sent to OpenCode Go.
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

// Provider implements react.Provider for OpenCode Go.
type Provider struct {
	cfg              provider.Config
	defaultSessionID string
}

// New creates a new OpenCode Go provider.
func New(cfg provider.Config) *Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	return &Provider{
		cfg:              cfg,
		defaultSessionID: generateUUID(),
	}
}

// NewResponses creates a new OpenCode Go provider configured to use the Responses API.
func NewResponses(cfg provider.Config) *Provider {
	cfg.APIType = provider.APITypeResponses
	return New(cfg)
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "opencode-go"
}

// Generate implements react.Provider (non-streaming).
func (p *Provider) Generate(ctx context.Context, req *react.Request) (*react.Response, error) {
	return p.cfg.GenerateOpenAICompat(ctx, p.Name(), req, p.headers(ctx))
}

// Stream implements react.Provider (server-sent events).
func (p *Provider) Stream(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return p.cfg.StreamOpenAICompat(ctx, p.Name(), req, p.headers(ctx))
}

func (p *Provider) headers(ctx context.Context) map[string]string {
	sessionID := p.defaultSessionID
	if rc := react.RunContextFrom(ctx); rc != nil && rc.SessionID != "" {
		sessionID = rc.SessionID
	}
	return p.cfg.BearerHeaders(map[string]string{
		"User-Agent":  DefaultUserAgent,
		HeaderSession: sessionID,
	})
}

func generateUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
