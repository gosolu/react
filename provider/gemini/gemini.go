// Package gemini implements a Google Gemini react.Provider using
// the Gemini Interactions API (/v1beta/interactions).
package gemini

import (
	"context"
	"encoding/json"
	"iter"
	"maps"
	"strings"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider"
)

// DefaultBaseURL is the official Google Gemini API v1beta base URL.
const DefaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// Latest Google Gemini model identifiers.
const (
	ModelGemini4Argon      = "gemini-4-argon"
	ModelGemini4Flash      = "gemini-4-flash"
	ModelGemini38Flash     = "gemini-3.8-flash"
	ModelGeminiFlashLatest = "gemini-flash-latest"
	ModelGeminiProLatest   = "gemini-pro-latest"
	ModelGeminiFlashLite   = "gemini-flash-lite-latest"
)

// Provider implements react.Provider for Google Gemini models.
type Provider struct {
	cfg provider.Config
}

// New creates a new Gemini provider. By default, it uses the Gemini Interactions API (/v1beta/interactions).
func New(cfg provider.Config) *Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.APIType == "" || cfg.APIType == provider.APITypeChatCompletions {
		cfg.APIType = provider.APITypeInteractions
	}
	return &Provider{cfg: cfg}
}

// NewInteractions creates a new Gemini provider configured to use the
// Gemini Interactions API (/v1beta/interactions).
func NewInteractions(cfg provider.Config) *Provider {
	cfg.APIType = provider.APITypeInteractions
	return New(cfg)
}

// NewChatCompletions creates a Gemini provider. Note that the Gemini API no longer
// supports OpenAI-compatible Chat Completions, so this routes to the Interactions API.
func NewChatCompletions(cfg provider.Config) *Provider {
	cfg.APIType = provider.APITypeInteractions
	return New(cfg)
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "gemini"
}

// Generate implements react.Provider (non-streaming).
func (p *Provider) Generate(ctx context.Context, req *react.Request) (*react.Response, error) {
	return p.generateInteractions(ctx, req)
}

// Stream implements react.Provider (server-sent events).
func (p *Provider) Stream(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return p.streamInteractions(ctx, req)
}

func (p *Provider) headers() map[string]string {
	h := make(map[string]string, len(p.cfg.ExtraHeader)+2)
	h["Content-Type"] = "application/json"
	if p.cfg.APIKey != "" {
		if strings.HasPrefix(p.cfg.APIKey, "Bearer ") {
			h["Authorization"] = p.cfg.APIKey
		} else {
			h["x-goog-api-key"] = p.cfg.APIKey
		}
	}
	maps.Copy(h, p.cfg.ExtraHeader)
	return h
}

// Shared Gemini helpers

func collectSystemInstruction(req *react.Request) string {
	var parts []string
	if req.System != "" {
		parts = append(parts, req.System)
	}
	for _, m := range req.Messages {
		if m.Role == react.RoleSystem {
			parts = append(parts, m.Text())
		}
	}
	return strings.Join(parts, "\n\n")
}

func buildToolNamesByID(msgs []react.Message) map[string]string {
	names := make(map[string]string)
	for _, m := range msgs {
		if m.Role == react.RoleAssistant {
			for _, tc := range m.ToolCalls {
				if tc.ID != "" && tc.Name != "" {
					names[tc.ID] = tc.Name
				}
			}
		}
	}
	return names
}

func resolveToolName(m react.Message, namesByID map[string]string) string {
	if m.Name != "" {
		return m.Name
	}
	if m.ToolCallID != "" {
		if name := namesByID[m.ToolCallID]; name != "" {
			return name
		}
	}
	return m.ToolCallID
}

func resolveToolCallSignature(m react.Message, tc react.ToolCall, index int) string {
	if tc.ThoughtSignature != "" {
		return tc.ThoughtSignature
	}
	if index == 0 && m.Reasoning == "" && m.ReasoningSignature != "" {
		return m.ReasoningSignature
	}
	return ""
}

func parseToolArgs(raw json.RawMessage) any {
	var parsed any
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &parsed)
	}
	if parsed == nil {
		return map[string]any{}
	}
	return parsed
}

func parseDataURI(uri string) (mime, data string, ok bool) {
	if !strings.HasPrefix(uri, "data:") {
		return "", "", false
	}
	header, payload, found := strings.Cut(uri, ",")
	if !found {
		return "", "", false
	}
	mime = "image/jpeg"
	if semi := strings.Index(header, ";"); semi > 5 {
		mime = header[5:semi]
	}
	return mime, payload, true
}

func parseCleanedSchema(raw json.RawMessage) map[string]any {
	if len(raw) == 0 {
		return nil
	}
	var schema map[string]any
	if err := json.Unmarshal(raw, &schema); err != nil {
		return nil
	}
	cleanGeminiSchema(schema)
	if len(schema) == 0 {
		return nil
	}
	return schema
}

// cleanGeminiSchema removes unsupported fields (such as $schema and empty/boolean additionalProperties)
// from JSON Schema before passing to Gemini tool definitions.
func cleanGeminiSchema(m map[string]any) {
	delete(m, "$schema")
	if ap, exists := m["additionalProperties"]; exists {
		if b, ok := ap.(bool); ok && !b {
			delete(m, "additionalProperties")
		} else if mAp, ok := ap.(map[string]any); ok && len(mAp) == 0 {
			delete(m, "additionalProperties")
		}
	}
	if props, ok := m["properties"].(map[string]any); ok {
		for _, v := range props {
			if subMap, ok := v.(map[string]any); ok {
				cleanGeminiSchema(subMap)
			}
		}
	}
	if items, ok := m["items"].(map[string]any); ok {
		cleanGeminiSchema(items)
	}
}
