// Package gemini implements a Google Gemini react.Provider.
// It supports the new standard Gemini Interactions API (/v1beta/interactions) by default,
// as well as the legacy native generateContent API when explicitly configured.
package gemini

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"maps"
	"net/http"
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

// Config configures the Gemini provider.
type Config struct {
	APIKey      string
	BaseURL     string           // defaults to DefaultBaseURL
	APIType     provider.APIType // defaults to Interactions API; or provider.APITypeGenerateContent
	HTTPClient  *http.Client     // defaults to http.DefaultClient
	RetryPolicy provider.RetryPolicy
	ExtraHeader map[string]string
}

// Provider implements react.Provider for Google Gemini models.
type Provider struct {
	cfg Config
}

// New creates a new Gemini provider. By default, it uses the modern Gemini Interactions API (/v1beta/interactions).
func New(cfg Config) *Provider {
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
func NewInteractions(cfg Config) *Provider {
	cfg.APIType = provider.APITypeInteractions
	return New(cfg)
}

// NewChatCompletions creates a Gemini provider. Note that the Gemini API no longer
// supports OpenAI-compatible Chat Completions, so this switches directly to the Interactions API.
func NewChatCompletions(cfg Config) *Provider {
	cfg.APIType = provider.APITypeInteractions
	return New(cfg)
}

// NewGenerateContent creates a new Gemini provider explicitly configured to use
// the legacy native generateContent API (/models/{model}:generateContent).
func NewGenerateContent(cfg Config) *Provider {
	cfg.APIType = provider.APITypeGenerateContent
	return New(cfg)
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "gemini"
}

// Generate implements react.Provider (non-streaming).
func (p *Provider) Generate(ctx context.Context, req *react.Request) (*react.Response, error) {
	if p.cfg.APIType == provider.APITypeGenerateContent {
		return p.generateNative(ctx, req)
	}
	return p.generateInteractions(ctx, req)
}

// Stream implements react.Provider (server-sent events).
func (p *Provider) Stream(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	if p.cfg.APIType == provider.APITypeGenerateContent {
		return p.streamNative(ctx, req)
	}
	return p.streamInteractions(ctx, req)
}

func (p *Provider) generateChatCompletions(ctx context.Context, req *react.Request) (*react.Response, error) {
	bodyMap := provider.BuildChatCompletionsRequestBody(req, false)
	data, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	endpoint := p.chatCompletionsURL()
	resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(c, http.MethodPost, endpoint, bytes.NewReader(data))
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

func (p *Provider) streamChatCompletions(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		bodyMap := provider.BuildChatCompletionsRequestBody(req, true)
		data, err := json.Marshal(bodyMap)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("gemini: marshal request: %w", err))
			return
		}

		endpoint := p.chatCompletionsURL()
		resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
			httpReq, err := http.NewRequestWithContext(c, http.MethodPost, endpoint, bytes.NewReader(data))
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

func (p *Provider) generateNative(ctx context.Context, req *react.Request) (*react.Response, error) {
	bodyMap, err := p.buildNativeRequestBody(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: build request: %w", err)
	}
	data, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	endpoint := p.nativeEndpointURL(req.Model, false)
	resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(c, http.MethodPost, endpoint, bytes.NewReader(data))
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

	var genResp generateContentResponse
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return nil, fmt.Errorf("gemini: decode response: %w", err)
	}

	if genResp.Error != nil && genResp.Error.Message != "" {
		return nil, fmt.Errorf("gemini api error: %s (code %d)", genResp.Error.Message, genResp.Error.Code)
	}

	if len(genResp.Candidates) == 0 {
		if genResp.PromptFeedback != nil && genResp.PromptFeedback.BlockReason != "" {
			return &react.Response{
				StopReason: react.StopContentFilter,
			}, nil
		}
		return nil, errors.New("gemini: empty candidates in response")
	}

	cand := genResp.Candidates[0]
	msg := react.Message{
		Role: react.RoleAssistant,
	}

	for _, part := range cand.Content.Parts {
		if part.Thought {
			msg.Reasoning += part.Text
			if part.ThoughtSignature != "" {
				msg.ReasoningSignature = part.ThoughtSignature
			}
			continue
		}
		if part.FunctionCall != nil {
			args := part.FunctionCall.Args
			if len(args) == 0 {
				args = json.RawMessage("{}")
			}
			msg.ToolCalls = append(msg.ToolCalls, react.ToolCall{
				ID:               part.FunctionCall.ID,
				Name:             part.FunctionCall.Name,
				Arguments:        args,
				ThoughtSignature: part.ThoughtSignature,
			})
			continue
		}
		if part.Text != "" {
			msg.Content = append(msg.Content, react.TextPart(part.Text))
			if part.ThoughtSignature != "" && msg.ReasoningSignature == "" {
				msg.ReasoningSignature = part.ThoughtSignature
			}
		}
	}

	// CRITICAL GEMINI DIFFERENCE:
	// Gemini sets finishReason = "STOP" even when emitting functionCall parts!
	// If any tool calls were returned, the StopReason must be react.StopToolCall.
	stopReason := mapGeminiFinishReason(cand.FinishReason)
	if len(msg.ToolCalls) > 0 {
		stopReason = react.StopToolCall
	}

	var usage react.Usage
	if genResp.UsageMetadata != nil {
		usage.InputTokens = genResp.UsageMetadata.PromptTokenCount
		usage.OutputTokens = genResp.UsageMetadata.CandidatesTokenCount + genResp.UsageMetadata.ThoughtsTokenCount
		usage.CacheReadTokens = genResp.UsageMetadata.CachedContentTokenCount
	}

	return &react.Response{
		Message:    msg,
		Reasoning:  msg.Reasoning,
		StopReason: stopReason,
		Usage:      usage,
	}, nil
}

func (p *Provider) streamNative(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		bodyMap, err := p.buildNativeRequestBody(req)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("gemini: build request: %w", err))
			return
		}
		data, err := json.Marshal(bodyMap)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("gemini: marshal request: %w", err))
			return
		}

		endpoint := p.nativeEndpointURL(req.Model, true)
		resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
			httpReq, err := http.NewRequestWithContext(c, http.MethodPost, endpoint, bytes.NewReader(data))
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

		var toolCallSeen bool
		var toolCallIndex int
		var finishedCleanly bool

		for sse, sseErr := range provider.ReadSSE(resp.Body) {
			if sseErr != nil {
				if sseErr != io.EOF && ctx.Err() == nil {
					yield(react.StreamEvent{}, sseErr)
				}
				return
			}
			if sse.Data == "[DONE]" {
				return
			}

			// Mid-stream error check
			if sse.Event == "error" || strings.Contains(sse.Data, `"error"`) {
				var errPayload struct {
					Error *struct {
						Message string `json:"message"`
						Code    int    `json:"code"`
						Status  string `json:"status"`
					} `json:"error"`
				}
				if json.Unmarshal([]byte(sse.Data), &errPayload) == nil && errPayload.Error != nil && errPayload.Error.Message != "" {
					yield(react.StreamEvent{}, fmt.Errorf("gemini stream error: %s (code %d)", errPayload.Error.Message, errPayload.Error.Code))
					return
				}
				if sse.Event == "error" {
					yield(react.StreamEvent{}, fmt.Errorf("gemini stream error: %s", sse.Data))
					return
				}
			}

			var chunk generateContentResponse
			if err := json.Unmarshal([]byte(sse.Data), &chunk); err != nil {
				if s := strings.TrimSpace(sse.Data); s != "" && !strings.HasPrefix(s, ":") {
					yield(react.StreamEvent{}, fmt.Errorf("gemini stream invalid json: %w: %s", err, s))
					return
				}
				continue
			}

			if chunk.Error != nil && chunk.Error.Message != "" {
				yield(react.StreamEvent{}, fmt.Errorf("gemini stream error: %s (code %d)", chunk.Error.Message, chunk.Error.Code))
				return
			}

			if chunk.UsageMetadata != nil {
				u := react.Usage{
					InputTokens:     chunk.UsageMetadata.PromptTokenCount,
					OutputTokens:    chunk.UsageMetadata.CandidatesTokenCount + chunk.UsageMetadata.ThoughtsTokenCount,
					CacheReadTokens: chunk.UsageMetadata.CachedContentTokenCount,
				}
				if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &u}, nil) {
					return
				}
			}

			if len(chunk.Candidates) == 0 {
				continue
			}

			cand := chunk.Candidates[0]
			for _, part := range cand.Content.Parts {
				if part.Thought {
					if !yield(react.StreamEvent{
						Type:      react.StreamReasoningDelta,
						Text:      part.Text,
						Signature: part.ThoughtSignature,
					}, nil) {
						return
					}
					continue
				}
				if part.FunctionCall != nil {
					toolCallSeen = true
					idx := toolCallIndex
					toolCallIndex++
					args := part.FunctionCall.Args
					if len(args) == 0 {
						args = json.RawMessage("{}")
					}
					if !yield(react.StreamEvent{
						Type:       react.StreamToolCallStart,
						Index:      idx,
						ToolCallID: part.FunctionCall.ID,
						ToolName:   part.FunctionCall.Name,
						Signature:  part.ThoughtSignature,
					}, nil) {
						return
					}
					if !yield(react.StreamEvent{
						Type:      react.StreamToolCallDelta,
						Index:     idx,
						Signature: part.ThoughtSignature,
						ArgsDelta: string(args),
					}, nil) {
						return
					}
					continue
				}
				if part.Text != "" {
					if !yield(react.StreamEvent{
						Type: react.StreamTextDelta,
						Text: part.Text,
					}, nil) {
						return
					}
					if part.ThoughtSignature != "" {
						if !yield(react.StreamEvent{
							Type:      react.StreamReasoningDelta,
							Signature: part.ThoughtSignature,
						}, nil) {
							return
						}
					}
				}
			}

			if cand.FinishReason != "" {
				finishedCleanly = true
				stopReason := mapGeminiFinishReason(cand.FinishReason)
				if toolCallSeen {
					stopReason = react.StopToolCall
				}
				if !yield(react.StreamEvent{
					Type:       react.StreamDone,
					StopReason: stopReason,
				}, nil) {
					return
				}
			}
		}

		if !finishedCleanly && ctx.Err() == nil {
			yield(react.StreamEvent{}, io.ErrUnexpectedEOF)
		}
	}
}

func (p *Provider) nativeEndpointURL(model string, stream bool) string {
	baseURL := p.cfg.BaseURL
	if model == "" {
		model = ModelGemini38Flash
	}
	model = strings.TrimPrefix(model, "models/")
	action := "generateContent"
	if stream {
		action = "streamGenerateContent?alt=sse"
	}
	return fmt.Sprintf("%s/models/%s:%s", baseURL, model, action)
}

func (p *Provider) chatCompletionsURL() string {
	baseURL := strings.TrimSuffix(p.cfg.BaseURL, "/chat/completions")
	baseURL = strings.TrimSuffix(baseURL, "/openai")
	return baseURL + "/openai/chat/completions"
}

func (p *Provider) interactionsEndpointURL() string {
	baseURL := p.cfg.BaseURL
	baseURL = strings.TrimSuffix(baseURL, "/chat/completions")
	baseURL = strings.TrimSuffix(baseURL, "/openai")
	baseURL = strings.TrimSuffix(baseURL, "/interactions")
	return baseURL + "/interactions"
}

func (p *Provider) generateInteractions(ctx context.Context, req *react.Request) (*react.Response, error) {
	bodyMap, err := p.buildInteractionsRequestBody(req, false)
	if err != nil {
		return nil, fmt.Errorf("gemini: build interactions request: %w", err)
	}
	data, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	endpoint := p.interactionsEndpointURL()
	resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(c, http.MethodPost, endpoint, bytes.NewReader(data))
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

	var wire interactionResponseWire
	if err := json.NewDecoder(resp.Body).Decode(&wire); err != nil {
		return nil, fmt.Errorf("gemini: decode interaction response: %w", err)
	}

	if wire.Error != nil && wire.Error.Message != "" {
		return nil, fmt.Errorf("gemini api error: %s (code %v)", wire.Error.Message, wire.Error.Code)
	}
	if len(wire.Errors) > 0 && wire.Errors[0].Message != "" {
		return nil, fmt.Errorf("gemini api error: %s (code %s)", wire.Errors[0].Message, wire.Errors[0].Code)
	}

	msg := react.Message{
		Role: react.RoleAssistant,
	}

	for _, step := range wire.Steps {
		switch step.Type {
		case "thought":
			msg.Reasoning += step.Text
			if step.Signature != "" {
				msg.ReasoningSignature = step.Signature
			}
		case "function_call":
			args := step.Arguments
			if len(args) == 0 {
				args = json.RawMessage("{}")
			}
			msg.ToolCalls = append(msg.ToolCalls, react.ToolCall{
				ID:               step.ID,
				Name:             step.Name,
				Arguments:        args,
				ThoughtSignature: step.Signature,
			})
		case "model_output":
			for _, c := range step.Content {
				switch c.Type {
				case "text":
					if c.Text != "" {
						msg.Content = append(msg.Content, react.TextPart(c.Text))
					}
				case "image":
					if c.Data != "" {
						decoded, err := base64.StdEncoding.DecodeString(c.Data)
						if err == nil {
							msg.Content = append(msg.Content, react.ImageDataPart(c.MIMEType, decoded))
						}
					} else if c.URI != "" {
						msg.Content = append(msg.Content, react.ImageURLPart(c.URI))
					}
				case "thought":
					msg.Reasoning += c.Text
				}
			}
		default:
			if step.Text != "" && step.Type != "user_input" && step.Type != "function_result" {
				msg.Content = append(msg.Content, react.TextPart(step.Text))
			}
		}
	}

	stopReason := react.StopEnd
	if len(msg.ToolCalls) > 0 || wire.Status == "requires_action" {
		stopReason = react.StopToolCall
	} else {
		switch wire.Status {
		case "completed":
			stopReason = react.StopEnd
		case "incomplete":
			stopReason = react.StopLength
		case "failed", "cancelled":
			stopReason = react.StopContentFilter
		default:
			stopReason = react.StopEnd
		}
	}

	var usage react.Usage
	if wire.Usage != nil {
		usage.InputTokens = wire.Usage.TotalInputTokens
		usage.OutputTokens = wire.Usage.TotalOutputTokens + wire.Usage.TotalThoughtTokens
		usage.CacheReadTokens = wire.Usage.TotalCachedTokens
	}

	return &react.Response{
		Message:    msg,
		Reasoning:  msg.Reasoning,
		StopReason: stopReason,
		Usage:      usage,
	}, nil
}

func (p *Provider) streamInteractions(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		bodyMap, err := p.buildInteractionsRequestBody(req, true)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("gemini: build interactions request: %w", err))
			return
		}
		data, err := json.Marshal(bodyMap)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("gemini: marshal request: %w", err))
			return
		}

		endpoint := p.interactionsEndpointURL()
		resp, err := p.cfg.RetryPolicy.Do(ctx, p.cfg.HTTPClient, p.Name(), func(c context.Context) (*http.Request, error) {
			httpReq, err := http.NewRequestWithContext(c, http.MethodPost, endpoint, bytes.NewReader(data))
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

		var toolCallSeen bool
		var toolCallIndex int
		var finishedCleanly bool
		var textSeen bool

		for sse, sseErr := range provider.ReadSSE(resp.Body) {
			if sseErr != nil {
				if sseErr != io.EOF && ctx.Err() == nil {
					yield(react.StreamEvent{}, sseErr)
				}
				return
			}
			if sse.Data == "[DONE]" {
				return
			}

			// Mid-stream error check
			if sse.Event == "error" || strings.Contains(sse.Data, `"error"`) {
				var errPayload struct {
					Error *struct {
						Message string `json:"message"`
						Code    any    `json:"code"`
						Status  string `json:"status"`
					} `json:"error"`
					Errors []struct {
						Code    string `json:"code"`
						Message string `json:"message"`
					} `json:"errors"`
				}
				if json.Unmarshal([]byte(sse.Data), &errPayload) == nil {
					if errPayload.Error != nil && errPayload.Error.Message != "" {
						yield(react.StreamEvent{}, fmt.Errorf("gemini stream error: %s (code %v)", errPayload.Error.Message, errPayload.Error.Code))
						return
					}
					if len(errPayload.Errors) > 0 && errPayload.Errors[0].Message != "" {
						yield(react.StreamEvent{}, fmt.Errorf("gemini stream error: %s (code %s)", errPayload.Errors[0].Message, errPayload.Errors[0].Code))
						return
					}
				}
				if sse.Event == "error" {
					yield(react.StreamEvent{}, fmt.Errorf("gemini stream error: %s", sse.Data))
					return
				}
			}

			var eventWire interactionStreamEventWire
			if err := json.Unmarshal([]byte(sse.Data), &eventWire); err != nil {
				if s := strings.TrimSpace(sse.Data); s != "" && !strings.HasPrefix(s, ":") {
					yield(react.StreamEvent{}, fmt.Errorf("gemini stream invalid json: %w: %s", err, s))
					return
				}
				continue
			}

			eventType := sse.Event
			if eventType == "" || eventType == "message" {
				eventType = eventWire.Type
				if eventType == "" {
					eventType = eventWire.EventType
				}
			}

			switch eventType {
			case "step.start":
				if eventWire.Step != nil && eventWire.Step.Type == "function_call" {
					toolCallSeen = true
					idx := toolCallIndex
					toolCallIndex++
					if !yield(react.StreamEvent{
						Type:       react.StreamToolCallStart,
						Index:      idx,
						ToolCallID: eventWire.Step.ID,
						ToolName:   eventWire.Step.Name,
						Signature:  eventWire.Step.Signature,
					}, nil) {
						return
					}
				}

			case "step.delta":
				if eventWire.Delta != nil {
					d := eventWire.Delta
					switch d.Type {
					case "text":
						if d.Text != "" {
							textSeen = true
							if !yield(react.StreamEvent{
								Type: react.StreamTextDelta,
								Text: d.Text,
							}, nil) {
								return
							}
						}
					case "thought":
						if d.Text != "" || d.Signature != "" {
							if !yield(react.StreamEvent{
								Type:      react.StreamReasoningDelta,
								Text:      d.Text,
								Signature: d.Signature,
							}, nil) {
								return
							}
						}
					case "thought_signature":
						if d.Signature != "" {
							if !yield(react.StreamEvent{
								Type:      react.StreamReasoningDelta,
								Signature: d.Signature,
							}, nil) {
								return
							}
						}
					case "arguments", "arguments_delta":
						argsDelta := d.Arguments
						if argsDelta == "" {
							argsDelta = d.PartialArguments
						}
						idx := toolCallIndex - 1
						if idx < 0 {
							idx = 0
						}
						if !yield(react.StreamEvent{
							Type:      react.StreamToolCallDelta,
							Index:     idx,
							ArgsDelta: argsDelta,
							Signature: d.Signature,
						}, nil) {
							return
						}
					}
				}

				if eventWire.Metadata != nil && eventWire.Metadata.TotalUsage != nil {
					u := react.Usage{
						InputTokens:     eventWire.Metadata.TotalUsage.TotalInputTokens,
						OutputTokens:    eventWire.Metadata.TotalUsage.TotalOutputTokens + eventWire.Metadata.TotalUsage.TotalThoughtTokens,
						CacheReadTokens: eventWire.Metadata.TotalUsage.TotalCachedTokens,
					}
					if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &u}, nil) {
						return
					}
				}

			case "interaction.status_update":
				if eventWire.Usage != nil {
					u := react.Usage{
						InputTokens:     eventWire.Usage.TotalInputTokens,
						OutputTokens:    eventWire.Usage.TotalOutputTokens + eventWire.Usage.TotalThoughtTokens,
						CacheReadTokens: eventWire.Usage.TotalCachedTokens,
					}
					if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &u}, nil) {
						return
					}
				}
				if eventWire.Status == "requires_action" {
					finishedCleanly = true
					if !yield(react.StreamEvent{
						Type:       react.StreamDone,
						StopReason: react.StopToolCall,
					}, nil) {
						return
					}
				} else if eventWire.Status == "completed" {
					finishedCleanly = true
					stopReason := react.StopEnd
					if toolCallSeen {
						stopReason = react.StopToolCall
					}
					if !yield(react.StreamEvent{
						Type:       react.StreamDone,
						StopReason: stopReason,
					}, nil) {
						return
					}
				}

			case "interaction.requires_action":
				finishedCleanly = true
				if !yield(react.StreamEvent{
					Type:       react.StreamDone,
					StopReason: react.StopToolCall,
				}, nil) {
					return
				}

			case "interaction.completed":
				finishedCleanly = true
				if eventWire.Interaction != nil && eventWire.Interaction.Usage != nil {
					u := react.Usage{
						InputTokens:     eventWire.Interaction.Usage.TotalInputTokens,
						OutputTokens:    eventWire.Interaction.Usage.TotalOutputTokens + eventWire.Interaction.Usage.TotalThoughtTokens,
						CacheReadTokens: eventWire.Interaction.Usage.TotalCachedTokens,
					}
					if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &u}, nil) {
						return
					}
				}
				stopReason := react.StopEnd
				if toolCallSeen {
					stopReason = react.StopToolCall
				}
				if !yield(react.StreamEvent{
					Type:       react.StreamDone,
					StopReason: stopReason,
				}, nil) {
					return
				}
			}
		}

		if !finishedCleanly && ctx.Err() == nil {
			if toolCallSeen {
				yield(react.StreamEvent{
					Type:       react.StreamDone,
					StopReason: react.StopToolCall,
				}, nil)
			} else if textSeen {
				yield(react.StreamEvent{
					Type:       react.StreamDone,
					StopReason: react.StopEnd,
				}, nil)
			} else {
				yield(react.StreamEvent{}, io.ErrUnexpectedEOF)
			}
		}
	}
}

func (p *Provider) buildInteractionsRequestBody(req *react.Request, stream bool) (map[string]any, error) {
	body := make(map[string]any)
	maps.Copy(body, req.Extra)

	// Model
	model := req.Model
	if model == "" {
		model = ModelGeminiFlashLatest
	}
	model = strings.TrimPrefix(model, "models/")
	body["model"] = model

	// Agent override: if agent is explicitly provided in Extra and model wasn't set in Request
	if agent, ok := body["agent"].(string); ok && agent != "" && req.Model == "" {
		delete(body, "model")
	}

	if stream {
		body["stream"] = true
	}

	// 1. System instructions
	var systemParts []string
	if req.System != "" {
		systemParts = append(systemParts, req.System)
	}
	for _, m := range req.Messages {
		if m.Role == react.RoleSystem {
			systemParts = append(systemParts, m.Text())
		}
	}
	if len(systemParts) > 0 {
		body["system_instruction"] = strings.Join(systemParts, "\n\n")
	}

	// 2. Generation config
	genConfig := make(map[string]any)
	if existing, ok := body["generation_config"].(map[string]any); ok {
		maps.Copy(genConfig, existing)
	}
	if req.MaxTokens > 0 {
		genConfig["max_output_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		genConfig["temperature"] = *req.Temperature
	}
	if req.ToolChoice != "" {
		switch req.ToolChoice {
		case react.ToolChoiceAuto:
			genConfig["tool_choice"] = "auto"
		case react.ToolChoiceNone:
			genConfig["tool_choice"] = "none"
		case react.ToolChoiceRequired:
			genConfig["tool_choice"] = "any"
		default:
			genConfig["tool_choice"] = map[string]any{
				"allowed_tools": map[string]any{
					"mode":  "any",
					"tools": []string{req.ToolChoice},
				},
			}
		}
	}
	if len(genConfig) > 0 {
		body["generation_config"] = genConfig
	}

	// 3. Map tool call IDs to tool names
	toolNamesByID := make(map[string]string)
	for _, m := range req.Messages {
		if m.Role == react.RoleAssistant {
			for _, tc := range m.ToolCalls {
				if tc.ID != "" && tc.Name != "" {
					toolNamesByID[tc.ID] = tc.Name
				}
			}
		}
	}

	// 4. Build input
	var nonSystem []react.Message
	for _, m := range req.Messages {
		if m.Role != react.RoleSystem {
			nonSystem = append(nonSystem, m)
		}
	}

	_, hasPrevID := body["previous_interaction_id"]

	if !hasPrevID && len(nonSystem) == 1 && nonSystem[0].Role == react.RoleUser {
		userMsg := nonSystem[0]
		content := buildInteractionsContent(userMsg)
		if len(content) == 1 && content[0]["type"] == "text" {
			body["input"] = content[0]["text"]
		} else if len(content) > 0 {
			body["input"] = content
		} else {
			body["input"] = userMsg.Text()
		}
	} else {
		var targetMsgs []react.Message
		if hasPrevID && len(nonSystem) > 0 {
			targetMsgs = extractLatestTurn(nonSystem)
		} else {
			targetMsgs = nonSystem
		}

		var steps []map[string]any
		for _, m := range targetMsgs {
			switch m.Role {
			case react.RoleUser:
				content := buildInteractionsContent(m)
				steps = append(steps, map[string]any{
					"type":    "user_input",
					"content": content,
				})

			case react.RoleAssistant:
				content := buildInteractionsContent(m)
				if len(content) > 0 {
					steps = append(steps, map[string]any{
						"type":    "model_output",
						"content": content,
					})
				}
				for _, tc := range m.ToolCalls {
					var parsedArgs any
					if len(tc.Arguments) > 0 {
						_ = json.Unmarshal(tc.Arguments, &parsedArgs)
					}
					if parsedArgs == nil {
						parsedArgs = map[string]any{}
					}
					fcStep := map[string]any{
						"type":      "function_call",
						"id":        tc.ID,
						"name":      tc.Name,
						"arguments": parsedArgs,
					}
					if tc.ThoughtSignature != "" {
						fcStep["signature"] = tc.ThoughtSignature
					}
					steps = append(steps, fcStep)
				}

			case react.RoleTool:
				toolName := m.Name
				if toolName == "" && m.ToolCallID != "" {
					toolName = toolNamesByID[m.ToolCallID]
				}
				if toolName == "" {
					toolName = m.ToolCallID
				}
				resContent := buildInteractionsContent(m)
				if len(resContent) == 0 {
					resContent = []map[string]any{{"type": "text", "text": m.Text()}}
				}
				frStep := map[string]any{
					"type":    "function_result",
					"call_id": m.ToolCallID,
					"name":    toolName,
					"result":  resContent,
				}
				if m.IsError {
					frStep["is_error"] = true
				}
				steps = append(steps, frStep)
			}
		}
		body["input"] = steps
	}

	// 5. Tools
	var tools []map[string]any
	if existingTools, ok := body["tools"].([]any); ok {
		for _, et := range existingTools {
			if m, ok := et.(map[string]any); ok {
				tools = append(tools, m)
			}
		}
	} else if existingTools, ok := body["tools"].([]map[string]any); ok {
		tools = append(tools, existingTools...)
	}

	for _, t := range req.Tools {
		toolDef := map[string]any{
			"type":        "function",
			"name":        t.Name,
			"description": t.Description,
		}
		if len(t.InputSchema) > 0 {
			var schema map[string]any
			if err := json.Unmarshal(t.InputSchema, &schema); err == nil {
				cleanGeminiSchema(schema)
				if len(schema) > 0 {
					toolDef["parameters"] = schema
				}
			}
		}
		tools = append(tools, toolDef)
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}

	return body, nil
}

func extractLatestTurn(msgs []react.Message) []react.Message {
	if len(msgs) == 0 {
		return nil
	}
	lastRole := msgs[len(msgs)-1].Role
	if lastRole == react.RoleTool {
		start := len(msgs) - 1
		for start > 0 && msgs[start-1].Role == react.RoleTool {
			start--
		}
		return msgs[start:]
	}
	return msgs[len(msgs)-1:]
}

func buildInteractionsContent(m react.Message) []map[string]any {
	var content []map[string]any
	for _, p := range m.Content {
		switch p.Type {
		case react.PartText:
			if p.Text != "" {
				content = append(content, map[string]any{
					"type": "text",
					"text": p.Text,
				})
			}
		case react.PartImage:
			if len(p.Data) > 0 {
				mime := p.MIMEType
				if mime == "" {
					mime = "image/jpeg"
				}
				content = append(content, map[string]any{
					"type":      "image",
					"mime_type": mime,
					"data":      base64.StdEncoding.EncodeToString(p.Data),
				})
			} else if p.ImageURL != "" {
				if strings.HasPrefix(p.ImageURL, "data:") {
					parts := strings.SplitN(p.ImageURL, ",", 2)
					if len(parts) == 2 {
						header := parts[0]
						data := parts[1]
						mime := "image/jpeg"
						if semi := strings.Index(header, ";"); semi > 5 {
							mime = header[5:semi]
						}
						content = append(content, map[string]any{
							"type":      "image",
							"mime_type": mime,
							"data":      data,
						})
						continue
					}
				}
				imgItem := map[string]any{
					"type": "image",
					"uri":  p.ImageURL,
				}
				if p.MIMEType != "" {
					imgItem["mime_type"] = p.MIMEType
				}
				content = append(content, imgItem)
			}
		}
	}
	if len(content) == 0 && m.Text() != "" {
		content = append(content, map[string]any{
			"type": "text",
			"text": m.Text(),
		})
	}
	return content
}

// Interactions Wire format structs

type interactionResponseWire struct {
	ID     string                `json:"id"`
	Model  string                `json:"model"`
	Status string                `json:"status"`
	Steps  []interactionStepWire `json:"steps"`
	Usage  *interactionUsageWire `json:"usage"`
	Error  *struct {
		Code    any    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
	Errors []struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"errors"`
}

type interactionStepWire struct {
	Type      string                   `json:"type"`
	ID        string                   `json:"id,omitempty"`
	Name      string                   `json:"name,omitempty"`
	Arguments json.RawMessage          `json:"arguments,omitempty"`
	Content   []interactionContentWire `json:"content,omitempty"`
	Text      string                   `json:"text,omitempty"`
	Signature string                   `json:"signature,omitempty"`
}

type interactionContentWire struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	MIMEType string `json:"mime_type,omitempty"`
	Data     string `json:"data,omitempty"`
	URI      string `json:"uri,omitempty"`
}

type interactionUsageWire struct {
	TotalInputTokens   int `json:"total_input_tokens"`
	TotalOutputTokens  int `json:"total_output_tokens"`
	TotalThoughtTokens int `json:"total_thought_tokens"`
	TotalCachedTokens  int `json:"total_cached_tokens"`
	TotalTokens        int `json:"total_tokens"`
	TotalToolUseTokens int `json:"total_tool_use_tokens"`
}

type interactionStreamEventWire struct {
	Type        string                `json:"type"`
	EventType   string                `json:"event_type"`
	Index       int                   `json:"index"`
	Status      string                `json:"status,omitempty"`
	Step        *interactionStepWire  `json:"step,omitempty"`
	Delta       *interactionDeltaWire `json:"delta,omitempty"`
	Interaction *struct {
		ID     string                `json:"id"`
		Status string                `json:"status"`
		Usage  *interactionUsageWire `json:"usage"`
	} `json:"interaction,omitempty"`
	Usage    *interactionUsageWire `json:"usage,omitempty"`
	Metadata *struct {
		TotalUsage *interactionUsageWire `json:"total_usage"`
		StepUsage  *interactionUsageWire `json:"step_usage"`
	} `json:"metadata,omitempty"`
}

type interactionDeltaWire struct {
	Type             string `json:"type"`
	Text             string `json:"text,omitempty"`
	Arguments        string `json:"arguments,omitempty"`
	PartialArguments string `json:"partial_arguments,omitempty"`
	Signature        string `json:"signature,omitempty"`
}

func (p *Provider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if p.cfg.APIKey != "" {
		if strings.HasPrefix(p.cfg.APIKey, "Bearer ") {
			req.Header.Set("Authorization", p.cfg.APIKey)
		} else {
			req.Header.Set("x-goog-api-key", p.cfg.APIKey)
		}
	}
	for k, v := range p.cfg.ExtraHeader {
		req.Header.Set(k, v)
	}
}

func (p *Provider) buildNativeRequestBody(req *react.Request) (map[string]any, error) {
	body := make(map[string]any)
	maps.Copy(body, req.Extra)

	// 1. System instructions
	var systemParts []string
	if req.System != "" {
		systemParts = append(systemParts, req.System)
	}
	for _, m := range req.Messages {
		if m.Role == react.RoleSystem {
			systemParts = append(systemParts, m.Text())
		}
	}
	if len(systemParts) > 0 {
		body["systemInstruction"] = map[string]any{
			"parts": []map[string]any{
				{"text": strings.Join(systemParts, "\n\n")},
			},
		}
	}

	// 2. Generation config
	genConfig := make(map[string]any)
	if existing, ok := body["generationConfig"].(map[string]any); ok {
		maps.Copy(genConfig, existing)
	}
	if req.MaxTokens > 0 {
		genConfig["maxOutputTokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		genConfig["temperature"] = *req.Temperature
	}
	if len(genConfig) > 0 {
		body["generationConfig"] = genConfig
	}

	// 3. Map tool call IDs to tool names so RoleTool messages without Name can be resolved
	toolNamesByID := make(map[string]string)
	for _, m := range req.Messages {
		if m.Role == react.RoleAssistant {
			for _, tc := range m.ToolCalls {
				if tc.ID != "" && tc.Name != "" {
					toolNamesByID[tc.ID] = tc.Name
				}
			}
		}
	}

	// 4. Build contents array with turn collapsing for tool results
	var contents []map[string]any

	for _, m := range req.Messages {
		if m.Role == react.RoleSystem {
			continue // handled in systemInstruction
		}

		switch m.Role {
		case react.RoleUser:
			contents = append(contents, map[string]any{
				"role":  "user",
				"parts": buildGeminiUserParts(m),
			})

		case react.RoleAssistant:
			var parts []map[string]any
			if m.Reasoning != "" {
				thoughtPart := map[string]any{
					"thought": true,
					"text":    m.Reasoning,
				}
				if m.ReasoningSignature != "" {
					thoughtPart["thoughtSignature"] = m.ReasoningSignature
				}
				parts = append(parts, thoughtPart)
			}
			for _, p := range m.Content {
				if p.Type == react.PartText && p.Text != "" {
					parts = append(parts, map[string]any{"text": p.Text})
				}
			}
			for i, tc := range m.ToolCalls {
				var parsedArgs any
				if len(tc.Arguments) > 0 {
					if err := json.Unmarshal(tc.Arguments, &parsedArgs); err != nil || parsedArgs == nil {
						parsedArgs = map[string]any{}
					}
				} else {
					parsedArgs = map[string]any{}
				}
				fc := map[string]any{
					"name": tc.Name,
					"args": parsedArgs,
				}
				if tc.ID != "" {
					fc["id"] = tc.ID
				}
				part := map[string]any{
					"functionCall": fc,
				}
				// CRITICAL GEMINI DIFFERENCE:
				// When echoing model tool calls back to Gemini, thoughtSignature must be preserved
				// on the functionCall part.
				sig := tc.ThoughtSignature
				if sig == "" && i == 0 && m.Reasoning == "" && m.ReasoningSignature != "" {
					sig = m.ReasoningSignature
				}
				if sig != "" {
					part["thoughtSignature"] = sig
				}
				parts = append(parts, part)
			}
			contents = append(contents, map[string]any{
				"role":  "model",
				"parts": parts,
			})

		case react.RoleTool:
			// Resolve tool function name
			toolName := m.Name
			if toolName == "" && m.ToolCallID != "" {
				toolName = toolNamesByID[m.ToolCallID]
			}
			if toolName == "" {
				toolName = m.ToolCallID
			}

			// Format response payload: Gemini requires response to be a JSON object (Struct)
			var respObj any
			if m.IsError {
				respObj = map[string]any{"error": m.Text()}
			} else {
				var parsedMap map[string]any
				text := strings.TrimSpace(m.Text())
				if strings.HasPrefix(text, "{") && json.Unmarshal([]byte(text), &parsedMap) == nil && parsedMap != nil {
					respObj = parsedMap
				} else {
					respObj = map[string]any{"result": m.Text()}
				}
			}

			fr := map[string]any{
				"name":     toolName,
				"response": respObj,
			}
			if m.ToolCallID != "" {
				fr["id"] = m.ToolCallID
			}

			toolPart := map[string]any{"functionResponse": fr}
			var toolParts []map[string]any
			toolParts = append(toolParts, toolPart)

			// If the tool result contains multimodal images, include them
			for _, p := range m.Content {
				if p.Type == react.PartImage {
					if imgPart := buildGeminiImagePart(p); imgPart != nil {
						toolParts = append(toolParts, imgPart)
					}
				}
			}

			// CRITICAL GEMINI DIFFERENCE:
			// Gemini requires parallel tool responses following a model tool-call turn
			// to be grouped in a single turn with matching part count and order!
			if len(contents) > 0 && contents[len(contents)-1]["role"] == "user" {
				prevParts, ok := contents[len(contents)-1]["parts"].([]map[string]any)
				if ok {
					contents[len(contents)-1]["parts"] = append(prevParts, toolParts...)
					continue
				}
			}

			contents = append(contents, map[string]any{
				"role":  "user",
				"parts": toolParts,
			})
		}
	}
	body["contents"] = contents

	// 5. Tools & ToolChoice
	if len(req.Tools) > 0 {
		var decls []map[string]any
		for _, t := range req.Tools {
			decl := map[string]any{
				"name":        t.Name,
				"description": t.Description,
			}
			if len(t.InputSchema) > 0 {
				var schema map[string]any
				if err := json.Unmarshal(t.InputSchema, &schema); err == nil {
					cleanGeminiSchema(schema)
					if len(schema) > 0 {
						decl["parameters"] = schema
					}
				}
			}
			decls = append(decls, decl)
		}
		body["tools"] = []map[string]any{
			{"functionDeclarations": decls},
		}

		if req.ToolChoice != "" {
			switch req.ToolChoice {
			case react.ToolChoiceAuto:
				body["toolConfig"] = map[string]any{
					"functionCallingConfig": map[string]any{"mode": "AUTO"},
				}
			case react.ToolChoiceNone:
				body["toolConfig"] = map[string]any{
					"functionCallingConfig": map[string]any{"mode": "NONE"},
				}
			case react.ToolChoiceRequired:
				body["toolConfig"] = map[string]any{
					"functionCallingConfig": map[string]any{"mode": "ANY"},
				}
			default:
				body["toolConfig"] = map[string]any{
					"functionCallingConfig": map[string]any{
						"mode":                 "ANY",
						"allowedFunctionNames": []string{req.ToolChoice},
					},
				}
			}
		}
	}

	return body, nil
}

func buildGeminiUserParts(m react.Message) []map[string]any {
	var parts []map[string]any
	for _, p := range m.Content {
		switch p.Type {
		case react.PartText:
			if p.Text != "" {
				parts = append(parts, map[string]any{"text": p.Text})
			}
		case react.PartImage:
			if imgPart := buildGeminiImagePart(p); imgPart != nil {
				parts = append(parts, imgPart)
			}
		}
	}
	if len(parts) == 0 && m.Text() != "" {
		parts = append(parts, map[string]any{"text": m.Text()})
	}
	return parts
}

func buildGeminiImagePart(p react.Part) map[string]any {
	if len(p.Data) > 0 {
		mime := p.MIMEType
		if mime == "" {
			mime = "image/jpeg"
		}
		return map[string]any{
			"inlineData": map[string]string{
				"mimeType": mime,
				"data":     base64.StdEncoding.EncodeToString(p.Data),
			},
		}
	}
	if p.ImageURL != "" {
		if strings.HasPrefix(p.ImageURL, "data:") {
			parts := strings.SplitN(p.ImageURL, ",", 2)
			if len(parts) == 2 {
				header := parts[0]
				data := parts[1]
				mime := "image/jpeg"
				if semi := strings.Index(header, ";"); semi > 5 {
					mime = header[5:semi]
				}
				return map[string]any{
					"inlineData": map[string]string{
						"mimeType": mime,
						"data":     data,
					},
				}
			}
		}
		mime := p.MIMEType
		if mime == "" {
			mime = "image/jpeg"
		}
		return map[string]any{
			"fileData": map[string]string{
				"mimeType": mime,
				"fileUri":  p.ImageURL,
			},
		}
	}
	return nil
}

// cleanGeminiSchema removes unsupported fields (such as $schema and empty/boolean additionalProperties)
// from JSON Schema before passing to Gemini functionDeclarations.
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

func mapGeminiFinishReason(r string) react.StopReason {
	switch r {
	case "STOP":
		return react.StopEnd
	case "MAX_TOKENS":
		return react.StopLength
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY":
		return react.StopContentFilter
	default:
		return react.StopOther
	}
}

// Gemini Wire format structs

type generateContentResponse struct {
	Candidates []struct {
		Content struct {
			Role  string     `json:"role"`
			Parts []partWire `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
		Index        int    `json:"index"`
	} `json:"candidates"`
	UsageMetadata *struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount"`
		TotalTokenCount         int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

type partWire struct {
	Text             string                `json:"text,omitempty"`
	Thought          bool                  `json:"thought,omitempty"`
	ThoughtSignature string                `json:"thoughtSignature,omitempty"`
	FunctionCall     *functionCallWire     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponseWire `json:"functionResponse,omitempty"`
}

type functionCallWire struct {
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type functionResponseWire struct {
	ID       string `json:"id,omitempty"`
	Name     string `json:"name"`
	Response any    `json:"response"`
}
