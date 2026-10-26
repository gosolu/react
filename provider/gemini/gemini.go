// Package gemini implements a Google Gemini react.Provider.
// It supports native Gemini v1beta generateContent and streamGenerateContent API,
// as well as OpenAI-compatible Chat Completions mode when configured.
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
	ModelGemini4Argon  = "gemini-4-argon"
	ModelGemini4Flash  = "gemini-4-flash"
	ModelGemini38Flash = "gemini-3.8-flash"
)

// Config configures the Gemini provider.
type Config struct {
	APIKey      string
	BaseURL     string           // defaults to DefaultBaseURL
	APIType     provider.APIType // defaults to native Gemini API; or provider.APITypeChatCompletions
	HTTPClient  *http.Client     // defaults to http.DefaultClient
	RetryPolicy provider.RetryPolicy
	ExtraHeader map[string]string
}

// Provider implements react.Provider for Google Gemini models.
type Provider struct {
	cfg Config
}

// New creates a new Gemini provider using native generateContent API.
func New(cfg Config) *Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	return &Provider{cfg: cfg}
}

// NewChatCompletions creates a new Gemini provider configured to use the
// OpenAI-compatible Chat Completions endpoint (/openai/chat/completions).
func NewChatCompletions(cfg Config) *Provider {
	cfg.APIType = provider.APITypeChatCompletions
	return New(cfg)
}

// Name implements react.Provider.
func (p *Provider) Name() string {
	return "gemini"
}

// Generate implements react.Provider (non-streaming).
func (p *Provider) Generate(ctx context.Context, req *react.Request) (*react.Response, error) {
	if p.cfg.APIType == provider.APITypeChatCompletions {
		return p.generateChatCompletions(ctx, req)
	}
	return p.generateNative(ctx, req)
}

// Stream implements react.Provider (server-sent events).
func (p *Provider) Stream(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	if p.cfg.APIType == provider.APITypeChatCompletions {
		return p.streamChatCompletions(ctx, req)
	}
	return p.streamNative(ctx, req)
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
						Type:       react.StreamToolCallDelta,
						Index:      idx,
						Signature:  part.ThoughtSignature,
						ArgsDelta:  string(args),
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
