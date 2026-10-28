package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"maps"
	"strings"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider"
)

func (p *Provider) nativeEndpointURL(model string, stream bool) string {
	if model == "" {
		model = ModelGemini38Flash
	}
	model = strings.TrimPrefix(model, "models/")
	action := "generateContent"
	if stream {
		action = "streamGenerateContent?alt=sse"
	}
	return fmt.Sprintf("%s/models/%s:%s", p.cfg.BaseURL, model, action)
}

func (p *Provider) generateNative(ctx context.Context, req *react.Request) (*react.Response, error) {
	bodyMap, err := p.buildNativeRequestBody(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: build request: %w", err)
	}

	rc, err := p.cfg.PostJSON(
		ctx,
		p.Name(),
		p.nativeEndpointURL(req.Model, false),
		p.headers(),
		bodyMap,
	)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rc.Close() }()

	var genResp generateContentResponse
	if err := json.NewDecoder(rc).Decode(&genResp); err != nil {
		return nil, fmt.Errorf("gemini: decode response: %w", err)
	}
	if genResp.Error != nil && genResp.Error.Message != "" {
		return nil, fmt.Errorf("gemini api error: %s (code %d)", genResp.Error.Message, genResp.Error.Code)
	}
	if len(genResp.Candidates) == 0 {
		if genResp.PromptFeedback != nil && genResp.PromptFeedback.BlockReason != "" {
			return &react.Response{StopReason: react.StopContentFilter}, nil
		}
		return nil, errors.New("gemini: empty candidates in response")
	}

	cand := genResp.Candidates[0]
	msg := react.Message{Role: react.RoleAssistant}

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

	// Gemini sets finishReason = "STOP" even when emitting functionCall parts.
	stopReason := mapGeminiFinishReason(cand.FinishReason)
	if len(msg.ToolCalls) > 0 {
		stopReason = react.StopToolCall
	}

	return &react.Response{
		Message:    msg,
		Reasoning:  msg.Reasoning,
		StopReason: stopReason,
		Usage:      genResp.UsageMetadata.toUsage(),
	}, nil
}

func (p *Provider) streamNative(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		bodyMap, err := p.buildNativeRequestBody(req)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("gemini: build request: %w", err))
			return
		}

		var toolCallSeen, finishedCleanly bool
		var toolCallIndex int

		events := p.cfg.StreamJSON(
			ctx,
			p.Name(),
			p.nativeEndpointURL(req.Model, true),
			p.headers(),
			bodyMap,
		)

		for sse, sseErr := range events {
			if sseErr != nil {
				if sseErr != io.EOF && ctx.Err() == nil {
					yield(react.StreamEvent{}, sseErr)
				}
				return
			}
			if sse.Data == "[DONE]" {
				return
			}
			if err := provider.CheckSSEError(sse, "gemini stream error"); err != nil {
				yield(react.StreamEvent{}, err)
				return
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
				u := chunk.UsageMetadata.toUsage()
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
					if !yield(react.StreamEvent{Type: react.StreamTextDelta, Text: part.Text}, nil) {
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
				if !yield(react.StreamEvent{Type: react.StreamDone, StopReason: stopReason}, nil) {
					return
				}
			}
		}

		if !finishedCleanly && ctx.Err() == nil {
			yield(react.StreamEvent{}, io.ErrUnexpectedEOF)
		}
	}
}

func (p *Provider) buildNativeRequestBody(req *react.Request) (map[string]any, error) {
	body := make(map[string]any)
	maps.Copy(body, req.Extra)

	// 1. System instructions
	if sys := collectSystemInstruction(req); sys != "" {
		body["systemInstruction"] = map[string]any{
			"parts": []map[string]any{{"text": sys}},
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

	// 3. Map tool call IDs to tool names
	toolNamesByID := buildToolNamesByID(req.Messages)

	// 4. Build contents array with turn collapsing for tool results
	var contents []map[string]any
	for _, m := range req.Messages {
		switch m.Role {
		case react.RoleSystem:
			continue
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
				fc := map[string]any{
					"name": tc.Name,
					"args": parseToolArgs(tc.Arguments),
				}
				if tc.ID != "" {
					fc["id"] = tc.ID
				}
				part := map[string]any{"functionCall": fc}
				if sig := resolveToolCallSignature(m, tc, i); sig != "" {
					part["thoughtSignature"] = sig
				}
				parts = append(parts, part)
			}
			contents = append(contents, map[string]any{
				"role":  "model",
				"parts": parts,
			})
		case react.RoleTool:
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
				"name":     resolveToolName(m, toolNamesByID),
				"response": respObj,
			}
			if m.ToolCallID != "" {
				fr["id"] = m.ToolCallID
			}

			toolParts := []map[string]any{{"functionResponse": fr}}
			for _, p := range m.Content {
				if p.Type == react.PartImage {
					if imgPart := buildGeminiImagePart(p); imgPart != nil {
						toolParts = append(toolParts, imgPart)
					}
				}
			}

			// Collapse consecutive user/tool turns into a single user turn
			if n := len(contents); n > 0 && contents[n-1]["role"] == "user" {
				if prevParts, ok := contents[n-1]["parts"].([]map[string]any); ok {
					contents[n-1]["parts"] = append(prevParts, toolParts...)
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
		decls := make([]map[string]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			decl := map[string]any{
				"name":        t.Name,
				"description": t.Description,
			}
			if schema := parseCleanedSchema(t.InputSchema); schema != nil {
				decl["parameters"] = schema
			}
			decls = append(decls, decl)
		}
		body["tools"] = []map[string]any{
			{"functionDeclarations": decls},
		}

		if req.ToolChoice != "" {
			var fcCfg map[string]any
			switch req.ToolChoice {
			case react.ToolChoiceAuto:
				fcCfg = map[string]any{"mode": "AUTO"}
			case react.ToolChoiceNone:
				fcCfg = map[string]any{"mode": "NONE"}
			case react.ToolChoiceRequired:
				fcCfg = map[string]any{"mode": "ANY"}
			default:
				fcCfg = map[string]any{
					"mode":                 "ANY",
					"allowedFunctionNames": []string{req.ToolChoice},
				}
			}
			body["toolConfig"] = map[string]any{"functionCallingConfig": fcCfg}
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
		if mime, data, ok := parseDataURI(p.ImageURL); ok {
			return map[string]any{
				"inlineData": map[string]string{
					"mimeType": mime,
					"data":     data,
				},
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

// Gemini Native Wire format structs

type nativeUsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
}

func (u *nativeUsageMetadata) toUsage() react.Usage {
	if u == nil {
		return react.Usage{}
	}
	return react.Usage{
		InputTokens:     u.PromptTokenCount,
		OutputTokens:    u.CandidatesTokenCount + u.ThoughtsTokenCount,
		CacheReadTokens: u.CachedContentTokenCount,
	}
}

type generateContentResponse struct {
	Candidates []struct {
		Content struct {
			Role  string     `json:"role"`
			Parts []partWire `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
		Index        int    `json:"index"`
	} `json:"candidates"`
	UsageMetadata  *nativeUsageMetadata `json:"usageMetadata"`
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
