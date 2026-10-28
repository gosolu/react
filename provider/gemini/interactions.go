package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"maps"
	"strings"

	"github.com/gosolu/react"
	"github.com/gosolu/react/provider"
)

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

	rc, err := p.cfg.PostJSON(
		ctx,
		p.Name(),
		p.interactionsEndpointURL(),
		p.headers(),
		bodyMap,
	)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rc.Close() }()

	var wire interactionResponseWire
	if err := json.NewDecoder(rc).Decode(&wire); err != nil {
		return nil, fmt.Errorf("gemini: decode interaction response: %w", err)
	}

	if wire.Error != nil && wire.Error.Message != "" {
		return nil, fmt.Errorf("gemini api error: %s (code %v)", wire.Error.Message, wire.Error.Code)
	}
	if len(wire.Errors) > 0 && wire.Errors[0].Message != "" {
		return nil, fmt.Errorf("gemini api error: %s (code %s)", wire.Errors[0].Message, wire.Errors[0].Code)
	}

	msg := react.Message{Role: react.RoleAssistant}
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
						if decoded, err := base64.StdEncoding.DecodeString(c.Data); err == nil {
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
		case "incomplete":
			stopReason = react.StopLength
		case "failed", "cancelled":
			stopReason = react.StopContentFilter
		}
	}

	return &react.Response{
		Message:    msg,
		Reasoning:  msg.Reasoning,
		StopReason: stopReason,
		Usage:      wire.Usage.toUsage(),
	}, nil
}

func (p *Provider) streamInteractions(ctx context.Context, req *react.Request) iter.Seq2[react.StreamEvent, error] {
	return func(yield func(react.StreamEvent, error) bool) {
		bodyMap, err := p.buildInteractionsRequestBody(req, true)
		if err != nil {
			yield(react.StreamEvent{}, fmt.Errorf("gemini: build interactions request: %w", err))
			return
		}

		var toolCallSeen, finishedCleanly bool
		var toolCallIndex int

		events := p.cfg.StreamJSON(
			ctx,
			p.Name(),
			p.interactionsEndpointURL(),
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
				if d := eventWire.Delta; d != nil {
					switch d.Type {
					case "text":
						if d.Text != "" {
							if !yield(react.StreamEvent{Type: react.StreamTextDelta, Text: d.Text}, nil) {
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
						idx := max(toolCallIndex-1, 0)
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
					u := eventWire.Metadata.TotalUsage.toUsage()
					if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &u}, nil) {
						return
					}
				}

			case "interaction.status_update":
				if eventWire.Usage != nil {
					u := eventWire.Usage.toUsage()
					if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &u}, nil) {
						return
					}
				}
				if eventWire.Status == "requires_action" {
					finishedCleanly = true
					if !yield(react.StreamEvent{Type: react.StreamDone, StopReason: react.StopToolCall}, nil) {
						return
					}
				} else if eventWire.Status == "completed" {
					finishedCleanly = true
					stopReason := react.StopEnd
					if toolCallSeen {
						stopReason = react.StopToolCall
					}
					if !yield(react.StreamEvent{Type: react.StreamDone, StopReason: stopReason}, nil) {
						return
					}
				}

			case "interaction.requires_action":
				finishedCleanly = true
				if !yield(react.StreamEvent{Type: react.StreamDone, StopReason: react.StopToolCall}, nil) {
					return
				}

			case "interaction.completed":
				finishedCleanly = true
				if eventWire.Interaction != nil && eventWire.Interaction.Usage != nil {
					u := eventWire.Interaction.Usage.toUsage()
					if !yield(react.StreamEvent{Type: react.StreamUsage, Usage: &u}, nil) {
						return
					}
				}
				stopReason := react.StopEnd
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

func (p *Provider) buildInteractionsRequestBody(req *react.Request, stream bool) (map[string]any, error) {
	body := make(map[string]any)
	maps.Copy(body, req.Extra)

	model := req.Model
	if model == "" {
		model = ModelGeminiFlashLatest
	}
	body["model"] = strings.TrimPrefix(model, "models/")

	// Agent override: if agent is explicitly provided in Extra and model wasn't set in Request
	if agent, ok := body["agent"].(string); ok && agent != "" && req.Model == "" {
		delete(body, "model")
	}

	if stream {
		body["stream"] = true
	}

	// 1. System instructions
	if sys := collectSystemInstruction(req); sys != "" {
		body["system_instruction"] = sys
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
	toolNamesByID := buildToolNamesByID(req.Messages)

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
		targetMsgs := nonSystem
		if hasPrevID && len(nonSystem) > 0 {
			targetMsgs = extractLatestTurn(nonSystem)
		}

		var steps []map[string]any
		for _, m := range targetMsgs {
			switch m.Role {
			case react.RoleUser:
				steps = append(steps, map[string]any{
					"type":    "user_input",
					"content": buildInteractionsContent(m),
				})

			case react.RoleAssistant:
				if m.Reasoning != "" || (m.ReasoningSignature != "" && len(m.ToolCalls) == 0) {
					thoughtStep := map[string]any{"type": "thought"}
					if m.Reasoning != "" {
						thoughtStep["text"] = m.Reasoning
					}
					if m.ReasoningSignature != "" {
						thoughtStep["signature"] = m.ReasoningSignature
					}
					steps = append(steps, thoughtStep)
				}
				if content := buildInteractionsContent(m); len(content) > 0 {
					steps = append(steps, map[string]any{
						"type":    "model_output",
						"content": content,
					})
				}
				for i, tc := range m.ToolCalls {
					fcStep := map[string]any{
						"type":      "function_call",
						"id":        tc.ID,
						"name":      tc.Name,
						"arguments": parseToolArgs(tc.Arguments),
					}
					if sig := resolveToolCallSignature(m, tc, i); sig != "" {
						fcStep["signature"] = sig
					}
					steps = append(steps, fcStep)
				}

			case react.RoleTool:
				resContent := buildInteractionsContent(m)
				if len(resContent) == 0 {
					resContent = []map[string]any{{"type": "text", "text": m.Text()}}
				}
				frStep := map[string]any{
					"type":    "function_result",
					"call_id": m.ToolCallID,
					"name":    resolveToolName(m, toolNamesByID),
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
		if schema := parseCleanedSchema(t.InputSchema); schema != nil {
			toolDef["parameters"] = schema
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
	if msgs[len(msgs)-1].Role == react.RoleTool {
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
				if mime, data, ok := parseDataURI(p.ImageURL); ok {
					content = append(content, map[string]any{
						"type":      "image",
						"mime_type": mime,
						"data":      data,
					})
					continue
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

func (u *interactionUsageWire) toUsage() react.Usage {
	if u == nil {
		return react.Usage{}
	}
	return react.Usage{
		InputTokens:     u.TotalInputTokens,
		OutputTokens:    u.TotalOutputTokens + u.TotalThoughtTokens,
		CacheReadTokens: u.TotalCachedTokens,
	}
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
