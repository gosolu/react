package react

import (
	"encoding/json"
	"strings"
)

// Role is the author of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// PartType identifies the kind of content a Part carries.
type PartType string

const (
	PartText  PartType = "text"
	PartImage PartType = "image"
)

// Part is a single piece of message content.
type Part struct {
	Type PartType `json:"type"`
	Text string   `json:"text,omitempty"`

	// Image content: either a URL or inline Data with a MIMEType.
	ImageURL string `json:"image_url,omitempty"`
	MIMEType string `json:"mime_type,omitempty"`
	Data     []byte `json:"data,omitempty"`
}

// TextPart returns a text Part.
func TextPart(s string) Part { return Part{Type: PartText, Text: s} }

// ImageURLPart returns an image Part referencing a URL.
func ImageURLPart(url string) Part { return Part{Type: PartImage, ImageURL: url} }

// ImageDataPart returns an image Part with inline bytes.
func ImageDataPart(mime string, data []byte) Part {
	return Part{Type: PartImage, MIMEType: mime, Data: data}
}

// ToolCall is a model's request to invoke a tool.
type ToolCall struct {
	ID               string          `json:"id"`
	Name             string          `json:"name"`
	Arguments        json.RawMessage `json:"arguments"`
	ThoughtSignature string          `json:"thought_signature,omitempty"`
}

// Message is one entry in a conversation.
type Message struct {
	Role    Role   `json:"role"`
	Content []Part `json:"content,omitempty"`

	// Reasoning is model reasoning/thinking text (e.g. DeepSeek reasoner, Claude extended thinking).
	Reasoning string `json:"reasoning,omitempty"`
	// ReasoningSignature holds the provider-specific cryptographic signature (e.g. Anthropic).
	ReasoningSignature string `json:"reasoning_signature,omitempty"`

	// ToolCalls is set on assistant messages that request tool execution.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// ToolCallID and Name are set on tool-result messages (Role == RoleTool).
	ToolCallID string `json:"tool_call_id,omitempty"`
	Name       string `json:"name,omitempty"`
	// IsError marks a tool result as a failure.
	IsError bool `json:"is_error,omitempty"`
}

// Text concatenates all text parts of the message.
func (m Message) Text() string {
	var b strings.Builder
	for _, p := range m.Content {
		if p.Type == PartText {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// Clone returns a deep copy of the message.
func (m Message) Clone() Message {
	out := m
	if m.Content != nil {
		out.Content = make([]Part, len(m.Content))
		for i, p := range m.Content {
			if p.Data != nil {
				p.Data = append([]byte(nil), p.Data...)
			}
			out.Content[i] = p
		}
	}
	if m.ToolCalls != nil {
		out.ToolCalls = make([]ToolCall, len(m.ToolCalls))
		for i, c := range m.ToolCalls {
			c.Arguments = append(json.RawMessage(nil), c.Arguments...)
			out.ToolCalls[i] = c
		}
	}
	return out
}

// CloneMessages deep-copies a slice of messages.
func CloneMessages(msgs []Message) []Message {
	if msgs == nil {
		return nil
	}
	out := make([]Message, len(msgs))
	for i, m := range msgs {
		out[i] = m.Clone()
	}
	return out
}

// UserText builds a user message with a single text part.
func UserText(s string) Message {
	return Message{Role: RoleUser, Content: []Part{TextPart(s)}}
}

// AssistantText builds an assistant message with a single text part.
func AssistantText(s string) Message {
	return Message{Role: RoleAssistant, Content: []Part{TextPart(s)}}
}

// SystemText builds a system message with a single text part.
func SystemText(s string) Message {
	return Message{Role: RoleSystem, Content: []Part{TextPart(s)}}
}

// ToolResultMessage builds a tool-result message.
func ToolResultMessage(callID, name string, content []Part, isError bool) Message {
	return Message{Role: RoleTool, ToolCallID: callID, Name: name, Content: content, IsError: isError}
}
