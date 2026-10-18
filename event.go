package react

import (
	"encoding/json"
	"time"
)

// EventType identifies an Event.
type EventType string

const (
	EventRunStart          EventType = "run_start"
	EventTurnStart         EventType = "turn_start"
	EventTextDelta         EventType = "text_delta"      // stream mode only
	EventReasoningDelta    EventType = "reasoning_delta" // stream mode only
	EventToolCallStart     EventType = "tool_call_start" // stream mode only
	EventToolCallDelta     EventType = "tool_call_delta" // stream mode only
	EventMessageComplete   EventType = "message_complete"
	EventToolCallReady     EventType = "tool_call_ready"
	EventApprovalRequested EventType = "approval_requested"
	EventToolCallRewritten EventType = "tool_call_rewritten"
	EventToolResult        EventType = "tool_result"
	EventTurnEnd           EventType = "turn_end"
	EventRunEnd            EventType = "run_end"
	EventError             EventType = "error"
)

// Event is emitted during a run. Only the fields relevant to Type are set.
type Event struct {
	Type  EventType
	Agent string // name of the emitting agent (differs for sub-agent events)
	RunID string
	Depth int // 0 for the top-level agent
	Turn  int

	Text string // TextDelta, ReasoningDelta, ToolCallDelta (args fragment), ToolCallRewritten (summary)

	ToolCall     *ToolCall        // ToolCallStart, ToolCallReady, ApprovalRequested, ToolResult, ToolCallRewritten
	ToolIndex    int              // index of the call within the turn
	ToolResult   *ToolResult      // ToolResult
	Approval     *ApprovalRequest // ApprovalRequested
	OriginalArgs json.RawMessage  // ToolCallRewritten

	Message *Message // MessageComplete
	Usage   *Usage   // MessageComplete (per call), RunEnd (aggregate)
	Result  *Result  // RunEnd
	Err     error    // Error

	// Duration is set on EventToolResult (one tool call), EventMessageComplete
	// (one model call), and EventRunEnd (the whole run).
	Duration time.Duration
}

// Result summarises a completed run.
type Result struct {
	// Output is the final assistant text.
	Output string
	// Messages are all messages produced in this run, starting with the user input.
	Messages []Message
	// Usage is the token usage of this run, including its sub-agents.
	Usage Usage
	// Budget is what this run and its sub-agents consumed. For a sub-agent this
	// is the sub-agent's own share.
	Budget BudgetUsage
	// RunID identifies this run; it is also carried by emitted events.
	RunID string
	// Duration is the wall-clock time of this run.
	Duration   time.Duration
	Turns      int
	StopReason StopReason
}
