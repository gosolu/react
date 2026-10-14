package react

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

	Text string // TextDelta, ReasoningDelta, ToolCallDelta (args fragment)

	ToolCall   *ToolCall        // ToolCallStart, ToolCallReady, ApprovalRequested, ToolResult
	ToolIndex  int              // index of the call within the turn
	ToolResult *ToolResult      // ToolResult
	Approval   *ApprovalRequest // ApprovalRequested

	Message *Message // MessageComplete
	Usage   *Usage   // MessageComplete (per call), RunEnd (aggregate)
	Result  *Result  // RunEnd
	Err     error    // Error
}

// Result summarises a completed run.
type Result struct {
	// Output is the final assistant text.
	Output string
	// Messages are all messages produced in this run, starting with the user input.
	Messages   []Message
	Usage      Usage
	Turns      int
	StopReason StopReason
}
