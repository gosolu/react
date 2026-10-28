package provider

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"strings"
)

// SSEEvent is a single server-sent event.
type SSEEvent struct {
	Event string
	Data  string
}

// ReadSSE parses a text/event-stream body.
func ReadSSE(r io.Reader) iter.Seq2[SSEEvent, error] {
	return func(yield func(SSEEvent, error) bool) {
		sc := bufio.NewScanner(r)
		sc.Buffer(make([]byte, 64*1024), 16*1024*1024)
		var ev SSEEvent
		var data []string
		flush := func() bool {
			if len(data) == 0 && ev.Event == "" {
				return true
			}
			ev.Data = strings.Join(data, "\n")
			ok := yield(ev, nil)
			ev, data = SSEEvent{}, nil
			return ok
		}
		for sc.Scan() {
			line := sc.Text()
			if line == "" {
				if !flush() {
					return
				}
				continue
			}
			if strings.HasPrefix(line, ":") {
				continue // comment
			}
			field, value, _ := strings.Cut(line, ":")
			value = strings.TrimPrefix(value, " ")
			switch field {
			case "event":
				ev.Event = value
			case "data":
				data = append(data, value)
			}
		}
		if err := sc.Err(); err != nil {
			yield(SSEEvent{}, err)
			return
		}
		flush()
	}
}

// CheckSSEError inspects an SSE event for mid-stream error payloads and returns
// a formatted error with prefix if found, or nil if the event is not an error.
func CheckSSEError(sse SSEEvent, prefix string) error {
	if sse.Event != "error" && !strings.Contains(sse.Data, `"error"`) {
		return nil
	}
	var payload struct {
		Type  string `json:"type"`
		Error *struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		} `json:"error"`
		Errors []struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"errors"`
		Response *struct {
			Error *struct {
				Message string `json:"message"`
				Type    string `json:"type"`
			} `json:"error"`
		} `json:"response"`
	}
	if json.Unmarshal([]byte(sse.Data), &payload) == nil {
		if e := payload.Error; e != nil && e.Message != "" {
			if e.Type != "" {
				return fmt.Errorf("%s: %s (%s)", prefix, e.Message, e.Type)
			}
			if e.Code != nil && e.Code != "" && e.Code != float64(0) {
				return fmt.Errorf("%s: %s (code %v)", prefix, e.Message, e.Code)
			}
			return fmt.Errorf("%s: %s", prefix, e.Message)
		}
		if len(payload.Errors) > 0 && payload.Errors[0].Message != "" {
			if c := payload.Errors[0].Code; c != "" {
				return fmt.Errorf("%s: %s (code %s)", prefix, payload.Errors[0].Message, c)
			}
			return fmt.Errorf("%s: %s", prefix, payload.Errors[0].Message)
		}
		if payload.Response != nil && payload.Response.Error != nil && payload.Response.Error.Message != "" {
			return fmt.Errorf("%s: %s", prefix, payload.Response.Error.Message)
		}
		if payload.Type == "error" {
			return fmt.Errorf("%s: %s", prefix, sse.Data)
		}
	}
	if sse.Event == "error" {
		return fmt.Errorf("%s: %s", prefix, sse.Data)
	}
	return nil
}
