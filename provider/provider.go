// Package provider provides common HTTP, SSE, and retry utilities for
// model provider implementations.
package provider

import (
	"fmt"

	"github.com/gosolu/react/pkg/utf8util"
)

// APIError is returned when a provider responds with a non-success status.
type APIError struct {
	Provider   string
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	body := e.Body
	if len(body) > 500 {
		body = utf8util.TruncateString(body, 500) + "..."
	}
	return fmt.Sprintf("%s: HTTP %d: %s", e.Provider, e.StatusCode, body)
}

// Retryable reports whether the request may succeed if retried.
func (e *APIError) Retryable() bool {
	return e.StatusCode == 429 || e.StatusCode >= 500
}
