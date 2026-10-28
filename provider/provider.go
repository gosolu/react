// Package provider provides common configuration, transport, SSE, and protocol
// utilities for model provider implementations.
package provider

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
	"math/rand/v2"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gosolu/react"
	"github.com/gosolu/react/pkg/utf8util"
)

// APIType specifies the model API protocol family.
type APIType string

const (
	// APITypeChatCompletions routes requests to the standard /chat/completions endpoint.
	APITypeChatCompletions APIType = "chat_completions"

	// APITypeResponses routes requests to the OpenAI-compatible Responses API (/responses).
	APITypeResponses APIType = "responses"

	// APITypeInteractions routes requests to the Google Gemini Interactions API (/interactions).
	APITypeInteractions APIType = "interactions"
)

// APIError is returned when a provider responds with a non-success HTTP status.
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
	if e.Provider != "" {
		return fmt.Sprintf("%s: HTTP %d: %s", e.Provider, e.StatusCode, body)
	}
	return fmt.Sprintf("HTTP %d: %s", e.StatusCode, body)
}

// Retryable reports whether the request may succeed if retried.
func (e *APIError) Retryable() bool {
	return e.StatusCode == 429 || e.StatusCode >= 500
}

// Client defines the remote request methods required by model providers.
type Client interface {
	// Post sends a JSON POST request and returns the response body stream.
	// The caller is responsible for closing the returned ReadCloser.
	Post(ctx context.Context, url string, headers map[string]string, body io.Reader) (io.ReadCloser, error)

	// Stream sends a streaming JSON POST request ("Accept: text/event-stream")
	// and returns an iterator of SSE events, closing the response stream automatically when done.
	Stream(ctx context.Context, url string, headers map[string]string, body io.Reader) iter.Seq2[SSEEvent, error]
}

// Config is the unified configuration shared by all model providers.
type Config struct {
	APIKey      string
	BaseURL     string
	APIType     APIType
	ExtraHeader map[string]string
	Client      Client // defaults to DefaultClient when nil
}

// DefaultClient is the default remote Client implementation with transport
// timeouts and exponential-backoff retries on transient HTTP failures (429/5xx).
var DefaultClient Client = &defaultClient{
	httpClient: &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			ResponseHeaderTimeout: 60 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			IdleConnTimeout:       90 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
		},
	},
	maxRetries: 2,
	baseDelay:  500 * time.Millisecond,
	maxDelay:   10 * time.Second,
}

type defaultClient struct {
	httpClient *http.Client
	maxRetries int
	baseDelay  time.Duration
	maxDelay   time.Duration
}

func (c *defaultClient) Post(ctx context.Context, url string, headers map[string]string, body io.Reader) (io.ReadCloser, error) {
	var payload []byte
	if body != nil {
		var err error
		payload, err = io.ReadAll(body)
		if err != nil {
			return nil, fmt.Errorf("read request body: %w", err)
		}
	}

	var lastErr error
	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		for k, v := range headers {
			req.Header.Set(k, v)
		}

		resp, err := c.httpClient.Do(req)
		var wait time.Duration
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			lastErr = err
		} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return resp.Body, nil
		} else {
			respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
			_ = resp.Body.Close()
			apiErr := &APIError{StatusCode: resp.StatusCode, Body: string(respBody)}
			if !apiErr.Retryable() {
				return nil, apiErr
			}
			lastErr = apiErr
			if s, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && s > 0 {
				wait = time.Duration(s) * time.Second
			}
		}

		if attempt >= c.maxRetries {
			return nil, lastErr
		}
		if wait == 0 {
			wait = c.baseDelay << attempt
			wait += time.Duration(rand.Int64N(int64(wait)/2 + 1))
		}
		wait = min(wait, c.maxDelay)
		select {
		case <-ctx.Done():
			return nil, errors.Join(ctx.Err(), lastErr)
		case <-time.After(wait):
		}
	}
}

func (c *defaultClient) Stream(ctx context.Context, url string, headers map[string]string, body io.Reader) iter.Seq2[SSEEvent, error] {
	return func(yield func(SSEEvent, error) bool) {
		streamHeaders := make(map[string]string, len(headers)+1)
		maps.Copy(streamHeaders, headers)
		streamHeaders["Accept"] = "text/event-stream"

		rc, err := c.Post(ctx, url, streamHeaders, body)
		if err != nil {
			yield(SSEEvent{}, err)
			return
		}
		defer func() { _ = rc.Close() }()

		for ev, err := range ReadSSE(rc) {
			if !yield(ev, err) {
				return
			}
		}
	}
}

func (c Config) client() Client {
	if c.Client != nil {
		return c.Client
	}
	return DefaultClient
}

// BearerHeaders builds a header map with Content-Type, Bearer Authorization (if APIKey is set),
// any provider default headers, and c.ExtraHeader overrides.
func (c Config) BearerHeaders(defaults map[string]string) map[string]string {
	headers := make(map[string]string, len(defaults)+len(c.ExtraHeader)+2)
	headers["Content-Type"] = "application/json"
	if c.APIKey != "" {
		headers["Authorization"] = "Bearer " + c.APIKey
	}
	maps.Copy(headers, defaults)
	maps.Copy(headers, c.ExtraHeader)
	return headers
}

// PostJSON marshals payload as JSON and invokes Client.Post.
func (c Config) PostJSON(ctx context.Context, providerName, url string, headers map[string]string, payload map[string]any) (io.ReadCloser, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("%s: marshal request: %w", providerName, err)
	}
	rc, err := c.client().Post(ctx, url, headers, bytes.NewReader(data))
	if err != nil {
		var apiErr *APIError
		if errors.As(err, &apiErr) && apiErr.Provider == "" {
			apiErr.Provider = providerName
		}
		return nil, err
	}
	return rc, nil
}

// StreamJSON marshals payload as JSON and invokes Client.Stream.
func (c Config) StreamJSON(ctx context.Context, providerName, url string, headers map[string]string, payload map[string]any) iter.Seq2[SSEEvent, error] {
	return func(yield func(SSEEvent, error) bool) {
		data, err := json.Marshal(payload)
		if err != nil {
			yield(SSEEvent{}, fmt.Errorf("%s: marshal request: %w", providerName, err))
			return
		}
		for ev, err := range c.client().Stream(ctx, url, headers, bytes.NewReader(data)) {
			if err != nil {
				var apiErr *APIError
				if errors.As(err, &apiErr) && apiErr.Provider == "" {
					apiErr.Provider = providerName
				}
			}
			if !yield(ev, err) {
				return
			}
		}
	}
}

// ResolveOpenAIEndpoint returns the endpoint URL for OpenAI-compatible
// Chat Completions (/chat/completions) or Responses (/responses) APIs.
func ResolveOpenAIEndpoint(baseURL string, apiType APIType) string {
	baseURL = strings.TrimRight(baseURL, "/")
	if apiType == APITypeResponses {
		baseURL = strings.TrimSuffix(baseURL, "/chat/completions")
		if strings.HasSuffix(baseURL, "/responses") {
			return baseURL
		}
		return baseURL + "/responses"
	}
	baseURL = strings.TrimSuffix(baseURL, "/responses")
	if strings.HasSuffix(baseURL, "/chat/completions") {
		return baseURL
	}
	return baseURL + "/chat/completions"
}

// GenerateOpenAICompat executes a non-streaming request against an OpenAI-compatible
// Chat Completions or Responses endpoint.
func (c Config) GenerateOpenAICompat(
	ctx context.Context,
	providerName string,
	req *react.Request,
	headers map[string]string,
) (*react.Response, error) {
	endpoint := ResolveOpenAIEndpoint(c.BaseURL, c.APIType)
	var body map[string]any
	if c.APIType == APITypeResponses {
		body = BuildResponsesRequestBody(req, false)
	} else {
		body = BuildChatCompletionsRequestBody(req, false)
	}

	rc, err := c.PostJSON(ctx, providerName, endpoint, headers, body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rc.Close() }()

	if c.APIType == APITypeResponses {
		return ParseResponsesResponse(rc)
	}
	return ParseChatCompletionsResponse(rc)
}

// StreamOpenAICompat executes a streaming SSE request against an OpenAI-compatible
// Chat Completions or Responses endpoint.
func (c Config) StreamOpenAICompat(
	ctx context.Context,
	providerName string,
	req *react.Request,
	headers map[string]string,
) iter.Seq2[react.StreamEvent, error] {
	endpoint := ResolveOpenAIEndpoint(c.BaseURL, c.APIType)
	var body map[string]any
	if c.APIType == APITypeResponses {
		body = BuildResponsesRequestBody(req, true)
	} else {
		body = BuildChatCompletionsRequestBody(req, true)
	}

	events := c.StreamJSON(ctx, providerName, endpoint, headers, body)
	if c.APIType == APITypeResponses {
		return StreamResponsesEvents(ctx, events)
	}
	return StreamChatCompletionsEvents(ctx, events)
}

// HasImageParts reports whether parts contains at least one PartImage.
func HasImageParts(parts []react.Part) bool {
	for _, p := range parts {
		if p.Type == react.PartImage {
			return true
		}
	}
	return false
}

// ResolveImageURL returns p.ImageURL if non-empty, or encodes p.Data as a base64 data URI.
func ResolveImageURL(p react.Part) string {
	if p.ImageURL != "" {
		return p.ImageURL
	}
	if len(p.Data) == 0 {
		return ""
	}
	mime := p.MIMEType
	if mime == "" {
		mime = "image/jpeg"
	}
	return fmt.Sprintf("data:%s;base64,%s", mime, base64.StdEncoding.EncodeToString(p.Data))
}
