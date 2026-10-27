package provider

import (
	"bufio"
	"context"
	"errors"
	"io"
	"iter"
	"math/rand/v2"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
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

// DefaultHTTPClient provides a sane default HTTP client with transport timeouts.
var DefaultHTTPClient = &http.Client{
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
}

// RetryPolicy controls retries of transient HTTP failures.
type RetryPolicy struct {
	MaxRetries int           // default 2 when zero; set negative to disable
	BaseDelay  time.Duration // default 500ms
	MaxDelay   time.Duration // default 10s
}

// Do sends the request built by newReq, retrying on network errors, 429 and
// 5xx responses with exponential backoff (honouring Retry-After). A non-2xx
// final response is returned as *APIError with the body consumed.
func (p RetryPolicy) Do(ctx context.Context, client *http.Client, provider string, newReq func(context.Context) (*http.Request, error)) (*http.Response, error) {
	if client == nil {
		client = DefaultHTTPClient
	}
	maxRetries := p.MaxRetries
	if maxRetries == 0 {
		maxRetries = 2
	} else if maxRetries < 0 {
		maxRetries = 0
	}
	base, maxD := p.BaseDelay, p.MaxDelay
	if base <= 0 {
		base = 500 * time.Millisecond
	}
	if maxD <= 0 {
		maxD = 10 * time.Second
	}

	var lastErr error
	for attempt := 0; ; attempt++ {
		req, err := newReq(ctx)
		if err != nil {
			return nil, err
		}
		resp, err := client.Do(req)
		var wait time.Duration
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			lastErr = err
		} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return resp, nil
		} else {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
			_ = resp.Body.Close()
			apiErr := &APIError{Provider: provider, StatusCode: resp.StatusCode, Body: string(body)}
			if !apiErr.Retryable() {
				return nil, apiErr
			}
			lastErr = apiErr
			if s, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && s > 0 {
				wait = time.Duration(s) * time.Second
			}
		}
		if attempt >= maxRetries {
			return nil, lastErr
		}
		if wait == 0 {
			wait = base << attempt
			wait += time.Duration(rand.Int64N(int64(wait)/2 + 1))
		}
		wait = min(wait, maxD)
		select {
		case <-ctx.Done():
			return nil, errors.Join(ctx.Err(), lastErr)
		case <-time.After(wait):
		}
	}
}
