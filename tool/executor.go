package tool

import (
	"context"
	"errors"
	"fmt"
	"runtime/debug"
	"sync"
	"time"

	"github.com/gosolu/react/llm"
)

// CallResult is the outcome of executing one call.
type CallResult struct {
	Call     llm.ToolCall
	Result   Result
	Err      error // original error, if any (already reflected in Result)
	Duration time.Duration
}

// Executor runs batches of tool calls.
type Executor struct {
	// MaxParallel bounds concurrency (<=0 means 4).
	MaxParallel int
	// Timeout bounds each call (0 means no per-call timeout).
	Timeout time.Duration
}

// Run executes calls and returns results in the same order as calls.
//
// Consecutive non-sequential calls run concurrently; a call whose spec is
// Sequential (per isSequential) runs alone after preceding calls finish.
// Handler errors, panics and timeouts are converted into error Results so the
// model can react to them. onDone (optional) is invoked once per call as it
// completes; calls to onDone are serialized.
func (e Executor) Run(ctx context.Context, calls []llm.ToolCall, isSequential func(name string) bool, h Handler, onDone func(i int, cr CallResult)) []CallResult {
	results := make([]CallResult, len(calls))
	maxPar := e.MaxParallel
	if maxPar <= 0 {
		maxPar = 4
	}
	sem := make(chan struct{}, maxPar)
	var doneMu sync.Mutex

	runOne := func(i int) {
		cr := e.exec(ctx, calls[i], h)
		results[i] = cr
		if onDone != nil {
			doneMu.Lock()
			onDone(i, cr)
			doneMu.Unlock()
		}
	}

	var wg sync.WaitGroup
	for i, c := range calls {
		if isSequential != nil && isSequential(c.Name) {
			wg.Wait()
			runOne(i)
			continue
		}
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			runOne(i)
		}()
	}
	wg.Wait()
	return results
}

func (e Executor) exec(ctx context.Context, call llm.ToolCall, h Handler) (cr CallResult) {
	start := time.Now()
	cr.Call = call
	defer func() {
		if r := recover(); r != nil {
			cr.Err = fmt.Errorf("tool %q panicked: %v\n%s", call.Name, r, debug.Stack())
			cr.Result = ErrorResult(fmt.Sprintf("Error: tool %q panicked: %v", call.Name, r))
		}
		cr.Duration = time.Since(start)
	}()
	if e.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, e.Timeout)
		defer cancel()
	}
	res, err := h(ctx, call)
	if err != nil {
		cr.Err = err
		msg := err.Error()
		if errors.Is(err, context.DeadlineExceeded) {
			msg = "tool call timed out"
		}
		res.IsError = true
		res.Content = append([]llm.Part{llm.TextPart("Error: " + msg)}, res.Content...)
	}
	cr.Result = res
	return cr
}
