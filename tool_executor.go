package react

import (
	"context"
	"errors"
	"fmt"
	"runtime/debug"
	"sync"
	"time"
)

// ToolCallResult is the outcome of executing one call.
type ToolCallResult struct {
	Call     ToolCall
	Result   ToolResult
	Err      error // original error, if any (already reflected in Result)
	Duration time.Duration
}

// ToolExecutor runs batches of tool calls.
type ToolExecutor struct {
	// MaxParallel bounds concurrency (<=0 means 4).
	MaxParallel int
	// Timeout bounds each call (0 means no per-call timeout).
	Timeout time.Duration
}

// Run executes calls and returns results in the same order as calls.
//
// Consecutive non-sequential calls run concurrently; a call whose spec is
// Sequential (per isSequential) runs alone after preceding calls finish.
// ToolHandler errors, panics and timeouts are converted into error Results so the
// model can react to them. onDone (optional) is invoked once per call as it
// completes and receives a pointer to the stored result, so it may mutate it;
// calls to onDone are serialized.
func (e ToolExecutor) Run(ctx context.Context, calls []ToolCall, isSequential func(name string) bool, h ToolHandler, onDone func(i int, cr *ToolCallResult)) []ToolCallResult {
	results := make([]ToolCallResult, len(calls))
	maxPar := e.MaxParallel
	if maxPar <= 0 {
		maxPar = 4
	}
	sem := make(chan struct{}, maxPar)
	var doneMu sync.Mutex

	runOne := func(i int) {
		results[i] = e.exec(ctx, calls[i], h)
		if onDone != nil {
			doneMu.Lock()
			onDone(i, &results[i])
			doneMu.Unlock()
		}
	}

	cancelOne := func(i int, err error) {
		results[i] = ToolCallResult{
			Call:   calls[i],
			Result: ErrorResult("Error: tool execution canceled: " + err.Error()),
			Err:    err,
		}
		if onDone != nil {
			doneMu.Lock()
			onDone(i, &results[i])
			doneMu.Unlock()
		}
	}

	var wg sync.WaitGroup
	for i, c := range calls {
		if err := ctx.Err(); err != nil {
			cancelOne(i, err)
			continue
		}
		if isSequential != nil && isSequential(c.Name) {
			wg.Wait()
			if err := ctx.Err(); err != nil {
				cancelOne(i, err)
				continue
			}
			runOne(i)
			continue
		}
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			cancelOne(i, ctx.Err())
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			runOne(i)
		}()
	}
	wg.Wait()
	return results
}

func (e ToolExecutor) exec(ctx context.Context, call ToolCall, h ToolHandler) (cr ToolCallResult) {
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

	type outcome struct {
		res   ToolResult
		err   error
		panic any
		stack []byte
	}
	done := make(chan outcome, 1)
	go func() {
		var o outcome
		defer func() {
			if r := recover(); r != nil {
				o.panic = r
				o.stack = debug.Stack()
			}
			done <- o
		}()
		o.res, o.err = h(ctx, call)
	}()

	var (
		res ToolResult
		err error
	)
	select {
	case o := <-done:
		if o.panic != nil {
			cr.Err = fmt.Errorf("tool %q panicked: %v\n%s", call.Name, o.panic, o.stack)
			cr.Result = ErrorResult(fmt.Sprintf("Error: tool %q panicked: %v", call.Name, o.panic))
			return cr
		}
		res, err = o.res, o.err
	case <-ctx.Done():
		err = ctx.Err()
	}

	if err != nil {
		cr.Err = err
		msg := err.Error()
		if errors.Is(err, context.DeadlineExceeded) {
			msg = "tool call timed out"
		}
		res.IsError = true
		res.Content = append([]Part{TextPart("Error: " + msg)}, res.Content...)
	}
	cr.Result = res
	return cr
}
