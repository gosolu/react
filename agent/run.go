package agent

import (
	"context"
	"iter"
	"sync"
)

// Run executes the agent to completion using the provider's non-streaming
// API and returns the final result. sessionID selects the conversation in the
// memory store; an empty sessionID runs statelessly (see WithHistory).
//
// On ErrMaxTurns the partial Result is returned alongside the error.
func (a *Agent) Run(ctx context.Context, sessionID, input string, opts ...RunOption) (*Result, error) {
	rcfg := newRunConfig(opts)
	return a.execute(ctx, sessionID, input, rcfg, false, func(Event) bool { return true })
}

// Stream executes the agent using the provider's streaming API and yields
// events as they happen. The final event of a successful run is EventRunEnd
// carrying the Result. If the run fails, a final EventError is yielded with
// the error. Breaking out of the loop cancels the run and waits for it to
// stop.
//
// Events from sub-agents are forwarded with their own Agent name and Depth>0.
func (a *Agent) Stream(ctx context.Context, sessionID, input string, opts ...RunOption) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		rcfg := newRunConfig(opts)

		ch := make(chan Event)
		var runErr error
		var wg sync.WaitGroup
		wg.Go(func() {
			defer close(ch)
			_, runErr = a.execute(ctx, sessionID, input, rcfg, true, func(ev Event) bool {
				select {
				case ch <- ev:
					return true
				case <-ctx.Done():
					return false
				}
			})
		})

		for ev := range ch {
			if !yield(ev, nil) {
				cancel()
				for range ch {
				}
				wg.Wait()
				return
			}
		}
		wg.Wait()
		if runErr != nil {
			yield(Event{Type: EventError, Agent: a.cfg.name, Err: runErr}, runErr)
		}
	}
}

func newRunConfig(opts []RunOption) runConfig {
	var r runConfig
	for _, o := range opts {
		o(&r)
	}
	return r
}
