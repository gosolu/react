// Memory defines conversation session storage.
//
// The agent only depends on the Store interface. NewInMemory is the default
// implementation; integrate external session management (Redis, SQL, a remote
// service, ...) by implementing Store and passing it to WithMemory.
package react

import (
	"context"
	"sync"

)

// Store persists conversation history keyed by session ID.
//
// Implementations must be safe for concurrent use. Load of an unknown session
// returns an empty history and no error. Implementations must not retain or
// mutate the caller's slices.
type Store interface {
	Load(ctx context.Context, sessionID string) ([]Message, error)
	Append(ctx context.Context, sessionID string, msgs ...Message) error
	Replace(ctx context.Context, sessionID string, msgs []Message) error
	Delete(ctx context.Context, sessionID string) error
}

// InMemory is a process-local Store.
type InMemory struct {
	mu       sync.RWMutex
	sessions map[string][]Message
}

// NewInMemory returns an empty in-memory store.
func NewInMemory() *InMemory { return &InMemory{sessions: map[string][]Message{}} }

func (s *InMemory) Load(_ context.Context, id string) ([]Message, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return CloneMessages(s.sessions[id]), nil
}

func (s *InMemory) Append(_ context.Context, id string, msgs ...Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions[id] = append(s.sessions[id], CloneMessages(msgs)...)
	return nil
}

func (s *InMemory) Replace(_ context.Context, id string, msgs []Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions[id] = CloneMessages(msgs)
	return nil
}

func (s *InMemory) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.sessions, id)
	return nil
}

// Sessions lists known session IDs.
func (s *InMemory) Sessions() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]string, 0, len(s.sessions))
	for k := range s.sessions {
		out = append(out, k)
	}
	return out
}
