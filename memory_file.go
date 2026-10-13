package react

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// FileStore persists conversation history to disk as JSON files,
// using atomic write-and-rename for crash resistance.
type FileStore struct {
	dir string
	mu  sync.RWMutex
}

type fileSessionRecord struct {
	SessionID string    `json:"session_id"`
	Messages  []Message `json:"messages"`
	UpdatedAt time.Time `json:"updated_at"`
}

// NewFileStore creates or opens a directory-backed session store.
func NewFileStore(dir string) (*FileStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("react: failed to create store directory: %w", err)
	}
	return &FileStore{dir: dir}, nil
}

func (s *FileStore) filePath(id string) string {
	encoded := hex.EncodeToString([]byte(id))
	return filepath.Join(s.dir, encoded+".json")
}

// Load loads messages for sessionID. If the session does not exist,
// an empty slice and nil error are returned.
func (s *FileStore) Load(_ context.Context, sessionID string) ([]Message, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	p := s.filePath(sessionID)
	data, err := os.ReadFile(p)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("react: failed to read session file: %w", err)
	}

	var rec fileSessionRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return nil, fmt.Errorf("react: failed to parse session file: %w", err)
	}
	return CloneMessages(rec.Messages), nil
}

// Append appends messages to sessionID.
func (s *FileStore) Append(ctx context.Context, sessionID string, msgs ...Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var existing []Message
	p := s.filePath(sessionID)
	if data, err := os.ReadFile(p); err == nil {
		var rec fileSessionRecord
		if err := json.Unmarshal(data, &rec); err == nil {
			existing = rec.Messages
		}
	} else if !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("react: failed to read session file: %w", err)
	}

	all := append(existing, CloneMessages(msgs)...)
	return s.atomicWrite(sessionID, all)
}

// Replace overwrites the message history of sessionID.
func (s *FileStore) Replace(_ context.Context, sessionID string, msgs []Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.atomicWrite(sessionID, CloneMessages(msgs))
}

// Delete deletes the session history file.
func (s *FileStore) Delete(_ context.Context, sessionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	p := s.filePath(sessionID)
	if err := os.Remove(p); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("react: failed to delete session file: %w", err)
	}
	return nil
}

// Sessions returns all stored session IDs.
func (s *FileStore) Sessions() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil
	}

	var out []string
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") || strings.Contains(entry.Name(), ".tmp") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(s.dir, entry.Name()))
		if err != nil {
			continue
		}
		var rec fileSessionRecord
		if err := json.Unmarshal(data, &rec); err == nil && rec.SessionID != "" {
			out = append(out, rec.SessionID)
		}
	}
	return out
}

func (s *FileStore) atomicWrite(sessionID string, msgs []Message) error {
	rec := fileSessionRecord{
		SessionID: sessionID,
		Messages:  msgs,
		UpdatedAt: time.Now(),
	}
	data, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("react: failed to marshal session: %w", err)
	}

	var randSuffix [4]byte
	_, _ = rand.Read(randSuffix[:])
	tmpPath := filepath.Join(s.dir, fmt.Sprintf("%s.tmp.%s", hex.EncodeToString([]byte(sessionID)), hex.EncodeToString(randSuffix[:])))

	if err := os.WriteFile(tmpPath, data, 0o600); err != nil {
		return fmt.Errorf("react: failed to write temp file: %w", err)
	}

	targetPath := s.filePath(sessionID)
	if err := os.Rename(tmpPath, targetPath); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("react: failed to commit session file: %w", err)
	}
	return nil
}
