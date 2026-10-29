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

	// cache holds the last known messages per session, populated on
	// Load, Append, and Replace. It lets Append skip the disk read and
	// JSON parse of the whole history on every turn; the file itself is
	// still rewritten (atomically) on every mutation, so the on-disk
	// format is unchanged and a fresh FileStore instance sees identical
	// data. The cache assumes this instance owns the directory: edits
	// made by another process or FileStore instance are invisible until
	// Invalidate is called (or a fresh instance is opened). Cached
	// slices are never handed out directly; Load always returns a clone.
	cache map[string][]Message
}

type fileSessionRecord struct {
	SessionID string    `json:"session_id"`
	Messages  []Message `json:"messages"`
	UpdatedAt time.Time `json:"updated_at"`
}

// NewFileStore creates or opens a directory-backed session store.
func NewFileStore(dir string) (*FileStore, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
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
//
// A cache hit returns a clone without touching disk. On a miss, the file
// is read and parsed exactly as before (including the parse error for a
// corrupt file), and the parsed record populates the cache.
func (s *FileStore) Load(_ context.Context, sessionID string) ([]Message, error) {
	s.mu.RLock()
	if cached, ok := s.cache[sessionID]; ok {
		out := CloneMessages(cached)
		s.mu.RUnlock()
		return out, nil
	}
	s.mu.RUnlock()

	s.mu.Lock()
	defer s.mu.Unlock()
	if cached, ok := s.cache[sessionID]; ok {
		return CloneMessages(cached), nil
	}

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
	if s.cache == nil {
		s.cache = map[string][]Message{}
	}
	// rec.Messages is fresh from Unmarshal; nothing else aliases it.
	s.cache[sessionID] = rec.Messages
	return CloneMessages(rec.Messages), nil
}

// Invalidate drops the cached record for sessionID, so the next Load or
// Append re-reads the file from disk. Call it after an external writer
// (another process or FileStore instance) has modified the session file.
func (s *FileStore) Invalidate(sessionID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.cache, sessionID)
}

// Append appends messages to sessionID.
//
// On a cache hit, the disk read and JSON parse of the whole history are
// skipped; the file is still atomically rewritten with the full history,
// so Append remains O(history) for marshal and write — the cache removes
// the read/parse half, not the write half. On a cache miss the file is
// read and parsed exactly as before, including the parse error for a
// corrupt file, and a failed write leaves the cache untouched.
func (s *FileStore) Append(ctx context.Context, sessionID string, msgs ...Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var existing []Message
	if cached, ok := s.cache[sessionID]; ok {
		existing = cached
	} else {
		p := s.filePath(sessionID)
		if data, err := os.ReadFile(p); err == nil {
			var rec fileSessionRecord
			if err := json.Unmarshal(data, &rec); err == nil {
				existing = rec.Messages
			} else {
				return fmt.Errorf("react: failed to parse session file: %w", err)
			}
		} else if !errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("react: failed to read session file: %w", err)
		}
	}

	existing = append(existing, CloneMessages(msgs)...)
	if err := s.atomicWrite(sessionID, existing); err != nil {
		return err
	}
	if s.cache == nil {
		s.cache = map[string][]Message{}
	}
	s.cache[sessionID] = existing
	return nil
}

// Replace overwrites the message history of sessionID.
func (s *FileStore) Replace(_ context.Context, sessionID string, msgs []Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	cloned := CloneMessages(msgs)
	if err := s.atomicWrite(sessionID, cloned); err != nil {
		return err
	}
	if s.cache == nil {
		s.cache = map[string][]Message{}
	}
	s.cache[sessionID] = cloned
	return nil
}

// Delete deletes the session history file.
func (s *FileStore) Delete(_ context.Context, sessionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	p := s.filePath(sessionID)
	if err := os.Remove(p); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("react: failed to delete session file: %w", err)
	}
	delete(s.cache, sessionID)
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
		hexPart := strings.TrimSuffix(entry.Name(), ".json")
		if raw, err := hex.DecodeString(hexPart); err == nil && len(raw) > 0 {
			out = append(out, string(raw))
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
