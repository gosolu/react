// Package builtin provides common built-in tools for agents:
// read_file, write_file, list_dir, and shell.
package builtin

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/gosolu/react/tool"
)

type readFileInput struct {
	Path string `json:"path" desc:"The path of the file to read"`
}

// ReadFile returns a tool to inspect file contents. Marked ReadOnly.
func ReadFile(opts ...tool.Option) tool.Tool {
	defaultOpts := []tool.Option{tool.ReadOnly()}
	defaultOpts = append(defaultOpts, opts...)
	return tool.New("read_file", "Read the entire contents of a file at the given path.", func(ctx context.Context, in readFileInput) (string, error) {
		clean := filepath.Clean(in.Path)
		data, err := os.ReadFile(clean)
		if err != nil {
			return "", err
		}
		if len(data) > 512*1024 {
			return string(data[:512*1024]) + "\n...[truncated]", nil
		}
		return string(data), nil
	}, defaultOpts...)
}

type writeFileInput struct {
	Path    string `json:"path" desc:"The path of the file to write"`
	Content string `json:"content" desc:"The content to write into the file"`
}

// WriteFile returns a tool to create or overwrite files. Marked Destructive.
func WriteFile(opts ...tool.Option) tool.Tool {
	defaultOpts := []tool.Option{tool.Destructive()}
	defaultOpts = append(defaultOpts, opts...)
	return tool.New("write_file", "Write text content to a file. Overwrites existing files and creates parent directories if needed.", func(ctx context.Context, in writeFileInput) (string, error) {
		clean := filepath.Clean(in.Path)
		if err := os.MkdirAll(filepath.Dir(clean), 0755); err != nil {
			return "", err
		}
		if err := os.WriteFile(clean, []byte(in.Content), 0644); err != nil {
			return "", err
		}
		return fmt.Sprintf("Successfully wrote %d bytes to %s", len(in.Content), clean), nil
	}, defaultOpts...)
}

type listDirInput struct {
	Path string `json:"path" desc:"The directory path to list"`
}

// ListDir returns a tool to list directory entries. Marked ReadOnly.
func ListDir(opts ...tool.Option) tool.Tool {
	defaultOpts := []tool.Option{tool.ReadOnly()}
	defaultOpts = append(defaultOpts, opts...)
	return tool.New("list_dir", "List files and subdirectories in a directory path.", func(ctx context.Context, in listDirInput) (string, error) {
		p := in.Path
		if p == "" {
			p = "."
		}
		entries, err := os.ReadDir(filepath.Clean(p))
		if err != nil {
			return "", err
		}
		var lines []string
		for _, e := range entries {
			kind := "file"
			if e.IsDir() {
				kind = "dir"
			}
			lines = append(lines, fmt.Sprintf("%-5s %s", kind, e.Name()))
		}
		if len(lines) == 0 {
			return "(empty directory)", nil
		}
		return strings.Join(lines, "\n"), nil
	}, defaultOpts...)
}

type shellInput struct {
	Command string `json:"command" desc:"The shell command to execute"`
}

// Shell returns a tool to execute arbitrary bash/sh commands. Marked Destructive and Sequential.
func Shell(opts ...tool.Option) tool.Tool {
	defaultOpts := []tool.Option{tool.Destructive(), tool.Sequential()}
	defaultOpts = append(defaultOpts, opts...)
	return tool.New("shell", "Execute a command in the system shell. Returns stdout and stderr.", func(ctx context.Context, in shellInput) (string, error) {
		if strings.TrimSpace(in.Command) == "" {
			return "", fmt.Errorf("command cannot be empty")
		}
		// Run with bash or sh
		shell := "bash"
		if _, err := exec.LookPath("bash"); err != nil {
			shell = "sh"
		}

		cctx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()

		cmd := exec.CommandContext(cctx, shell, "-c", in.Command)
		var outBuf, errBuf bytes.Buffer
		cmd.Stdout = &outBuf
		cmd.Stderr = &errBuf

		runErr := cmd.Run()
		out := outBuf.String()
		errStr := errBuf.String()

		var res strings.Builder
		if out != "" {
			res.WriteString(out)
		}
		if errStr != "" {
			if res.Len() > 0 {
				res.WriteString("\n[stderr]:\n")
			}
			res.WriteString(errStr)
		}

		if runErr != nil {
			if res.Len() > 0 {
				return res.String(), fmt.Errorf("exit status: %w", runErr)
			}
			return fmt.Sprintf("failed to execute: %v", runErr), runErr
		}
		if res.Len() == 0 {
			return "(command executed with no output)", nil
		}
		return res.String(), nil
	}, defaultOpts...)
}

// All returns a slice of all standard built-in tools.
func All() []tool.Tool {
	return []tool.Tool{
		ReadFile(),
		WriteFile(),
		ListDir(),
		Shell(),
	}
}
