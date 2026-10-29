// Package tools provides the small shell/file toolset for the react CLI
// and for embedders that want the same tools: read_file, write_file, and
// exec. The library itself intentionally ships no shell/file tools; this
// package is the opt-in layer, wired into cmd/react behind the --tools
// flag (on by default).
//
// Safety is left to react's permission system, not duplicated here:
// read_file is ReadOnly (auto-allowed under the default policy), while
// write_file and exec are Destructive (approval-gated). Callers that
// embed these tools should keep a permission policy and approver in
// place; registering them on an agent with no approver denies
// destructive calls under the default policy.
package tools

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/gosolu/react"
)

// All returns the full toolset: read_file, write_file, exec.
func All() []react.Tool {
	return []react.Tool{ReadFileTool(), WriteFileTool(), ExecTool()}
}

type readFileInput struct {
	Path string `json:"path" desc:"File to read, as a filesystem path."`
}

// ReadFileTool returns a ReadOnly tool that reads a file as text.
func ReadFileTool() react.Tool {
	return react.NewTool("read_file", "Read a file and return its contents as text.", func(ctx context.Context, in readFileInput) (string, error) {
		if strings.TrimSpace(in.Path) == "" {
			return "", fmt.Errorf("read_file: path is required")
		}
		data, err := os.ReadFile(in.Path)
		if err != nil {
			return "", err
		}
		return string(data), nil
	}, react.ReadOnly())
}

type writeFileInput struct {
	Path    string `json:"path" desc:"File to write, as a filesystem path."`
	Content string `json:"content" desc:"Full contents to write to the file."`
}

// WriteFileTool returns a Destructive tool that overwrites a file with
// the given contents (creating it with 0644 if it does not exist).
func WriteFileTool() react.Tool {
	return react.NewTool("write_file", "Write contents to a file, overwriting it if it exists.", func(ctx context.Context, in writeFileInput) (string, error) {
		if strings.TrimSpace(in.Path) == "" {
			return "", fmt.Errorf("write_file: path is required")
		}
		if err := os.WriteFile(in.Path, []byte(in.Content), 0o644); err != nil {
			return "", err
		}
		return fmt.Sprintf("wrote %d bytes to %s", len(in.Content), in.Path), nil
	}, react.Destructive())
}

type execInput struct {
	Command string `json:"command" desc:"Shell command to run."`
	WorkDir string `json:"work_dir,omitempty" desc:"Working directory for the command; defaults to the process working directory."`
}

// ExecTool returns a Destructive, Sequential tool that runs a shell
// command and returns its combined stdout/stderr. A non-zero exit is
// reported as an error whose message carries the (truncated) output, so
// the model sees what the command printed before failing.
func ExecTool() react.Tool {
	return react.NewTool("exec", "Run a shell command and return its combined output.", func(ctx context.Context, in execInput) (string, error) {
		if strings.TrimSpace(in.Command) == "" {
			return "", fmt.Errorf("exec: command is required")
		}
		var cmd *exec.Cmd
		if runtime.GOOS == "windows" {
			cmd = exec.CommandContext(ctx, "cmd", "/C", in.Command)
		} else {
			cmd = exec.CommandContext(ctx, "sh", "-c", in.Command)
		}
		if in.WorkDir != "" {
			cmd.Dir = in.WorkDir
		}
		out, err := cmd.CombinedOutput()
		if err != nil {
			const maxErrOutput = 4096
			shown := string(out)
			if len(shown) > maxErrOutput {
				shown = shown[:maxErrOutput] + "\n...[output truncated]"
			}
			return "", fmt.Errorf("exec: command failed: %w\noutput:\n%s", err, shown)
		}
		return string(out), nil
	}, react.Destructive(), react.Sequential())
}
