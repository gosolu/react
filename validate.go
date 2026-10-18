package react

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Validate checks the agent's in-memory configuration. It does not inspect
// prompts, files, or provider responses.
//
// New calls Validate, so invalid limits or tool schemas fail before any model
// call. Validate is idempotent and safe for concurrent use.
func (a *Agent) Validate() error {
	if a == nil {
		return errors.New("agent: nil")
	}
	var errs []error

	if a.cfg.maxTurns <= 0 {
		errs = append(errs, fmt.Errorf("agent: max turns must be > 0, got %d", a.cfg.maxTurns))
	}
	if a.cfg.maxParallel < 0 {
		errs = append(errs, fmt.Errorf("agent: max parallel tools must be >= 0, got %d", a.cfg.maxParallel))
	}
	if a.cfg.maxDepth < 0 {
		errs = append(errs, fmt.Errorf("agent: max depth must be >= 0, got %d", a.cfg.maxDepth))
	}
	if a.cfg.maxTokens < 0 {
		errs = append(errs, fmt.Errorf("agent: max tokens must be >= 0, got %d", a.cfg.maxTokens))
	}
	if a.cfg.maxToolOutput < 0 {
		errs = append(errs, fmt.Errorf("agent: max tool output must be >= 0, got %d", a.cfg.maxToolOutput))
	}
	if a.cfg.contextBudget < 0 {
		errs = append(errs, fmt.Errorf("agent: context budget must be >= 0, got %d", a.cfg.contextBudget))
	}
	if err := validateDuration("tool timeout", a.cfg.toolTimeout); err != nil {
		errs = append(errs, err)
	}
	if err := validateDuration("model timeout", a.cfg.modelTimeout); err != nil {
		errs = append(errs, err)
	}
	if err := validateBudget(a.cfg.budget); err != nil {
		errs = append(errs, err)
	}
	if a.skills != nil {
		if err := a.skills.Validate(); err != nil {
			errs = append(errs, err)
		}
	}

	var specs []ToolSpec
	if a.registry != nil {
		specs = a.registry.Specs()
	}
	for _, spec := range specs {
		if spec.Name == "" {
			errs = append(errs, errors.New("agent: tool with empty name"))
			continue
		}
		if len(spec.InputSchema) == 0 {
			errs = append(errs, fmt.Errorf("agent: tool %q has no input schema", spec.Name))
			continue
		}
		if !json.Valid(spec.InputSchema) {
			errs = append(errs, fmt.Errorf("agent: tool %q has an invalid JSON input schema", spec.Name))
		}
		if spec.MaxOutput < 0 {
			errs = append(errs, fmt.Errorf("agent: tool %q has a negative max output", spec.Name))
		}
	}
	return errors.Join(errs...)
}

func validateDuration(name string, d time.Duration) error {
	if d < 0 {
		return fmt.Errorf("agent: %s must be >= 0, got %s", name, d)
	}
	return nil
}

func validateBudget(b BudgetLimits) error {
	var errs []error
	if b.MaxModelCalls < 0 {
		errs = append(errs, fmt.Errorf("agent: budget max model calls must be >= 0, got %d", b.MaxModelCalls))
	}
	if b.MaxToolCalls < 0 {
		errs = append(errs, fmt.Errorf("agent: budget max tool calls must be >= 0, got %d", b.MaxToolCalls))
	}
	if b.MaxRequestTokens < 0 {
		errs = append(errs, fmt.Errorf("agent: budget max request tokens must be >= 0, got %d", b.MaxRequestTokens))
	}
	if b.MaxTotalTokens < 0 {
		errs = append(errs, fmt.Errorf("agent: budget max total tokens must be >= 0, got %d", b.MaxTotalTokens))
	}
	if b.MaxWallTime < 0 {
		errs = append(errs, fmt.Errorf("agent: budget max wall time must be >= 0, got %s", b.MaxWallTime))
	}
	return errors.Join(errs...)
}
