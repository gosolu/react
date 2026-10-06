package memory_test

import (
	"testing"

	"github.com/gosolu/react/memory"
	"github.com/gosolu/react/memory/storetest"
)

func TestInMemory(t *testing.T) {
	storetest.Run(t, func() memory.Store { return memory.NewInMemory() })
}
