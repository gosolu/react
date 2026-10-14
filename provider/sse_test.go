package provider

import (
	"strings"
	"testing"
)

func TestReadSSE(t *testing.T) {
	in := ": comment\nevent: a\ndata: one\ndata: two\n\ndata: {\"x\":1}\n\ndata: [DONE]\n"
	var got []SSEEvent
	for ev, err := range ReadSSE(strings.NewReader(in)) {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, ev)
	}
	if len(got) != 3 || got[0].Event != "a" || got[0].Data != "one\ntwo" || got[1].Data != `{"x":1}` || got[2].Data != "[DONE]" {
		t.Fatalf("%+v", got)
	}
}
