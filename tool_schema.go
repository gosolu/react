package react

import (
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// SchemaFor returns the JSON Schema for T.
//
// Supported struct tags:
//
//	json:"name,omitempty"  property name; omitempty or a pointer type makes it optional
//	desc:"..."             property description
//	enum:"a,b,c"           allowed values
//	required:"true|false"  override the inferred required-ness
func SchemaFor[T any]() json.RawMessage {
	return Schema(reflect.TypeFor[T]())
}

// Schema returns the JSON Schema for t.
func Schema(t reflect.Type) json.RawMessage {
	s := schemaOf(t, map[reflect.Type]bool{})
	if s["type"] == nil && t.Kind() != reflect.Interface {
		s["type"] = "object"
	}
	b, err := json.Marshal(s)
	if err != nil {
		panic(err) // only plain maps/strings are marshalled; cannot fail
	}
	return b
}

var (
	timeType = reflect.TypeFor[time.Time]()
	rawType  = reflect.TypeFor[json.RawMessage]()
)

func schemaOf(t reflect.Type, visiting map[reflect.Type]bool) map[string]any {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t {
	case timeType:
		return map[string]any{"type": "string", "format": "date-time"}
	case rawType:
		return map[string]any{}
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		if t.Elem().Kind() == reflect.Uint8 {
			return map[string]any{"type": "string", "contentEncoding": "base64"}
		}
		return map[string]any{"type": "array", "items": schemaOf(t.Elem(), visiting)}
	case reflect.Map:
		return map[string]any{"type": "object", "additionalProperties": schemaOf(t.Elem(), visiting)}
	case reflect.Struct:
		if visiting[t] {
			return map[string]any{"type": "object"} // recursive type: stop
		}
		visiting[t] = true
		defer delete(visiting, t)
		props := map[string]any{}
		var required []string
		addStructFields(t, visiting, props, &required)
		s := map[string]any{"type": "object", "properties": props}
		if len(required) > 0 {
			s["required"] = required
		}
		return s
	default: // interface, any, etc.
		return map[string]any{}
	}
}

func addStructFields(t reflect.Type, visiting map[reflect.Type]bool, props map[string]any, required *[]string) {
	for i := range t.NumField() {
		f := t.Field(i)
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}
		name, opts, _ := strings.Cut(tag, ",")
		if f.Anonymous && name == "" {
			ft := f.Type
			if ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct {
				addStructFields(ft, visiting, props, required)
				continue
			}
		}
		if !f.IsExported() {
			continue
		}
		if name == "" {
			name = f.Name
		}
		ps := schemaOf(f.Type, visiting)
		if d := f.Tag.Get("desc"); d != "" {
			ps["description"] = d
		}
		if e := f.Tag.Get("enum"); e != "" {
			ps["enum"] = enumValues(e, ps["type"])
		}
		props[name] = ps

		req := f.Type.Kind() != reflect.Pointer && !strings.Contains(opts, "omitempty")
		switch f.Tag.Get("required") {
		case "true":
			req = true
		case "false":
			req = false
		}
		if req {
			*required = append(*required, name)
		}
	}
}

func enumValues(spec string, typ any) []any {
	var out []any
	for _, v := range strings.Split(spec, ",") {
		v = strings.TrimSpace(v)
		switch typ {
		case "integer":
			if n, err := strconv.ParseInt(v, 10, 64); err == nil {
				out = append(out, n)
				continue
			}
		case "number":
			if n, err := strconv.ParseFloat(v, 64); err == nil {
				out = append(out, n)
				continue
			}
		}
		out = append(out, v)
	}
	return out
}
