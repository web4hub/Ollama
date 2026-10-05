package renderers

import "encoding/json"

// marshalWithSpaces marshals v to JSON and adds a space after each ':' and ','
// that appears outside of string values. This matches the formatting expected
// by certain model architectures.
func marshalWithSpaces(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return addJSONSpaces(b), nil
}

// marshalWithSpacesNoHTMLEscape is marshalWithSpaces for templates rendered
// with transformers' tojson (Python's json.dumps), which writes '&', '<', '>',
// U+2028 and U+2029 unescaped.
func marshalWithSpacesNoHTMLEscape(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return addJSONSpaces(unescapeJSONHTML(b)), nil
}

// unescapeJSONHTML reverses json.HTMLEscape. Encoder.SetEscapeHTML(false) is
// not enough because nested json.Marshaler implementations, such as the
// ordered maps behind tool properties and call arguments, escape on their own.
func unescapeJSONHTML(b []byte) []byte {
	out := make([]byte, 0, len(b))
	for i := 0; i < len(b); i++ {
		if b[i] != '\\' || i+1 == len(b) {
			out = append(out, b[i])
			continue
		}
		if b[i+1] == 'u' && i+6 <= len(b) {
			switch string(b[i+2 : i+6]) {
			case "0026":
				out = append(out, '&')
			case "003c":
				out = append(out, '<')
			case "003e":
				out = append(out, '>')
			case "2028":
				out = append(out, "\u2028"...)
			case "2029":
				out = append(out, "\u2029"...)
			default:
				out = append(out, b[i:i+6]...)
			}
			i += 5
			continue
		}
		// Copy other escapes whole so an escaped backslash is not misread.
		out = append(out, b[i], b[i+1])
		i++
	}
	return out
}

func addJSONSpaces(b []byte) []byte {
	out := make([]byte, 0, len(b)+len(b)/8)
	inStr, esc := false, false
	for _, c := range b {
		if inStr {
			out = append(out, c)
			if esc {
				esc = false
				continue
			}
			if c == '\\' {
				esc = true
				continue
			}
			if c == '"' {
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
			out = append(out, c)
		case ':':
			out = append(out, ':', ' ')
		case ',':
			out = append(out, ',', ' ')
		default:
			out = append(out, c)
		}
	}
	return out
}
