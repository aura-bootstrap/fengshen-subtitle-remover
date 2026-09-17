package ffx

import (
	"reflect"
	"testing"
)

func TestColorEncodeArgs(t *testing.T) {
	mi := &MediaInfo{
		ColorRange:     "tv",
		ColorSpace:     "bt709",
		ColorTransfer:  "bt709",
		ColorPrimaries: "bt709",
	}
	want := []string{
		"-color_range", "tv",
		"-colorspace", "bt709",
		"-color_trc", "bt709",
		"-color_primaries", "bt709",
	}
	if got := mi.ColorEncodeArgs(); !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestColorEncodeArgsSkipsUnknown(t *testing.T) {
	mi := &MediaInfo{ColorSpace: "bt2020nc", ColorTransfer: "unknown", ColorRange: ""}
	want := []string{"-colorspace", "bt2020nc"}
	if got := mi.ColorEncodeArgs(); !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
	var empty MediaInfo
	if got := empty.ColorEncodeArgs(); got != nil {
		t.Errorf("empty MediaInfo produced args: %v", got)
	}
}
