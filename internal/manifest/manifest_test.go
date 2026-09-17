package manifest

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ffx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/mask"
)

func testManifest(t *testing.T, dir string) (*Manifest, string) {
	t.Helper()
	input := filepath.Join(dir, "in.mp4")
	if err := os.WriteFile(input, []byte("fake video bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	mt := time.Date(2026, 9, 17, 12, 0, 0, 0, time.UTC)
	if err := os.Chtimes(input, mt, mt); err != nil {
		t.Fatal(err)
	}
	info := &ffx.MediaInfo{W: 640, H: 360, FPS: 25, Duration: 4.0}
	fp, err := FingerprintInput(input, info)
	if err != nil {
		t.Fatal(err)
	}
	return &Manifest{
		Version: Version, Input: fp,
		Params: map[string]any{"crf": 17, "alpha": true},
		Band:   Band{Y: 208, H: 152}, CharH: 14,
		Segments: []Segment{{
			Index: 0, StartF: 10, EndF: 19, Start: 0.4, End: 0.8,
			Masks: []mask.Frame{{RLE: []uint16{5, 3, 9}}, {RLE: []uint16{6, 2, 11}}},
			Tier:  5, Engine: "propainter", Coverage: 0.05, Risk: true,
			Reason: []string{"persistent occlusion"},
		}},
	}, input
}

// Write→Read round-trips every field.
func TestWriteReadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	m, _ := testManifest(t, dir)
	p := filepath.Join(dir, "m.json")
	if err := Write(p, m); err != nil {
		t.Fatal(err)
	}
	got, err := Read(p)
	if err != nil {
		t.Fatal(err)
	}
	if got.Version != Version || got.CharH != 14 || got.Band.Y != 208 {
		t.Fatalf("header = %+v", got)
	}
	if got.Input.Size != m.Input.Size || !got.Input.ModTime.Equal(m.Input.ModTime) {
		t.Fatalf("fingerprint = %+v", got.Input)
	}
	if len(got.Segments) != 1 {
		t.Fatalf("%d segments", len(got.Segments))
	}
	s := got.Segments[0]
	if s.StartF != 10 || s.EndF != 19 || len(s.Masks) != 2 || s.Masks[1].RLE[2] != 11 {
		t.Fatalf("segment = %+v", s)
	}
	if s.Tier != 5 || s.Engine != "propainter" || !s.Risk || s.Reason[0] != "persistent occlusion" {
		t.Fatalf("segment routing = %+v", s)
	}
}

// Verify accepts the recorded input and rejects any drift (R8.4).
func TestVerify(t *testing.T) {
	dir := t.TempDir()
	m, input := testManifest(t, dir)
	info := &ffx.MediaInfo{W: 640, H: 360, FPS: 25, Duration: 4.0}
	if err := m.Verify(input, info); err != nil {
		t.Fatalf("same input rejected: %v", err)
	}
	if err := os.WriteFile(input, []byte("fake video bytes!!"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := m.Verify(input, info); err == nil {
		t.Fatal("size drift accepted")
	}
	m2, input2 := testManifest(t, dir)
	m2.Input.Frames = 101
	if err := m2.Verify(input2, info); err == nil {
		t.Fatal("frame-count drift accepted")
	}
}

// Read rejects a foreign version.
func TestReadVersion(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "m.json")
	if err := os.WriteFile(p, []byte(`{"version": 99}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(p); err == nil {
		t.Fatal("version 99 accepted")
	}
}

// SegmentOf resolves indices and reports unknown ones.
func TestSegmentOf(t *testing.T) {
	dir := t.TempDir()
	m, _ := testManifest(t, dir)
	if s, err := m.SegmentOf(0); err != nil || s.StartF != 10 {
		t.Fatalf("SegmentOf(0) = %v, %v", s, err)
	}
	if _, err := m.SegmentOf(7); err == nil {
		t.Fatal("unknown segment accepted")
	}
}
