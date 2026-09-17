// Package manifest records a removal run so individual segments can be
// re-run later without redoing the whole feature (R8): input fingerprint,
// global parameters, and per-event frame ranges with their masks, routing
// verdicts and coverage.
package manifest

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/events"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ffx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/mask"
)

const Version = 1

// Fingerprint identifies the exact input a manifest was produced from.
type Fingerprint struct {
	Path     string    `json:"path"`
	Size     int64     `json:"size"`
	ModTime  time.Time `json:"mtime"`
	Width    int       `json:"width"`
	Height   int       `json:"height"`
	FPS      float64   `json:"fps"`
	Duration float64   `json:"duration"`
	Frames   int       `json:"frames"`
}

// Band mirrors the subtitle band geometry (kept flat to avoid a subs import).
type Band struct {
	Y int `json:"y"`
	H int `json:"h"`
}

// Segment is one repairable event with everything a rerun needs.
type Segment struct {
	Index    int          `json:"index"`
	StartF   int          `json:"start_f"`
	EndF     int          `json:"end_f"`
	Start    float64      `json:"start"`
	End      float64      `json:"end"`
	Box      imgx.Rect    `json:"box"`
	Masks    []mask.Frame `json:"masks"` // band coordinates, one per frame in [StartF, EndF]
	Tier     int          `json:"tier"`
	Engine   string       `json:"engine"`
	Coverage float64      `json:"coverage"`
	Risk     bool         `json:"risk,omitempty"`
	Reason   []string     `json:"reason,omitempty"`
}

// Rerun records one segment re-run (R8.5).
type Rerun struct {
	Segment int            `json:"segment"`
	Time    time.Time      `json:"time"`
	Params  map[string]any `json:"params,omitempty"`
}

// Manifest is the full record of one removal run.
type Manifest struct {
	Version  int            `json:"version"`
	Input    Fingerprint    `json:"input"`
	Params   map[string]any `json:"params"`
	Band     Band           `json:"band"`
	CharH    int            `json:"char_h"`
	Segments []Segment      `json:"segments"`
	Reruns   []Rerun        `json:"reruns,omitempty"`
}

// FingerprintInput probes the input and stats the file.
func FingerprintInput(path string, info *ffx.MediaInfo) (Fingerprint, error) {
	st, err := os.Stat(path)
	if err != nil {
		return Fingerprint{}, err
	}
	return Fingerprint{
		Path: path, Size: st.Size(), ModTime: st.ModTime().UTC().Truncate(time.Second),
		Width: info.W, Height: info.H, FPS: info.FPS,
		Duration: info.Duration, Frames: int(info.Duration*info.FPS + 0.5),
	}, nil
}

// Write serializes the manifest (atomically via a temp file).
func Write(path string, m *Manifest) error {
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Read loads a manifest and checks its version.
func Read(path string) (*Manifest, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("manifest: %v", err)
	}
	if m.Version != Version {
		return nil, fmt.Errorf("manifest: version %d, want %d", m.Version, Version)
	}
	return &m, nil
}

// Verify refuses reruns when the current input no longer matches the
// recorded fingerprint (R8.4).
func (m *Manifest) Verify(input string, info *ffx.MediaInfo) error {
	fp, err := FingerprintInput(input, info)
	if err != nil {
		return err
	}
	want, got := m.Input, fp
	if want.Size != got.Size {
		return fmt.Errorf("manifest: input size %d, want %d", got.Size, want.Size)
	}
	if !want.ModTime.Equal(got.ModTime) {
		return fmt.Errorf("manifest: input mtime %s, want %s", got.ModTime, want.ModTime)
	}
	if want.Width != got.Width || want.Height != got.Height {
		return fmt.Errorf("manifest: input %dx%d, want %dx%d", got.Width, got.Height, want.Width, want.Height)
	}
	if want.Frames != got.Frames {
		return fmt.Errorf("manifest: input %d frames, want %d", got.Frames, want.Frames)
	}
	return nil
}

// SegmentOf finds a segment by index.
func (m *Manifest) SegmentOf(index int) (*Segment, error) {
	for i := range m.Segments {
		if m.Segments[i].Index == index {
			return &m.Segments[i], nil
		}
	}
	return nil, fmt.Errorf("manifest: no segment %d", index)
}

// NewSegment builds a segment from pipeline outputs; masks is the full-run
// mask sequence in band coordinates.
func NewSegment(index int, ev events.Event, masks []mask.Frame, tier int, engineName string, coverage float64, risk bool, reason []string) Segment {
	end := ev.EndF
	if end >= len(masks) {
		end = len(masks) - 1
	}
	seg := Segment{
		Index: index, StartF: ev.StartF, EndF: end,
		Start: ev.Start, End: ev.End, Box: ev.Box,
		Tier: tier, Engine: engineName, Coverage: coverage,
		Risk: risk, Reason: reason,
	}
	if end >= ev.StartF && ev.StartF >= 0 {
		seg.Masks = masks[ev.StartF : end+1]
	}
	return seg
}
