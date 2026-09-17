// Package report builds the high-risk segment list (R6): events whose fill
// was mostly synthesized (low true-pixel coverage) or that routing already
// flagged, plus verification residues, exported as JSON or HH:MM:SS.mmm
// timecode text for review.
package report

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/manifest"
)

// DefaultCoverage is the true-pixel coverage below which an event is flagged
// high-risk when the user does not set --risk-coverage (R6.1).
const DefaultCoverage = 0.35

// RiskItem is one flagged span for reviewer attention (R6.3).
type RiskItem struct {
	StartF   int       `json:"start_f"`
	EndF     int       `json:"end_f"`
	Start    float64   `json:"start"`
	End      float64   `json:"end"`
	Timecode [2]string `json:"timecode"`
	Box      imgx.Rect `json:"box"`
	Coverage float64   `json:"coverage"`
	Reason   []string  `json:"reason"`
}

// Residue is one verification re-detection overlapping the original subtitle
// region. Defined locally (mirroring pipeline.ResidueBox) so pipeline can
// depend on report without an import cycle.
type Residue struct {
	Frame int
	Time  float64
	Box   imgx.Rect
}

// Timecode renders seconds as HH:MM:SS.mmm (R6.4).
func Timecode(sec float64) string {
	if sec < 0 {
		sec = 0
	}
	ms := int64(sec*1000 + 0.5)
	h := ms / 3600000
	ms %= 3600000
	m := ms / 60000
	ms %= 60000
	s := ms / 1000
	ms %= 1000
	return fmt.Sprintf("%02d:%02d:%02d.%03d", h, m, s, ms)
}

// RiskList merges two sources into one sorted list: manifest segments that
// routing flagged or whose coverage falls below covThr (R6.1, R6.2), and
// verification residues merged into frame spans (R6.5).
func RiskList(m *manifest.Manifest, residues []Residue, covThr float64) []RiskItem {
	var out []RiskItem
	for _, seg := range m.Segments {
		reason := append([]string(nil), seg.Reason...)
		low := seg.Coverage < covThr
		if low {
			reason = append(reason, fmt.Sprintf("coverage %.3f < %.3f", seg.Coverage, covThr))
		}
		if !seg.Risk && !low {
			continue
		}
		out = append(out, RiskItem{
			StartF: seg.StartF, EndF: seg.EndF,
			Start: seg.Start, End: seg.End,
			Timecode: [2]string{Timecode(seg.Start), Timecode(seg.End)},
			Box:      seg.Box,
			Coverage: seg.Coverage,
			Reason:   reason,
		})
	}
	out = append(out, mergeResidues(residues)...)
	sort.Slice(out, func(i, j int) bool { return out[i].StartF < out[j].StartF })
	return out
}

// mergeResidues folds consecutive-frame residues with overlapping boxes into
// single spans so a lingering stroke produces one entry, not one per frame.
func mergeResidues(residues []Residue) []RiskItem {
	if len(residues) == 0 {
		return nil
	}
	rs := append([]Residue(nil), residues...)
	sort.Slice(rs, func(i, j int) bool { return rs[i].Frame < rs[j].Frame })
	var out []RiskItem
	cur := rs[0]
	span := RiskItem{
		StartF: cur.Frame, EndF: cur.Frame,
		Start: cur.Time, End: cur.Time,
		Box:    cur.Box,
		Reason: []string{"residue"},
	}
	flush := func() {
		span.Timecode = [2]string{Timecode(span.Start), Timecode(span.End)}
		out = append(out, span)
	}
	for _, r := range rs[1:] {
		if r.Frame <= span.EndF+1 && overlap(span.Box, r.Box) {
			span.EndF = r.Frame
			span.End = r.Time
			span.Box = union(span.Box, r.Box)
			continue
		}
		flush()
		span = RiskItem{
			StartF: r.Frame, EndF: r.Frame,
			Start: r.Time, End: r.Time,
			Box:    r.Box,
			Reason: []string{"residue"},
		}
	}
	flush()
	return out
}

func overlap(a, b imgx.Rect) bool {
	return a.X < b.X+b.W && b.X < a.X+a.W && a.Y < b.Y+b.H && b.Y < a.Y+a.H
}

func union(a, b imgx.Rect) imgx.Rect {
	x0, y0 := min(a.X, b.X), min(a.Y, b.Y)
	x1 := max(a.X+a.W, b.X+b.W)
	y1 := max(a.Y+a.H, b.Y+b.H)
	return imgx.Rect{X: x0, Y: y0, W: x1 - x0, H: y1 - y0}
}

// WriteTimecodes exports the list as a readable HH:MM:SS.mmm text (R6.4).
func WriteTimecodes(w io.Writer, items []RiskItem) error {
	for _, it := range items {
		if _, err := fmt.Fprintf(w, "%s - %s  frames %d-%d  box %d,%d %dx%d  coverage %.3f  %s\n",
			it.Timecode[0], it.Timecode[1], it.StartF, it.EndF,
			it.Box.X, it.Box.Y, it.Box.W, it.Box.H,
			it.Coverage, strings.Join(it.Reason, ", ")); err != nil {
			return err
		}
	}
	return nil
}
