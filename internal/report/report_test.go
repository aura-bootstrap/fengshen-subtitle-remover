package report

import (
	"strings"
	"testing"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/manifest"
)

func TestTimecode(t *testing.T) {
	cases := []struct {
		sec  float64
		want string
	}{
		{0, "00:00:00.000"},
		{1.5, "00:00:01.500"},
		{61.04, "00:01:01.040"},
		{3661.001, "01:01:01.001"},
		{3599.9996, "01:00:00.000"}, // rounds up across the hour boundary
	}
	for _, c := range cases {
		if got := Timecode(c.sec); got != c.want {
			t.Errorf("Timecode(%v) = %s, want %s", c.sec, got, c.want)
		}
	}
}

func seg(index, startF, endF int, cov float64, risk bool, reason ...string) manifest.Segment {
	return manifest.Segment{
		Index: index, StartF: startF, EndF: endF,
		Start: float64(startF) / 25, End: float64(endF+1) / 25,
		Coverage: cov, Risk: risk, Reason: reason,
	}
}

func TestRiskListThresholds(t *testing.T) {
	m := &manifest.Manifest{Segments: []manifest.Segment{
		seg(0, 0, 49, 0.9, false),                     // fine
		seg(1, 50, 99, 0.2, false),                    // low coverage only
		seg(2, 100, 149, 0.9, true, "tier T5 hazard"), // flagged by routing
		seg(3, 150, 199, 0.1, true, "tier T5 hazard"), // both
	}}
	items := RiskList(m, nil, 0.35)
	if len(items) != 3 {
		t.Fatalf("got %d items, want 3", len(items))
	}
	if items[0].StartF != 50 || items[1].StartF != 100 || items[2].StartF != 150 {
		t.Fatalf("not sorted by start frame: %+v", items)
	}
	if items[0].Reason[len(items[0].Reason)-1] != "coverage 0.200 < 0.350" {
		t.Errorf("low-coverage reason missing: %v", items[0].Reason)
	}
	if items[1].Reason[0] != "tier T5 hazard" {
		t.Errorf("routing reason dropped: %v", items[1].Reason)
	}
	if len(items[2].Reason) != 2 {
		t.Errorf("segment with both causes should carry both reasons: %v", items[2].Reason)
	}
	if items[0].Timecode[0] != "00:00:02.000" || items[0].Timecode[1] != "00:00:04.000" {
		t.Errorf("timecodes wrong: %v", items[0].Timecode)
	}
}

func TestRiskListMergeResidues(t *testing.T) {
	res := []Residue{
		{Frame: 11, Time: 0.44, Box: imgx.Rect{X: 10, Y: 300, W: 40, H: 20}},
		{Frame: 10, Time: 0.40, Box: imgx.Rect{X: 12, Y: 302, W: 40, H: 20}}, // unsorted input
		{Frame: 12, Time: 0.48, Box: imgx.Rect{X: 11, Y: 301, W: 42, H: 20}},
		{Frame: 30, Time: 1.20, Box: imgx.Rect{X: 400, Y: 300, W: 40, H: 20}}, // disjoint
		{Frame: 31, Time: 1.24, Box: imgx.Rect{X: 405, Y: 300, W: 40, H: 20}},
	}
	items := RiskList(&manifest.Manifest{}, res, 0.35)
	if len(items) != 2 {
		t.Fatalf("got %d items, want 2 merged spans", len(items))
	}
	a := items[0]
	if a.StartF != 10 || a.EndF != 12 {
		t.Errorf("first span = %d-%d, want 10-12", a.StartF, a.EndF)
	}
	if a.Box.X != 10 || a.Box.Y != 300 || a.Box.W != 43 || a.Box.H != 22 {
		t.Errorf("union box = %+v", a.Box)
	}
	if a.Reason[0] != "residue" {
		t.Errorf("reason = %v", a.Reason)
	}
	b := items[1]
	if b.StartF != 30 || b.EndF != 31 || b.Timecode[0] != "00:00:01.200" {
		t.Errorf("second span wrong: %+v", b)
	}
}

func TestWriteTimecodes(t *testing.T) {
	items := []RiskItem{{
		StartF: 25, EndF: 49, Start: 1.0, End: 2.0,
		Timecode: [2]string{Timecode(1.0), Timecode(2.0)},
		Box:      imgx.Rect{X: 5, Y: 300, W: 100, H: 30},
		Coverage: 0.123,
		Reason:   []string{"coverage 0.123 < 0.350"},
	}}
	var sb strings.Builder
	if err := WriteTimecodes(&sb, items); err != nil {
		t.Fatal(err)
	}
	want := "00:00:01.000 - 00:00:02.000  frames 25-49  box 5,300 100x30  coverage 0.123  coverage 0.123 < 0.350\n"
	if sb.String() != want {
		t.Errorf("got %q, want %q", sb.String(), want)
	}
}
