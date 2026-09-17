// Package events aggregates per-frame detections into subtitle events
// (start/end + union box), tolerating short detection gaps and forcing
// splits at scene cuts.
package events

import "github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"

type Frame struct {
	Boxes []imgx.Rect
}

type Event struct {
	StartF int       `json:"start_f"`
	EndF   int       `json:"end_f"`
	Start  float64   `json:"start"`
	End    float64   `json:"end"`
	Box    imgx.Rect `json:"box"`
	Frames int       `json:"frames"`
}

func unionBox(rs []imgx.Rect) imgx.Rect {
	var u imgx.Rect
	for _, r := range rs {
		u = u.Union(r)
	}
	return u
}

// robustUnion unions the per-frame boxes after dropping outliers. The
// dominant cluster is found by bucketing box y-centres (bucket height charH);
// a box stays only if its bucket is at most four buckets from the modal one
// and holds at least a third of its count. Four buckets ≈ two line pitches:
// charH is an estimate that runs small on landscape footage, and a genuine
// second subtitle line must survive (it does) while label/prop text detected
// elsewhere in the band is still rejected by the count and height gates.
func robustUnion(rs []imgx.Rect, charH int) imgx.Rect {
	if len(rs) <= 2 {
		return unionBox(rs)
	}
	if charH <= 0 {
		charH = 24
	}
	cy := func(r imgx.Rect) int { return 2*r.Y + r.H }
	counts := map[int]int{}
	bestK, bestN := 0, -1
	for _, r := range rs {
		k := cy(r) / charH
		counts[k]++
		if counts[k] > bestN {
			bestN, bestK = counts[k], k
		}
	}
	minN := (bestN + 2) / 3
	var kept []imgx.Rect
	for _, r := range rs {
		k := cy(r) / charH
		if d := k - bestK; d >= -4 && d <= 4 && counts[k] >= minN && r.H <= 3*charH {
			kept = append(kept, r)
		}
	}
	if len(kept) == 0 {
		return unionBox(rs)
	}
	return unionBox(kept)
}

func Build(frames []Frame, fps float64, maxGap int, cuts []float64, charH int) []Event {
	if fps <= 0 {
		fps = 25
	}
	cutFrame := make(map[int]bool, len(cuts))
	for _, c := range cuts {
		if f := int(c*fps + 0.5); f > 0 {
			cutFrame[f] = true
		}
	}
	var evs []Event
	n := len(frames)
	i := 0
	for i < n {
		if len(frames[i].Boxes) == 0 {
			i++
			continue
		}
		start, last := i, i
		fb := []imgx.Rect{frames[i].Boxes[0]}
		if b := frames[i].Boxes; len(b) > 1 {
			fb = b
		}
		j := i + 1
		for j < n {
			if len(frames[j].Boxes) > 0 {
				fb = append(fb, frames[j].Boxes...)
				last = j
				j++
				continue
			}
			if j-last <= maxGap && !cutFrame[j] {
				j++
				continue
			}
			break
		}
		evs = append(evs, Event{
			StartF: start,
			EndF:   last,
			Start:  float64(start) / fps,
			End:    float64(last+1) / fps,
			Box:    robustUnion(fb, charH),
			Frames: last - start + 1,
		})
		i = j
	}
	return evs
}
