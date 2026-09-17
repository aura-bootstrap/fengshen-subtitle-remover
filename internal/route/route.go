// Package route classifies each subtitle event into a difficulty tier
// (T0..T5) and dispatches it to an engine tier: T0–T2 go to real-pixel
// repair (motion), T3+ to the generative tier (propainter), with T4/T5
// additionally flagged as high-risk for human review.
package route

import (
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/events"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/mask"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/subs"
)

// Tier is the difficulty grade of one subtitle event (roadmap T0–T5).
type Tier int

const (
	T0 Tier = iota // static shot, plain hard subtitles
	T1             // slow pan / gentle motion
	T2             // scrims, two-line subs, general motion
	T3             // artsy text, large area, fast motion
	T4             // text over faces/signage/dense texture
	T5             // persistent same-position occlusion (logo/watermark)
)

// Classification thresholds.
const (
	T0Motion  = 0.5  // px/frame mean homography displacement: static below this
	T1Motion  = 2.0  // slow motion below this
	T2Motion  = 8.0  // general motion below this; above is T3-hard
	T3Area    = 0.20 // mask area fraction of the band above this is T3-hard
	T4Texture = 28.0 // mean gradient energy under the mask above this is T4
	T5DurFrac = 0.7  // event spanning this fraction of the whole video is T5
)

// Features describes one event for classification.
type Features struct {
	AreaFrac  float64 // mask area as a fraction of the band
	MotionMag float64 // mean homography displacement over the event (px/frame)
	DurFrames int
	Texture   float64 // gradient energy under the mask
}

// Classify grades an event from its features and records which conditions
// fired. Persistent-occlusion T5 cannot be judged from one event alone;
// Dispatch applies it as an override.
func Classify(f Features) (Tier, []string) {
	switch {
	case f.Texture >= T4Texture:
		return T4, []string{"dense texture under mask"}
	case f.AreaFrac >= T3Area:
		return T3, []string{"large mask area"}
	case f.MotionMag > T2Motion:
		return T3, []string{"fast motion"}
	case f.MotionMag > T1Motion:
		return T2, []string{"general motion"}
	case f.MotionMag > T0Motion:
		return T1, []string{"slow motion"}
	default:
		return T0, []string{"static shot"}
	}
}

// EngineFor maps a tier to its engine tier: T0–T2 repair with real pixels,
// T3+ need the generative sidecar.
func EngineFor(t Tier) string {
	if t >= T3 {
		return "propainter"
	}
	return "motion"
}

// Decision is the routing verdict for one event.
type Decision struct {
	Event  events.Event `json:"event"`
	Tier   Tier         `json:"tier"`
	Engine string       `json:"engine"`
	Risk   bool         `json:"risk,omitempty"`
	Reason []string     `json:"reason,omitempty"`
}

// Dispatch classifies every event in the plan. mags[f] is the homography
// displacement between frames f and f+1 (nil means unknown → static). tex,
// when non-nil, estimates the gradient energy under a box; nil means 0.
// forced, when non-empty, overrides the engine for every event (R7.5) while
// tiers are still classified for the report. T4/T5 always set Risk (R7.4).
func Dispatch(p *subs.Plan, bandW int, mags []float64, tex func(imgx.Rect) float64, forced string) []Decision {
	out := make([]Decision, 0, len(p.Events))
	bandArea := float64(p.Band.H * bandW)
	total := len(p.Masks)
	for _, ev := range p.Events {
		f := Features{DurFrames: ev.EndF - ev.StartF + 1}
		var px, nMag int
		var magSum float64
		for fr := ev.StartF; fr <= ev.EndF && fr < len(p.Masks); fr++ {
			px += maskPx(p.Masks[fr])
			if fr < len(mags) {
				magSum += mags[fr]
				nMag++
			}
		}
		if bandArea > 0 && f.DurFrames > 0 {
			f.AreaFrac = float64(px) / float64(f.DurFrames) / bandArea
		}
		if nMag > 0 {
			f.MotionMag = magSum / float64(nMag)
		}
		if tex != nil {
			f.Texture = tex(ev.Box)
		}
		tier, reason := Classify(f)
		if total > 0 && float64(f.DurFrames) >= T5DurFrac*float64(total) {
			tier = T5
			reason = append(reason, "persistent occlusion")
		}
		d := Decision{Event: ev, Tier: tier, Engine: EngineFor(tier), Reason: reason}
		if tier >= T4 {
			d.Risk = true
		}
		if forced != "" {
			d.Engine = forced
			d.Reason = append(d.Reason, "forced engine")
		}
		out = append(out, d)
	}
	return out
}

// maskPx counts repair-mask pixels from the RLE triples (row, x0, x1
// inclusive).
func maskPx(m mask.Frame) int {
	n := 0
	for k := 0; k+2 < len(m.RLE); k += 3 {
		n += int(m.RLE[k+2]) - int(m.RLE[k+1]) + 1
	}
	return n
}
