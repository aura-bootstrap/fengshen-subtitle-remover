// Package engine implements the removal backends. Both consume detections
// and write a new file; audio is stream-copied, video is re-encoded once.
package engine

import (
	"fmt"
	"strings"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/events"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ffx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
)

type DelogoOptions struct {
	Input  string
	Output string
	W, H   int
	Events []events.Event // full-frame coordinates
	Pad    int
	CRF    int
	Preset string
	// EncColor carries ffx.MediaInfo.ColorEncodeArgs() output (R9.3).
	EncColor []string
}

// RunDelogo builds one ffmpeg filtergraph with a timed delogo box per event.
func RunDelogo(o DelogoOptions) error {
	const maxBoxes = 400
	var parts []string
	for _, ev := range o.Events {
		r := ev.Box.Expand(o.Pad).Clamp(o.W, o.H)
		r = r.Intersect(imgx.Rect{X: 1, Y: 1, W: o.W - 2, H: o.H - 2})
		if r.W < 6 || r.H < 6 {
			continue
		}
		parts = append(parts, fmt.Sprintf("delogo=x=%d:y=%d:w=%d:h=%d:enable='between(t,%.3f,%.3f)'",
			r.X, r.Y, r.W, r.H, ev.Start, ev.End))
	}
	if len(parts) == 0 {
		return fmt.Errorf("delogo: no usable events")
	}
	if len(parts) > maxBoxes {
		return fmt.Errorf("delogo: %d boxes exceeds single-pass limit (%d), split the input", len(parts), maxBoxes)
	}
	vf := strings.Join(parts, ",") + ",format=yuv420p"
	args := []string{"-hide_banner", "-nostdin", "-y", "-loglevel", "error",
		"-i", o.Input, "-vf", vf,
		"-map", "0:v:0", "-map", "0:a?"}
	args = append(args, o.EncColor...)
	args = append(args,
		"-c:v", "libx264", "-crf", fmt.Sprint(o.CRF), "-preset", o.Preset,
		"-threads", fmt.Sprint(ffx.CPUWorkers()),
		"-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", o.Output)
	return ffx.Run(args...)
}
