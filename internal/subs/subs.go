// Package subs builds the subtitle plan: per-frame detection, event
// aggregation with temporal closing, and the repair/visibility masks each
// engine consumes. It is the single entry point for mask production; the
// engines only consume what a Plan carries.
package subs

import (
	"os"
	"fmt"
	"io"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/detect"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/events"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ffx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/mask"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ocr"
)

type Band struct {
	Y, H int
}

type Detection struct {
	Frames     int `json:"frames"`
	TextFrames int `json:"text_frames"`
	Boxes      int `json:"boxes"`
	TextPix    int `json:"text_pixels"`
	// Only meaningful for the post-repair check: detections that overlap a
	// region the original had subtitles in (true residue) versus everything
	// else (prop text that was never a subtitle).
	ResidueFrames int `json:"residue_frames,omitempty"`
	ResidueBoxes  int `json:"residue_boxes,omitempty"`
}

// Plan is everything the fill engines need: the events (already closed over
// detection gaps), the per-frame repair masks (event-wide unions, clipped to
// each event's rows), and the shot id per frame.
type Plan struct {
	Events    []events.Event
	Masks     []mask.Frame
	Frames    []events.Frame
	Detection Detection
	Band      Band
	Params    detect.Params
	Cuts      []float64
	FPS       float64
	ShotID    []int
}

type Options struct {
	Input        string
	W            int
	FPS          float64
	Band         Band
	Params       detect.Params
	Cuts         []float64
	MaxGap       int
	CloseGap     int     // temporal closing window in frames; events closer than this with matching boxes merge
	CloseOverlap float64 // min intersection-over-smaller-box for merging adjacent events
	EdgePad      int     // frames padded before/after each event to cover fade-in/out residue
	DumpDir      string
	DumpLimit    int
	DumpStride   int
	OCR          *ocr.Client // nil disables OCR fusion
	OCRStride    int         // sample every Nth frame for OCR; default 12
	Log          io.Writer
}

// ComputeBand picks the subtitle band; the y offset stays even so yuv420p
// cropping is chroma-aligned.
func ComputeBand(h int, startFrac float64) Band {
	y := int(float64(h)*startFrac + 0.5)
	y -= y % 2
	if y > h-16 {
		y = h - 16
	}
	bh := h - y
	if bh%2 != 0 {
		bh--
	}
	return Band{Y: y, H: bh}
}

// ShotIDs assigns a monotonically increasing shot id per frame from the cut
// timestamps.
func ShotIDs(total int, cuts []float64, fps float64) []int {
	sid := make([]int, total)
	s, ci := 0, 0
	for f := 0; f < total; f++ {
		t := float64(f) / fps
		for ci < len(cuts) && cuts[ci] <= t+1e-9 {
			s++
			ci++
		}
		sid[f] = s
	}
	return sid
}

// overlapRatio is the intersection over the SMALLER box. Subtitle lines at
// the same position differ in length (a short line's box sits inside the
// long one's), which makes plain IoU misleadingly small; containment is what
// "same line position" actually means.
func overlapRatio(a, b imgx.Rect) float64 {
	inter := a.Intersect(b)
	if inter.Empty() {
		return 0
	}
	small := a.Area()
	if b.Area() < small {
		small = b.Area()
	}
	if small <= 0 {
		return 0
	}
	return float64(inter.Area()) / float64(small)
}

// closeEvents merges adjacent events whose boxes sit at the same position
// (overlap ≥ overlapThr) and whose gap is at most gap frames, within one
// shot. This is the temporal closing step: subtitle fades and line
// transitions drop detection on the in-between frames, and those gap frames
// must join the event — otherwise they keep their text un-repaired AND,
// worse, serve as "clean" sample sources for the neighbours, leaking the
// text back in.
func closeEvents(evs []events.Event, shotID []int, gap int, overlapThr float64) []events.Event {
	if len(evs) < 2 || gap < 0 {
		return evs
	}
	out := make([]events.Event, 0, len(evs))
	cur := evs[0]
	for _, ev := range evs[1:] {
		gapF := ev.StartF - cur.EndF - 1
		merge := gapF >= 0 && gapF <= gap &&
			cur.EndF < len(shotID) && ev.StartF < len(shotID) &&
			shotID[cur.EndF] == shotID[ev.StartF] &&
			overlapRatio(cur.Box, ev.Box) >= overlapThr
		if merge {
			cur.EndF = ev.EndF
			cur.End = ev.End
			cur.Box = cur.Box.Union(ev.Box)
			cur.Frames = cur.EndF - cur.StartF + 1
			continue
		}
		out = append(out, cur)
		cur = ev
	}
	return append(out, cur)
}

// padEvents extends every event by pad frames on both sides to catch the
// low-contrast residue of subtitle fade-in/out that detection misses.
// Padding never crosses into a neighbour's original range.
func padEvents(evs []events.Event, pad, total int, fps float64) []events.Event {
	if pad <= 0 {
		return evs
	}
	out := make([]events.Event, len(evs))
	prevEnd := -1
	for i, ev := range evs {
		lo := ev.StartF - pad
		if lo < prevEnd+1 {
			lo = prevEnd + 1
		}
		if lo < 0 {
			lo = 0
		}
		hi := ev.EndF + pad
		if i+1 < len(evs) && hi >= evs[i+1].StartF {
			hi = evs[i+1].StartF - 1
		}
		if hi >= total {
			hi = total - 1
		}
		if hi < lo {
			hi = lo
		}
		ev.StartF, ev.EndF = lo, hi
		ev.Start = float64(lo) / fps
		ev.End = float64(hi+1) / fps
		ev.Frames = hi - lo + 1
		out[i] = ev
		prevEnd = hi
	}
	return out
}

// makeMasksEventWide replaces every frame's mask with the union over its
// event. A per-frame mask misses text whenever detection drops a character
// or a whole frame, and the temporal engine would then leave it on screen;
// the union matches delogo, which covers the event's box for its whole time
// range.
func makeMasksEventWide(masks []mask.Frame, evs []events.Event) {
	for _, ev := range evs {
		end := ev.EndF
		if end >= len(masks) {
			end = len(masks) - 1
		}
		var u mask.Frame
		for f := ev.StartF; f <= end; f++ {
			u = u.Union(masks[f])
		}
		if u.Empty() {
			continue
		}
		for f := ev.StartF; f <= end; f++ {
			masks[f] = u
		}
	}
}

// limitMasks keeps each frame's mask only within its event's repair band, so
// prop/label text detected elsewhere in the band is never touched.
func limitMasks(masks []mask.Frame, evs []events.Event, pad, bandH int) {
	for _, ev := range evs {
		y0 := ev.Box.Y - pad
		if y0 < 0 {
			y0 = 0
		}
		y1 := ev.Box.Y + ev.Box.H + pad
		if y1 > bandH {
			y1 = bandH
		}
		for f := ev.StartF; f <= ev.EndF && f < len(masks); f++ {
			masks[f] = masks[f].ClipRows(y0, y1)
		}
	}
}

// EventBoxAt returns the band-coordinate box of the event covering frame f.
func EventBoxAt(evs []events.Event, f int) *imgx.Rect {
	for k := range evs {
		if f >= evs[k].StartF && f <= evs[k].EndF {
			return &evs[k].Box
		}
	}
	return nil
}

// Measure runs the detector over the band of input with a small worker
// pool (detection is per-frame independent). frames is always returned;
// masks only when storeMasks is set.
func Measure(input string, w int, b Band, params detect.Params, storeMasks bool, dumpDir string, dumpLimit, dumpStride int) ([]events.Frame, []mask.Frame, Detection, error) {
	var det Detection
	fr, err := ffx.NewFrameReader(input, fmt.Sprintf("crop=%d:%d:0:%d,format=gray", w, b.H, b.Y), w, b.H, "gray")
	if err != nil {
		return nil, nil, det, err
	}
	defer fr.Close()
	if dumpStride < 1 {
		dumpStride = 1
	}

	type job struct {
		idx int
		buf []byte
	}
	type detOut struct {
		idx     int
		boxes   []imgx.Rect
		mf      mask.Frame
		textPix int
	}
	workers := runtime.NumCPU()
	if workers > 8 {
		workers = 8
	}
	if workers < 1 {
		workers = 1
	}
	jobs := make(chan job, workers*2)
	outs := make(chan detOut, workers*2)
	pool := make(chan []byte, workers)
	for i := 0; i < workers; i++ {
		pool <- make([]byte, w*b.H)
	}
	var wg sync.WaitGroup
	var dumped atomic.Int32
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				res := detect.Detect(j.buf, w, b.H, params)
				o := detOut{idx: j.idx, boxes: res.Boxes, textPix: res.TextPix}
				if storeMasks {
					o.mf = mask.Encode(res.Mask, w, b.H)
				}
				if dumpDir != "" && len(res.Boxes) > 0 && j.idx%dumpStride == 0 && int(dumped.Load()) < dumpLimit {
					if dumped.Add(1) <= int32(dumpLimit) {
						_ = detect.DumpPreview(j.buf, w, b.H, res, fmt.Sprintf("%s/det_f%05d.png", dumpDir, j.idx))
					}
				}
				outs <- o
				pool <- j.buf
			}
		}()
	}
	readErr := make(chan error, 1)
	go func() {
		defer close(jobs)
		idx := 0
		for {
			buf := <-pool
			ok, err := fr.Next(buf)
			if err != nil {
				pool <- buf
				readErr <- err
				return
			}
			if !ok {
				pool <- buf
				readErr <- nil
				return
			}
			jobs <- job{idx: idx, buf: buf}
			idx++
		}
	}()
	go func() {
		wg.Wait()
		close(outs)
	}()

	results := make([]detOut, 0, 4096)
	for o := range outs {
		results = append(results, o)
	}
	if err := <-readErr; err != nil {
		return nil, nil, det, err
	}
	sort.Slice(results, func(i, j int) bool { return results[i].idx < results[j].idx })

	frames := make([]events.Frame, 0, len(results))
	var masks []mask.Frame
	if storeMasks {
		masks = make([]mask.Frame, 0, len(results))
	}
	for _, o := range results {
		frames = append(frames, events.Frame{Boxes: o.boxes})
		if storeMasks {
			masks = append(masks, o.mf)
		}
		det.Frames++
		if len(o.boxes) > 0 {
			det.TextFrames++
			det.Boxes += len(o.boxes)
			det.TextPix += o.textPix
		}
	}
	return frames, masks, det, nil
}

// extractBandFrame re-reads one frame's gray band via ffmpeg's select filter;
// used to refine OCR boxes, so it only runs on the few frames where OCR found
// text the classical detector missed.
func extractBandFrame(input string, f, w int, b Band) ([]uint8, error) {
	vf := fmt.Sprintf("select='eq(n,%d)',crop=%d:%d:0:%d,format=gray", f, w, b.H, b.Y)
	fr, err := ffx.NewFrameReader(input, vf, w, b.H, "gray")
	if err != nil {
		return nil, err
	}
	defer fr.Close()
	buf := make([]byte, fr.FrameSize())
	ok, err := fr.Next(buf)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("frame %d not readable", f)
	}
	return buf, nil
}

// refineOCRBox runs the stroke detector on a crop around an OCR box so the
// repair mask follows the glyph strokes instead of the whole rectangle. The
// centering gate is off: the crop is already centered on the text. Returns
// band-coordinate bits, or nil when no strokes validate.
func refineOCRBox(gray []uint8, w int, b Band, r imgx.Rect, p detect.Params) []uint8 {
	p.Centered = false
	pad := p.CharH / 2
	x0 := max(0, r.X-pad)
	y0 := max(0, r.Y-pad)
	x1 := min(w, r.X+r.W+pad)
	y1 := min(b.H, r.Y+r.H+pad)
	cw, ch := x1-x0, y1-y0
	if cw < 4 || ch < 4 {
		return nil
	}
	crop := make([]uint8, cw*ch)
	for y := 0; y < ch; y++ {
		copy(crop[y*cw:(y+1)*cw], gray[(y0+y)*w+x0:(y0+y)*w+x1])
	}
	res := detect.Detect(crop, cw, ch, p)
	n := 0
	for _, v := range res.Mask {
		n += int(v)
	}
	if n < 8 {
		return nil
	}
	bits := make([]uint8, w*b.H)
	for y := 0; y < ch; y++ {
		for x := 0; x < cw; x++ {
			if res.Mask[y*cw+x] != 0 {
				bits[(y0+y)*w+x0+x] = 1
			}
		}
	}
	return bits
}

// fuseOCR merges OCR sidecar detections into the per-frame boxes (union with
// the classical boxes) and, when masks are stored, into the repair masks.
// Boxes that substantially overlap a classical box add nothing; the rest are
// stroke-refined on a re-extracted band frame, falling back to the bare rect
// when no strokes validate inside. Returns the number of boxes added.
func fuseOCR(o Options, frames []events.Frame, masks []mask.Frame) int {
	stride := o.OCRStride
	if stride < 1 {
		stride = 12
	}
	var sample []int
	for f := 0; f < len(frames); f += stride {
		sample = append(sample, f)
	}
	boxes, err := o.OCR.DetectFrames(o.Input, o.Band.Y, o.Band.H, sample)
	if err != nil {
		logf(o.Log, "warn: ocr sidecar unavailable, continuing without it: %v\n", err)
		return 0
	}
	byFrame := make(map[int][]ocr.Box)
	for _, bx := range boxes {
		if bx.Frame < 0 || bx.Frame >= len(frames) || bx.Rect.Empty() {
			continue
		}
		byFrame[bx.Frame] = append(byFrame[bx.Frame], bx)
	}
	added := 0
	grayCache := make(map[int][]uint8)
	for f, obs := range byFrame {
		for _, ob := range obs {
			covered := false
			best := 0.0
			for _, cb := range frames[f].Boxes {
				// Covered means a classical box already explains most of
				// THIS OCR box (intersection over the OCR area). Min-area
				// overlap would also fire when the OCR box subsumes a small
				// classical box — but that is exactly the box fusion exists
				// to add.
				if inter := ob.Rect.Intersect(cb); !inter.Empty() {
					r := float64(inter.Area()) / float64(ob.Rect.Area())
					if r > best {
						best = r
					}
					if inter.Area()*2 >= ob.Rect.Area() {
						covered = true
						break
					}
				}
			}
			if os.Getenv("DESUB_DBG") != "" {
				fmt.Fprintf(os.Stderr, "fuseOCR f%d rect=%v classical=%d cov=%.2f covered=%v\n",
					f, ob.Rect, len(frames[f].Boxes), best, covered)
			}
			if covered {
				continue
			}
			frames[f].Boxes = append(frames[f].Boxes, ob.Rect)
			added++
			if masks == nil {
				continue
			}
			gray, ok := grayCache[f]
			if !ok {
				gray, err = extractBandFrame(o.Input, f, o.W, o.Band)
				if err != nil {
					logf(o.Log, "warn: ocr refine: frame %d unreadable: %v\n", f, err)
					gray = nil
				}
				grayCache[f] = gray
			}
			bits := make([]uint8, o.W*o.Band.H)
			masks[f].Decode(bits, o.W)
			if gray != nil {
				if ref := refineOCRBox(gray, o.W, o.Band, ob.Rect, o.Params); ref != nil {
					for i, v := range ref {
						bits[i] |= v
					}
				} else {
					rasterizeRect(bits, o.W, o.Band.H, ob.Rect)
				}
			} else {
				rasterizeRect(bits, o.W, o.Band.H, ob.Rect)
			}
			masks[f] = mask.Encode(bits, o.W, o.Band.H)
		}
	}
	return added
}

func rasterizeRect(bits []uint8, w, h int, r imgx.Rect) {
	x0 := max(0, r.X)
	y0 := max(0, r.Y)
	x1 := min(w, r.X+r.W)
	y1 := min(h, r.Y+r.H)
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			bits[y*w+x] = 1
		}
	}
}

func logf(w io.Writer, format string, args ...any) {
	if w != nil {
		fmt.Fprintf(w, format, args...)
	}
}

// calibrateParams re-estimates CharH from the accepted line box heights.
// DefaultParams guesses CharH from the frame height; when the guess is off
// (here: large glyphs in a small frame), component gates tuned to CharH
// reject whole glyph bodies and lines assemble only partially. The estimate
// is the tallest STRONG cluster: scanning from the maximum down, the first
// height whose ±1/7 bucket holds at least max(20, n/5) boxes. Percentiles
// cannot separate the two failure modes — a real tall-glyph population
// (T1: 88/386 boxes at 42px) from a gradual false-positive tail (slice1:
// ~10 scattered boxes at 60-76px above a 26-strong 50px cluster) — a
// support-gated cluster scan can. A clear mismatch (outside [3/4, 4/3]×
// CharH) triggers one re-detection with the measured value.
func calibrateParams(frames []events.Frame, p detect.Params) (detect.Params, bool) {
	var hs []int
	for _, f := range frames {
		for _, b := range f.Boxes {
			hs = append(hs, b.H)
		}
	}
	if len(hs) < 20 {
		return p, false
	}
	sort.Ints(hs)
	need := len(hs) / 5
	if need < 20 {
		need = 20
	}
	est := -1
	for i := len(hs) - 1; i >= 0 && est < 0; i-- {
		if i+1 < len(hs) && hs[i] == hs[i+1] {
			continue
		}
		h := hs[i]
		lo, hi := h*6/7, h*8/7
		var cnt, sum int
		for _, v := range hs {
			if v >= lo && v <= hi {
				cnt++
				sum += v
			}
		}
		if cnt >= need {
			est = (sum + cnt/2) / cnt
		}
	}
	if est < 0 {
		return p, false
	}
	fire := est*4 < p.CharH*3 || est*3 > p.CharH*4
	if os.Getenv("DESUB_DBG") != "" {
		fmt.Fprintf(os.Stderr, "calibrate: n=%d p50=%d p75=%d p90=%d max=%d est=%d charH=%d fire=%v\n",
			len(hs), hs[len(hs)/2], hs[len(hs)*3/4], hs[len(hs)*9/10], hs[len(hs)-1], est, p.CharH, fire)
		fmt.Fprintf(os.Stderr, "calibrate hist:")
		for i := 0; i < len(hs); {
			j := i
			for j < len(hs) && hs[j] == hs[i] {
				j++
			}
			fmt.Fprintf(os.Stderr, " %d:%d", hs[i], j-i)
			i = j
		}
		fmt.Fprintln(os.Stderr)
	}
	if !fire {
		return p, false
	}
	ch := est
	if ch < 12 {
		ch = 12
	}
	if ch > 96 {
		ch = 96
	}
	if ch == p.CharH {
		return p, false
	}
	np := p
	np.CharH = ch
	np.Stroke = ch / 9
	if np.Stroke < 2 {
		np.Stroke = 2
	}
	np.MinLineW = ch * 9 / 5
	np.MaskDilate = np.Stroke*2 + 1
	return np, true
}

// CalibrateParams exposes the charH recalibration for callers that run
// Measure directly instead of Build (e.g. the eval package).
func CalibrateParams(frames []events.Frame, p detect.Params) (detect.Params, bool) {
	return calibrateParams(frames, p)
}

// Build runs detection, event aggregation, temporal closing and edge
// padding, then produces the per-frame repair masks (event-wide union
// clipped to the event rows). When an OCR client is configured, sidecar
// detections are fused into the frames and masks before event aggregation.
func Build(o Options) (*Plan, error) {
	if o.CloseGap < 0 {
		o.CloseGap = 0
	}
	if o.CloseOverlap <= 0 {
		o.CloseOverlap = 0.5
	}
	frames, masks, det, err := Measure(o.Input, o.W, o.Band, o.Params, true, o.DumpDir, o.DumpLimit, o.DumpStride)
	if err != nil {
		return nil, err
	}
	if np, ok := calibrateParams(frames, o.Params); ok {
		logf(o.Log, "detect: charH recalibrated %d -> %d; re-running detection\n", o.Params.CharH, np.CharH)
		o.Params = np
		frames, masks, det, err = Measure(o.Input, o.W, o.Band, o.Params, true, o.DumpDir, o.DumpLimit, o.DumpStride)
		if err != nil {
			return nil, err
		}
	}
	if o.OCR != nil {
		if n := fuseOCR(o, frames, masks); n > 0 {
			logf(o.Log, "ocr: fused %d box(es) the classical detector missed\n", n)
			det.Boxes += n
		}
	}
	total := len(frames)
	evs := events.Build(frames, o.FPS, o.MaxGap, o.Cuts, o.Params.CharH)
	shotID := ShotIDs(total, o.Cuts, o.FPS)
	evs = closeEvents(evs, shotID, o.CloseGap, o.CloseOverlap)
	evs = padEvents(evs, o.EdgePad, total, o.FPS)
	if len(masks) == total {
		makeMasksEventWide(masks, evs)
		limitMasks(masks, evs, o.Params.MaskDilate*2, o.Band.H)
	}
	return &Plan{
		Events: evs, Masks: masks, Frames: frames, Detection: det,
		Band: o.Band, Params: o.Params, Cuts: o.Cuts, FPS: o.FPS, ShotID: shotID,
	}, nil
}
