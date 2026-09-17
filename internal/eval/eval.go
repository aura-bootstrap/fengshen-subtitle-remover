// Package eval implements the quantitative acceptance loop (R11): BurnIn
// synthesizes burned-subtitle samples with exact ground-truth masks, Compare
// scores a removal output against the clean source (mask-region PSNR/SSIM,
// temporal variance, residue), and DetectionRecall measures how much of the
// GT mask the detector recovers.
package eval

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/detect"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/events"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ffx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/mask"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/subs"
)

// Cue is one subtitle line burned over [Start, End] seconds.
type Cue struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// Style controls BurnIn rendering. Zero values pick defaults.
type Style struct {
	FontFile  string `json:"font_file"`
	FontSize  int    `json:"font_size"`  // default H/15
	Y         int    `json:"y"`          // top of the text block; 0 → 82% of H
	Box       bool   `json:"box"`        // semi-transparent backdrop bar
	FontColor string `json:"font_color"` // default white
	BoxColor  string `json:"box_color"`  // default black@0.55
}

var fontCandidates = []string{
	"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
	"/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
	`C:\Windows\Fonts\msyh.ttc`,
	`C:\Windows\Fonts\simhei.ttf`,
}

func defaultFont() (string, error) {
	for _, p := range fontCandidates {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("eval: no CJK font found (tried %v)", fontCandidates)
}

// BurnIn renders cues onto the clean source, writing the burned video and a
// lossless gray GT mask (255 = pixels the remover must restore). maskOut
// should use a container that accepts FFV1 (e.g. .mkv).
func BurnIn(src, out, maskOut string, cues []Cue, st Style) error {
	if len(cues) == 0 {
		return fmt.Errorf("eval: no cues")
	}
	info, err := ffx.Probe(src)
	if err != nil {
		return err
	}
	if st.FontFile == "" {
		if st.FontFile, err = defaultFont(); err != nil {
			return err
		}
	}
	if st.FontSize <= 0 {
		st.FontSize = info.H / 15
	}
	if st.Y == 0 {
		st.Y = int(0.82 * float64(info.H))
	}
	if st.FontColor == "" {
		st.FontColor = "white"
	}
	if st.BoxColor == "" {
		st.BoxColor = "black@0.55"
	}
	dir, err := os.MkdirTemp("", "desub-synth-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	lineH := int(float64(st.FontSize) * 1.35)
	// Text goes through textfile= so no drawtext escaping is needed.
	filter := func(fontColor, boxColor string) (string, error) {
		var vf string
		for i, c := range cues {
			tf := filepath.Join(dir, fmt.Sprintf("cue%d.txt", i))
			if err := os.WriteFile(tf, []byte(c.Text), 0o644); err != nil {
				return "", err
			}
			f := fmt.Sprintf("drawtext=fontfile='%s':textfile='%s':fontsize=%d:fontcolor=%s:x=(w-text_w)/2:y=%d:enable='between(t,%.3f,%.3f)'",
				st.FontFile, tf, st.FontSize, fontColor, st.Y+i*lineH, c.Start, c.End)
			if st.Box {
				f += fmt.Sprintf(":box=1:boxcolor=%s:boxborderw=%d", boxColor, st.FontSize/4+4)
			}
			vf += f + ","
		}
		return vf[:len(vf)-1], nil
	}
	vfOut, err := filter(st.FontColor, st.BoxColor)
	if err != nil {
		return err
	}
	// On the mask everything (text + backdrop) is opaque white on black.
	vfMask, err := filter("white", "white@1.0")
	if err != nil {
		return err
	}
	if err := ffx.Run("-hide_banner", "-nostdin", "-y", "-loglevel", "error",
		"-i", src, "-vf", vfOut+",format=yuv420p",
		"-map", "0:v:0", "-map", "0:a?",
		"-c:v", "libx264", "-crf", "16", "-preset", "medium",
		"-c:a", "copy", out); err != nil {
		return fmt.Errorf("burn-in: %w", err)
	}
	if err := ffx.Run("-hide_banner", "-nostdin", "-y", "-loglevel", "error",
		"-f", "lavfi", "-i", fmt.Sprintf("color=c=black:s=%dx%d:r=%.6f", info.W, info.H, info.FPS),
		"-t", fmt.Sprintf("%.6f", info.Duration),
		"-vf", vfMask, "-pix_fmt", "gray", "-c:v", "ffv1", maskOut); err != nil {
		return fmt.Errorf("gt mask: %w", err)
	}
	return nil
}

// Metrics is the machine-readable acceptance report (R11.4).
type Metrics struct {
	Frames     int     `json:"frames"`
	MaskedPx   int64   `json:"masked_px"`
	PSNR       float64 `json:"psnr_db"` // GT-mask region; 99 when identical
	SSIM       float64 `json:"ssim"`    // luma, 8x8 windows inside the GT mask
	MaskRecall float64 `json:"mask_recall,omitempty"`
	FrameVar   float64 `json:"frame_var"` // mean squared frame-to-frame diff inside the mask
	Residue    int     `json:"residue"`   // post-repair detections overlapping the GT mask
}

// PSNR converts mean squared error to dB, capped at 99 for identical inputs.
func PSNR(mse float64) float64 {
	if mse <= 1e-9 {
		return 99
	}
	return 10 * math.Log10(255*255/mse)
}

// SSIM scores two equal-length luma blocks with the standard constants.
func SSIM(x, y []uint8) float64 {
	n := float64(len(x))
	if len(x) == 0 || len(x) != len(y) {
		return 0
	}
	var mx, my float64
	for i := range x {
		mx += float64(x[i])
		my += float64(y[i])
	}
	mx /= n
	my /= n
	var vx, vy, cxy float64
	for i := range x {
		dx := float64(x[i]) - mx
		dy := float64(y[i]) - my
		vx += dx * dx
		vy += dy * dy
		cxy += dx * dy
	}
	vx /= n
	vy /= n
	cxy /= n
	const c1 = (0.01 * 255) * (0.01 * 255)
	const c2 = (0.03 * 255) * (0.03 * 255)
	return ((2*mx*my + c1) * (2*cxy + c2)) / ((mx*mx + my*my + c1) * (vx + vy + c2))
}

// Recall is the intersection-over-GT fraction (R11.2 acceptance input).
func Recall(intersect, gtTotal int64) float64 {
	if gtTotal == 0 {
		return 1
	}
	return float64(intersect) / float64(gtTotal)
}

// dilate grows the bit mask by r pixels (Chebyshev).
func dilate(bits []uint8, w, h, r int) []uint8 {
	out := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if bits[y*w+x] == 0 {
				continue
			}
			for dy := -r; dy <= r; dy++ {
				yy := y + dy
				if yy < 0 || yy >= h {
					continue
				}
				for dx := -r; dx <= r; dx++ {
					xx := x + dx
					if xx >= 0 && xx < w {
						out[yy*w+xx] = 1
					}
				}
			}
		}
	}
	return out
}

// Compare scores the removal output against the clean source over the GT
// mask region (R11.2, R11.3). Residue re-runs detection on the output and
// counts detections that overlap the GT mask.
func Compare(out, gt, gtMask string, band subs.Band) (*Metrics, error) {
	info, err := ffx.Probe(gt)
	if err != nil {
		return nil, err
	}
	w, h := info.W, band.H
	vf := fmt.Sprintf("crop=%d:%d:0:%d,format=rgb24", w, h, band.Y)
	vfMask := fmt.Sprintf("crop=%d:%d:0:%d,format=gray", w, h, band.Y)
	rOut, err := ffx.NewFrameReader(out, vf, w, h, "rgb24")
	if err != nil {
		return nil, err
	}
	defer rOut.Close()
	rGT, err := ffx.NewFrameReader(gt, vf, w, h, "rgb24")
	if err != nil {
		return nil, err
	}
	defer rGT.Close()
	rMask, err := ffx.NewFrameReader(gtMask, vfMask, w, h, "gray")
	if err != nil {
		return nil, err
	}
	defer rMask.Close()

	m := &Metrics{}
	bo := make([]byte, w*h*3)
	bg := make([]byte, w*h*3)
	bm := make([]byte, w*h)
	var prev []byte
	var boxes []imgx.Rect // GT-mask bbox per frame (zero when the frame has no mask)
	var seSum, ssimSum float64
	var ssimN, varN int64
	var varSum float64
	xb := make([]uint8, 64)
	yb := make([]uint8, 64)
	for {
		okO, err := rOut.Next(bo)
		if err != nil {
			return nil, err
		}
		okG, err := rGT.Next(bg)
		if err != nil {
			return nil, err
		}
		okM, err := rMask.Next(bm)
		if err != nil {
			return nil, err
		}
		if !okO || !okG || !okM {
			break
		}
		var bits []uint8
		var nBits int64
		bbox := imgx.Rect{}
		for p := 0; p < w*h; p++ {
			if bm[p] > 127 {
				nBits++
				x, y := p%w, p/w
				if nBits == 1 {
					bbox = imgx.Rect{X: x, Y: y, W: 1, H: 1}
				} else {
					bbox = growTo(bbox, x, y)
				}
			}
		}
		boxes = append(boxes, bbox)
		if nBits > 0 {
			raw := make([]uint8, w*h)
			for p := 0; p < w*h; p++ {
				if bm[p] > 127 {
					raw[p] = 1
				}
			}
			bits = dilate(raw, w, h, 2)
		}
		if bits == nil {
			m.Frames++
			prev = append(prev[:0], bo...)
			continue
		}
		for p := 0; p < w*h; p++ {
			if bits[p] == 0 {
				continue
			}
			nBits++
			for c := 0; c < 3; c++ {
				d := float64(bo[p*3+c]) - float64(bg[p*3+c])
				seSum += d * d
			}
		}
		if prev != nil {
			for p := 0; p < w*h; p++ {
				if bits[p] == 0 {
					continue
				}
				for c := 0; c < 3; c++ {
					d := float64(bo[p*3+c]) - float64(prev[p*3+c])
					varSum += d * d
					varN++
				}
			}
		}
		// SSIM over 8x8 windows at least half covered by the mask.
		for wy := 0; wy+8 <= h; wy += 8 {
			for wx := 0; wx+8 <= w; wx += 8 {
				cov := 0
				for dy := 0; dy < 8; dy++ {
					for dx := 0; dx < 8; dx++ {
						p := (wy+dy)*w + wx + dx
						if bits[p] != 0 {
							cov++
						}
						xb[dy*8+dx] = luma(bo[p*3], bo[p*3+1], bo[p*3+2])
						yb[dy*8+dx] = luma(bg[p*3], bg[p*3+1], bg[p*3+2])
					}
				}
				if cov >= 32 {
					ssimSum += SSIM(xb, yb)
					ssimN++
				}
			}
		}
		m.MaskedPx += nBits
		m.Frames++
		prev = append(prev[:0], bo...)
	}
	if m.Frames == 0 {
		return nil, fmt.Errorf("eval: no comparable frames")
	}
	if n := float64(m.MaskedPx * 3); n > 0 {
		m.PSNR = PSNR(seSum / n)
	} else {
		m.PSNR = 99
	}
	if ssimN > 0 {
		m.SSIM = ssimSum / float64(ssimN)
	}
	if varN > 0 {
		m.FrameVar = varSum / float64(varN)
	}
	res, err := residueCount(out, info, band, boxes)
	if err != nil {
		return nil, err
	}
	m.Residue = res
	return m, nil
}

// growTo expands r to include point (x, y).
func growTo(r imgx.Rect, x, y int) imgx.Rect {
	if x < r.X {
		r.W += r.X - x
		r.X = x
	}
	if y < r.Y {
		r.H += r.Y - y
		r.Y = y
	}
	if x >= r.X+r.W {
		r.W = x - r.X + 1
	}
	if y >= r.Y+r.H {
		r.H = y - r.Y + 1
	}
	return r
}

func luma(r, g, b byte) uint8 {
	return uint8((int(r) + 2*int(g) + int(b)) / 4)
}

// measureCalibrated runs Measure with the same charH recalibration the
// pipeline applies in subs.Build, so eval metrics see the detector the
// repair actually used. The final params are returned along with the pass.
func measureCalibrated(input string, info *ffx.MediaInfo, band subs.Band, params detect.Params, storeMasks bool) ([]events.Frame, []mask.Frame, detect.Params, error) {
	frames, masks, _, err := subs.Measure(input, info.W, band, params, storeMasks, "", 0, 1)
	if err != nil {
		return nil, nil, params, err
	}
	if np, ok := subs.CalibrateParams(frames, params); ok {
		params = np
		frames, masks, _, err = subs.Measure(input, info.W, band, np, storeMasks, "", 0, 1)
		if err != nil {
			return nil, nil, params, err
		}
	}
	return frames, masks, params, nil
}

// residueCount re-detects on the output and counts boxes whose overlap with
// that frame's GT-mask bbox (dilated by half a character) is at least half
// the box — true removal misses, not prop text (R11.3).
func residueCount(out string, info *ffx.MediaInfo, band subs.Band, boxes []imgx.Rect) (int, error) {
	frames, _, params, err := measureCalibrated(out, info, band, detect.DefaultParams(info.H), false)
	if err != nil {
		return 0, err
	}
	pad := params.CharH / 2
	res := 0
	for i, f := range frames {
		if i >= len(boxes) || boxes[i].W == 0 {
			continue
		}
		reg := boxes[i].Expand(pad).Clamp(info.W, band.H)
		for _, ob := range f.Boxes {
			inter := ob.Intersect(reg)
			if inter.W*inter.H*2 >= ob.W*ob.H {
				res++
			}
		}
	}
	return res, nil
}

// DetectionRecall measures how much of the GT mask the detector recovers on
// the burned video: |det ∩ gt| / |gt| over the band region (R11.2).
func DetectionRecall(burned, gtMask string, band subs.Band, params detect.Params) (float64, error) {
	info, err := ffx.Probe(burned)
	if err != nil {
		return 0, err
	}
	_, masks, _, err := measureCalibrated(burned, info, band, params, true)
	if err != nil {
		return 0, err
	}
	rm, err := ffx.NewFrameReader(gtMask, fmt.Sprintf("crop=%d:%d:0:%d,format=gray", info.W, band.H, band.Y), info.W, band.H, "gray")
	if err != nil {
		return 0, err
	}
	defer rm.Close()
	bits := make([]uint8, info.W*band.H)
	bm := make([]byte, info.W*band.H)
	var inter, total int64
	f := 0
	for {
		ok, err := rm.Next(bm)
		if err != nil {
			return 0, err
		}
		if !ok || f >= len(masks) {
			break
		}
		masks[f].Decode(bits, info.W)
		for p := 0; p < info.W*band.H; p++ {
			if bm[p] > 127 {
				total++
				if bits[p] != 0 {
					inter++
				}
			}
		}
		f++
	}
	return Recall(inter, total), nil
}
