// Package detect implements a classical CV detector for hardcoded dialogue
// subtitles: bright strokes with dark outline inside a search band. The
// white top-hat response isolates strokes; components and assembled lines
// are gated by geometry, contrast and centering. Output is a per-pixel mask
// (dilated) plus line boxes, all in band coordinates.
package detect

import (
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
)

type Params struct {
	FrameH      int
	CharH       int
	Stroke      int
	MinLineW    int
	MinLineComp int
	MaskDilate  int
	MinContrast int
	MinGray     int
	Centered    bool
	CenterTol   float64
}

func DefaultParams(frameH int) Params {
	charH := int(float64(frameH)*0.040 + 0.5)
	if charH < 12 {
		charH = 12
	}
	if charH > 96 {
		charH = 96
	}
	stroke := charH / 9
	if stroke < 2 {
		stroke = 2
	}
	return Params{
		FrameH:      frameH,
		CharH:       charH,
		Stroke:      stroke,
		MinLineW:    charH * 9 / 5,
		MinLineComp: 2,
		MaskDilate:  stroke*2 + 1,
		MinContrast: 14,
		MinGray:     120,
		Centered:    true,
		CenterTol:   0.25,
	}
}

type Result struct {
	Boxes   []imgx.Rect
	Mask    []uint8 // 0/1, dilated, band coordinates
	TextPix int
	Comps   int
}

type comp struct {
	x0, y0, x1, y1 int
	label          int32
	area           int
	sumTop         float64
	sumGray        float64
}

func (c comp) rect() imgx.Rect {
	return imgx.Rect{X: c.x0, Y: c.y0, W: c.x1 - c.x0 + 1, H: c.y1 - c.y0 + 1}
}

type line struct {
	comps []int
	box   imgx.Rect
}

func Detect(band []uint8, w, h int, p Params) Result {
	se := p.Stroke*2 + 1
	tmp := make([]uint8, w*h)
	open := make([]uint8, w*h)
	imgx.MorphRect(band, tmp, w, h, se, se, false)
	imgx.MorphRect(tmp, open, w, h, se, se, true)

	top := make([]uint8, w*h)
	var hist [256]int
	for i, g := range band {
		v := int(g) - int(open[i])
		if v < 0 {
			v = 0
		}
		top[i] = uint8(v)
		hist[top[i]]++
	}
	th := imgx.Otsu(&hist, w*h)
	if th < p.MinContrast {
		th = p.MinContrast
	}
	bin := make([]uint8, w*h)
	for i, v := range top {
		if int(v) >= th {
			bin[i] = 1
		}
	}

	labels, n := imgx.LabelBinary(bin, w, h, true)
	comps := make([]comp, n+1)
	for i := range comps {
		comps[i].x0, comps[i].y0 = 1<<30, 1<<30
	}
	for i, l := range labels {
		if l == 0 {
			continue
		}
		c := &comps[l]
		x, y := i%w, i/w
		if x < c.x0 {
			c.x0 = x
		}
		if y < c.y0 {
			c.y0 = y
		}
		if x > c.x1 {
			c.x1 = x
		}
		if y > c.y1 {
			c.y1 = y
		}
		c.area++
		c.sumTop += float64(top[i])
		c.sumGray += float64(band[i])
	}

	minH := p.Stroke - 1
	if minH < 2 {
		minH = 2
	}
	maxH := p.CharH * 7 / 5
	maxW := p.CharH * 9 / 5
	maxArea := p.CharH * p.CharH
	minArea := max(3, p.Stroke*p.Stroke/2)
	var kept []comp
	for l := 1; l <= n; l++ {
		c := &comps[l]
		c.label = int32(l)
		r := c.rect()
		if r.W > maxW || r.H > maxH || r.H < minH || c.area < minArea || c.area > maxArea {
			continue
		}
		fill := float64(c.area) / float64(r.W*r.H)
		if fill < 0.12 {
			continue
		}
		if c.sumTop/float64(c.area) < float64(th)*1.15 {
			continue
		}
		if c.sumGray/float64(c.area) < float64(p.MinGray) {
			continue
		}
		kept = append(kept, *c)
	}

	lines := groupLines(kept, p, w)
	acceptLbl := make([]bool, n+1)
	for _, l := range lines {
		for _, ci := range l.comps {
			acceptLbl[kept[ci].label] = true
		}
	}
	keep := make([]uint8, w*h)
	textPix := 0
	for i, l := range labels {
		if l != 0 && acceptLbl[l] {
			keep[i] = 1
			textPix++
		}
	}
	mask := make([]uint8, w*h)
	if p.MaskDilate > 0 {
		k := p.MaskDilate*2 + 1
		imgx.MorphBin(keep, mask, w, h, k, k, true)
	} else {
		copy(mask, keep)
	}

	boxes := make([]imgx.Rect, 0, len(lines))
	for _, l := range lines {
		boxes = append(boxes, l.box)
	}
	return Result{Boxes: boxes, Mask: mask, TextPix: textPix, Comps: len(kept)}
}

// groupLines merges components into text lines. A component joins a line
// when it vertically overlaps the line box and sits within ~one character
// horizontally. Lines are then gated: enough components, minimum width,
// minimum height and (by default) horizontal centering.
func groupLines(kept []comp, p Params, w int) []line {
	idx := make([]int, len(kept))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		ra, rb := kept[idx[a]].rect(), kept[idx[b]].rect()
		if ra.Y != rb.Y {
			return ra.Y < rb.Y
		}
		return ra.X < rb.X
	})
	used := make([]bool, len(kept))
	var lines []line
	maxGap := p.CharH * 4 / 5
	for _, i := range idx {
		if used[i] {
			continue
		}
		used[i] = true
		l := line{comps: []int{i}, box: kept[i].rect()}
		changed := true
		for changed {
			changed = false
			for _, j := range idx {
				if used[j] {
					continue
				}
				r := kept[j].rect()
				vOver := min(l.box.Bottom(), r.Bottom()) - max(l.box.Y, r.Y)
				if vOver <= 0 || vOver*2 < min(l.box.H, r.H) {
					continue
				}
				var gap int
				switch {
				case r.X > l.box.Right():
					gap = r.X - l.box.Right()
				case l.box.X > r.Right():
					gap = l.box.X - r.Right()
				}
				if gap > maxGap {
					continue
				}
				l.box = l.box.Union(r)
				l.comps = append(l.comps, j)
				used[j] = true
				changed = true
			}
		}
		lines = append(lines, l)
	}
	var out []line
	for _, l := range lines {
		if len(l.comps) < p.MinLineComp {
			continue
		}
		if l.box.W < p.MinLineW || l.box.H < p.CharH/2 {
			continue
		}
		// A subtitle line is about one character tall; taller groups are
		// label/menu textures chained together, not text lines.
		if l.box.H > p.CharH*3/2 {
			continue
		}
		if p.Centered {
			c := float64(l.box.X) + float64(l.box.W)/2
			if math.Abs(c-float64(w)/2) > p.CenterTol*float64(w) {
				continue
			}
		}
		out = append(out, l)
	}
	return out
}

// DumpPreview writes the band with mask overlay (red) and line boxes (green).
func DumpPreview(band []uint8, w, h int, res Result, path string) error {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for p := 0; p < w*h; p++ {
		g := band[p]
		r, gg, b := g, g, g
		if res.Mask[p] != 0 {
			r, gg, b = 255, g/2, g/2
		}
		img.SetRGBA(p%w, p/w, color.RGBA{R: r, G: gg, B: b, A: 255})
	}
	for _, bx := range res.Boxes {
		drawRect(img, bx, color.RGBA{R: 0, G: 255, B: 0, A: 255})
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func drawRect(img *image.RGBA, r imgx.Rect, c color.RGBA) {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	for x := r.X; x < r.Right(); x++ {
		if x < 0 || x >= w {
			continue
		}
		for _, y := range []int{r.Y, r.Bottom() - 1} {
			if y >= 0 && y < h {
				img.SetRGBA(x, y, c)
			}
		}
	}
	for y := r.Y; y < r.Bottom(); y++ {
		if y < 0 || y >= h {
			continue
		}
		for _, x := range []int{r.X, r.Right() - 1} {
			if x >= 0 && x < w {
				img.SetRGBA(x, y, c)
			}
		}
	}
}
