// Package mask stores per-frame 0/1 masks as row-run RLE so a full video's
// worth of masks stays small in memory.
package mask

import "sort"

type Frame struct {
	RLE []uint16 // triplets: y, x0, x1 (inclusive)
}

func Encode(bits []uint8, w, h int) Frame {
	var rle []uint16
	for y := 0; y < h; y++ {
		row := bits[y*w : (y+1)*w]
		for x := 0; x < w; {
			if row[x] == 0 {
				x++
				continue
			}
			x0 := x
			for x < w && row[x] != 0 {
				x++
			}
			rle = append(rle, uint16(y), uint16(x0), uint16(x-1))
		}
	}
	return Frame{RLE: rle}
}

func (f Frame) Empty() bool { return len(f.RLE) == 0 }

// Union merges two masks; both RLEs are sorted row-major as Encode emits
// them.
func (f Frame) Union(g Frame) Frame {
	var out []uint16
	i, j := 0, 0
	for i+2 < len(f.RLE) || j+2 < len(g.RLE) {
		var y uint16
		switch {
		case i+2 >= len(f.RLE):
			y = g.RLE[j]
		case j+2 >= len(g.RLE):
			y = f.RLE[i]
		default:
			y = min(f.RLE[i], g.RLE[j])
		}
		type run struct{ a, b uint16 }
		runs := make([]run, 0, 4)
		for i+2 < len(f.RLE) && f.RLE[i] == y {
			runs = append(runs, run{f.RLE[i+1], f.RLE[i+2]})
			i += 3
		}
		for j+2 < len(g.RLE) && g.RLE[j] == y {
			runs = append(runs, run{g.RLE[j+1], g.RLE[j+2]})
			j += 3
		}
		sort.Slice(runs, func(p, q int) bool {
			if runs[p].a != runs[q].a {
				return runs[p].a < runs[q].a
			}
			return runs[p].b < runs[q].b
		})
		cur := runs[0]
		for _, r := range runs[1:] {
			if r.a <= cur.b+1 {
				if r.b > cur.b {
					cur.b = r.b
				}
				continue
			}
			out = append(out, y, cur.a, cur.b)
			cur = r
		}
		out = append(out, y, cur.a, cur.b)
	}
	return Frame{RLE: out}
}

// ClipRows keeps only the runs on rows [y0,y1).
func (f Frame) ClipRows(y0, y1 int) Frame {
	var out []uint16
	for i := 0; i+2 < len(f.RLE); i += 3 {
		y := int(f.RLE[i])
		if y < y0 || y >= y1 {
			continue
		}
		out = append(out, f.RLE[i], f.RLE[i+1], f.RLE[i+2])
	}
	return Frame{RLE: out}
}

func (f Frame) Area() int {
	a := 0
	for i := 0; i+2 < len(f.RLE); i += 3 {
		a += int(f.RLE[i+2]) - int(f.RLE[i+1]) + 1
	}
	return a
}

// Decode fills dst (w*h, pre-zeroed or not) with this mask.
func (f Frame) Decode(dst []uint8, w int) {
	clear(dst)
	for i := 0; i+2 < len(f.RLE); i += 3 {
		y, x0, x1 := int(f.RLE[i]), int(f.RLE[i+1]), int(f.RLE[i+2])
		off := y*w + x0
		for x := x0; x <= x1; x++ {
			dst[off] = 1
			off++
		}
	}
}
