package grain

import (
	"math"
	"testing"
)

func prand(seed uint32) float64 {
	z := seed*2654435761 + 97
	z ^= z >> 15
	z *= 2246822519
	z ^= z >> 13
	return float64(z>>8)/16777216 - 0.5
}

func stddev(v []float64) float64 {
	var m float64
	for _, x := range v {
		m += x
	}
	m /= float64(len(v))
	var s float64
	for _, x := range v {
		s += (x - m) * (x - m)
	}
	return math.Sqrt(s / float64(len(v)))
}

func regionStd(f []byte, w, c, x0, y0, x1, y1 int) float64 {
	var v []float64
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			v = append(v, float64(f[(y*w+x)*3+c]))
		}
	}
	return stddev(v)
}

// Grain injected into a flat repair reaches roughly the source noise level.
func TestInjectMatchesSigma(t *testing.T) {
	const w, h = 128, 128
	src := make([]byte, w*h*3)
	for p := 0; p < w*h; p++ {
		for c := 0; c < 3; c++ {
			src[p*3+c] = uint8(128 + 20*prand(uint32(p*3+c+1)) + 0.5)
		}
	}
	srcStd := regionStd(src, w, 1, 4, 4, w-4, h-4)
	fill := make([]byte, w*h*3)
	for i := range fill {
		fill[i] = 128
	}
	maskBits := make([]uint8, w*h)
	for y := 48; y < 80; y++ {
		for x := 48; x < 80; x++ {
			maskBits[y*w+x] = 255
		}
	}
	Match(fill, w, h, maskBits, src, false)
	got := regionStd(fill, w, 1, 50, 50, 78, 78)
	if got < srcStd*0.5 || got > srcStd*1.6 {
		t.Fatalf("repair std %.2f, source std %.2f", got, srcStd)
	}
	if out := regionStd(fill, w, 1, 4, 4, 40, 40); out > 0.01 {
		t.Fatalf("outside region changed: std %.3f", out)
	}
}

// Pixels outside the feather (alpha 0) are never touched (R10.4).
func TestFeatherOutsideUntouched(t *testing.T) {
	const w, h = 64, 64
	src := make([]byte, w*h*3)
	for p := 0; p < w*h; p++ {
		for c := 0; c < 3; c++ {
			src[p*3+c] = uint8(120 + 24*prand(uint32(p*3+c)) + 0.5)
		}
	}
	fill := make([]byte, w*h*3)
	copy(fill, src)
	before := append([]byte(nil), fill...)
	Match(fill, w, h, make([]uint8, w*h), src, true)
	for i := range fill {
		if fill[i] != before[i] {
			t.Fatalf("pixel %d changed with empty mask", i)
		}
	}
}

// A red-tinted repair over a blue-tinted surround moves its chroma towards
// the surround (R10.2).
func TestChromaAlign(t *testing.T) {
	const w, h = 128, 128
	src := make([]byte, w*h*3)
	for p := 0; p < w*h; p++ {
		y := 100 + 10*prand(uint32(p))
		src[p*3] = uint8(y + 0.5)        // R ≈ Y
		src[p*3+1] = uint8(y + 0.5)      // G ≈ Y
		src[p*3+2] = uint8(y + 20 + 0.5) // B = Y + 20 (blue surround)
	}
	fill := make([]byte, w*h*3)
	for p := 0; p < w*h; p++ {
		y := 100.0
		fill[p*3] = uint8(y + 15 + 0.5) // R = Y + 15 (red repair)
		fill[p*3+1] = uint8(y + 0.5)
		fill[p*3+2] = uint8(y + 0.5)
	}
	maskBits := make([]uint8, w*h)
	for y := 48; y < 80; y++ {
		for x := 48; x < 80; x++ {
			maskBits[y*w+x] = 255
		}
	}
	chromaB := func(f []byte) float64 {
		var s float64
		var n int
		for y := 52; y < 76; y++ {
			for x := 52; x < 76; x++ {
				p := (y*w + x) * 3
				yy := 0.299*float64(f[p]) + 0.587*float64(f[p+1]) + 0.114*float64(f[p+2])
				s += float64(f[p+2]) - yy
				n++
			}
		}
		return s / float64(n)
	}
	pre := chromaB(fill)
	Match(fill, w, h, maskBits, src, false)
	post := chromaB(fill)
	if post < pre+6 {
		t.Fatalf("chroma not aligned: B−Y %.1f -> %.1f, want move towards 20", pre, post)
	}
}

// On a blocky source the repair regains an 8-aligned boundary step (R10.3).
func TestBlockyMatch(t *testing.T) {
	const w, h = 128, 128
	src := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			p := (y*w + x) * 3
			v := 100 + 6*prand(uint32(y*w+x))
			if (x/8)%2 == 0 {
				v += 8
			}
			for c := 0; c < 3; c++ {
				src[p+c] = uint8(v + 0.5)
			}
		}
	}
	fill := make([]byte, w*h*3)
	for i := range fill {
		fill[i] = 104
	}
	maskBits := make([]uint8, w*h)
	for y := 48; y < 80; y++ {
		for x := 48; x < 80; x++ {
			maskBits[y*w+x] = 255
		}
	}
	boundary := func(f []byte) (b, m float64) {
		for y := 56; y < 72; y++ {
			for x := 48; x < 79; x++ {
				d := 0.0
				for c := 0; c < 3; c++ {
					d += math.Abs(float64(f[(y*w+x+1)*3+c]) - float64(f[(y*w+x)*3+c]))
				}
				if (x+1)%8 == 0 {
					b += d
				}
				if (x+1)%8 == 4 {
					m += d
				}
			}
		}
		return b / 16, m / 16
	}
	bBefore, _ := boundary(fill)
	Match(fill, w, h, maskBits, src, true)
	bAfter, mAfter := boundary(fill)
	if bAfter <= bBefore+2 || bAfter <= mAfter {
		t.Fatalf("no boundary step: before %.2f after %.2f mid %.2f", bBefore, bAfter, mAfter)
	}
}
