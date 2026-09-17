// Package grain hides the repair: a filled patch is often suspiciously clean
// next to compressed video, so Match injects grain at the locally estimated
// noise level, aligns first-order chroma statistics with the surroundings,
// and (on blocky inputs) re-introduces matching 8x8 boundary steps. All of it
// applies only inside the feathered mask (R10.4).
package grain

import (
	"math"
	"sort"
)

// Match adjusts fill (the repaired frame) inside the feathered mask:
// maskBits is the feather alpha (0 = untouched, 255 = fully repaired). src is
// the original frame used to measure surrounding texture. blocky forces the
// 8x8 boundary-step matching; the step strength is always measured, so a
// clean source gets none either way.
func Match(fill []byte, w, h int, maskBits []uint8, src []byte, blocky bool) {
	ring := ringMask(maskBits, w, h, 14)
	noise := noiseSigma(src, ring, w, h)
	if math.Max(noise[0], math.Max(noise[1], noise[2])) > 0.3 {
		inject(fill, w, h, maskBits, noise)
	}
	alignChroma(fill, w, h, maskBits, src, ring)
	if blocky {
		matchBlocks(fill, w, h, maskBits, src, ring)
	}
}

// ringMask marks pixels that are outside the repaired area but within `rad`
// of it — the texture reference neighbourhood.
func ringMask(maskBits []uint8, w, h, rad int) []uint8 {
	inside := make([]uint8, w*h)
	for p, a := range maskBits {
		if a > 32 {
			inside[p] = 1
		}
	}
	ring := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if inside[y*w+x] != 0 {
				continue
			}
			// cheap square-distance check against the repair area
			found := false
			for dy := -rad; dy <= rad && !found; dy += 2 {
				for dx := -rad; dx <= rad && !found; dx += 2 {
					yy, xx := y+dy, x+dx
					if yy >= 0 && yy < h && xx >= 0 && xx < w && inside[yy*w+xx] != 0 {
						found = true
					}
				}
			}
			if found {
				ring[y*w+x] = 1
			}
		}
	}
	return ring
}

// noiseSigma estimates per-channel noise on the ring via the Laplacian MAD:
// for i.i.d. noise the kernel 4c − l − r − u − d has variance 20σ², and MAD
// resists the texture edges leaking into the estimate.
func noiseSigma(src []byte, ring []uint8, w, h int) [3]float64 {
	var out [3]float64
	for c := 0; c < 3; c++ {
		var laps []float64
		for y := 2; y < h-2; y++ {
			for x := 2; x < w-2; x++ {
				p := y*w + x
				if ring[p] == 0 {
					continue
				}
				lap := 4*float64(src[p*3+c]) - float64(src[(p-1)*3+c]) - float64(src[(p+1)*3+c]) -
					float64(src[(p-w)*3+c]) - float64(src[(p+w)*3+c])
				laps = append(laps, math.Abs(lap))
			}
		}
		if len(laps) < 64 {
			continue
		}
		sort.Float64s(laps)
		med := laps[len(laps)/2]
		out[c] = med / 0.6745 / math.Sqrt(20)
	}
	return out
}

// inject adds deterministic Gaussian-ish grain at the measured sigma, scaled
// by the feather alpha. The pattern is a pure function of the pixel position
// so re-runs are reproducible.
func inject(fill []byte, w, h int, maskBits []uint8, sigma [3]float64) {
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			p := y*w + x
			a := maskBits[p]
			if a == 0 {
				continue
			}
			aw := float64(a) / 255
			for c := 0; c < 3; c++ {
				n := gauss(uint32(p*3+c), uint32(c)) * sigma[c] * aw
				v := float64(fill[p*3+c]) + n
				if v < 0 {
					v = 0
				}
				if v > 255 {
					v = 255
				}
				fill[p*3+c] = uint8(v + 0.5)
			}
		}
	}
}

// gauss approximates N(0,1) from the sum of four position-hashed uniforms
// (variance 1/3 each normalised below).
func gauss(seed, salt uint32) float64 {
	var s float64
	for k := uint32(0); k < 4; k++ {
		z := seed*2654435761 + salt*2246822519 + k*3266489917
		z ^= z >> 15
		z *= 2246822519
		z ^= z >> 13
		s += float64(z>>8) / 16777216
	}
	return (s - 2) * math.Sqrt(3)
}

// alignChroma pulls the repaired area's chroma mean and variance towards the
// ring's (U = B − luma, V = R − luma), weighted by the feather alpha. Luma is
// left alone — touching it would disturb the repaired texture itself.
func alignChroma(fill []byte, w, h int, maskBits []uint8, src, ring []uint8) {
	var muS, muF, vaS, vaF [2]float64
	var nS, nF float64
	for p := 0; p < w*h; p++ {
		r, g, b := float64(src[p*3]), float64(src[p*3+1]), float64(src[p*3+2])
		y := 0.299*r + 0.587*g + 0.114*b
		if ring[p] != 0 {
			muS[0] += b - y
			muS[1] += r - y
			nS++
		}
	}
	if nS < 64 {
		return
	}
	muS[0] /= nS
	muS[1] /= nS
	for p := 0; p < w*h; p++ {
		if ring[p] == 0 {
			continue
		}
		r, g, b := float64(src[p*3]), float64(src[p*3+1]), float64(src[p*3+2])
		y := 0.299*r + 0.587*g + 0.114*b
		vaS[0] += (b - y - muS[0]) * (b - y - muS[0])
		vaS[1] += (r - y - muS[1]) * (r - y - muS[1])
	}
	vaS[0] /= nS
	vaS[1] /= nS
	for p := 0; p < w*h; p++ {
		if maskBits[p] < 128 {
			continue
		}
		r, g, b := float64(fill[p*3]), float64(fill[p*3+1]), float64(fill[p*3+2])
		y := 0.299*r + 0.587*g + 0.114*b
		muF[0] += b - y
		muF[1] += r - y
		nF++
	}
	if nF < 64 {
		return
	}
	muF[0] /= nF
	muF[1] /= nF
	for p := 0; p < w*h; p++ {
		if maskBits[p] < 128 {
			continue
		}
		r, g, b := float64(fill[p*3]), float64(fill[p*3+1]), float64(fill[p*3+2])
		y := 0.299*r + 0.587*g + 0.114*b
		vaF[0] += (b - y - muF[0]) * (b - y - muF[0])
		vaF[1] += (r - y - muF[1]) * (r - y - muF[1])
	}
	vaF[0] /= nF
	vaF[1] /= nF
	for p := 0; p < w*h; p++ {
		a := maskBits[p]
		if a == 0 {
			continue
		}
		aw := float64(a) / 255
		r, g, b := float64(fill[p*3]), float64(fill[p*3+1]), float64(fill[p*3+2])
		y := 0.299*r + 0.587*g + 0.114*b
		u, v := b-y, r-y
		for c := 0; c < 2; c++ {
			scale := 1.0
			if vaF[c] > 1e-3 {
				scale = math.Sqrt(vaS[c] / vaF[c])
				if scale > 2 {
					scale = 2
				}
				if scale < 0.5 {
					scale = 0.5
				}
			}
			t := (chromaOf(u, v, c)-muF[c])*scale + muS[c]
			switch c {
			case 0:
				u += aw * (t - u)
			case 1:
				v += aw * (t - v)
			}
		}
		fill[p*3] = clamp8(y + v)
		fill[p*3+2] = clamp8(y + u)
	}
}

func chromaOf(u, v float64, c int) float64 {
	if c == 0 {
		return u
	}
	return v
}

// matchBlocks measures the 8-aligned boundary step on the ring and
// re-creates it inside the repair: highly compressed sources show a step at
// every block edge, and a smooth patch stands out without one.
func matchBlocks(fill []byte, w, h int, maskBits []uint8, src, ring []uint8) {
	bx, nx := boundaryStep(src, ring, w, h, true)
	by, ny := boundaryStep(src, ring, w, h, false)
	excess := math.Max(bx-nx, by-ny)
	if excess < 1.5 {
		return // no measurable blockiness on the source
	}
	step := excess / 2
	for y := 1; y < h; y++ {
		for x := 1; x < w; x++ {
			p := y*w + x
			a := maskBits[p]
			if a == 0 {
				continue
			}
			aw := float64(a) / 255
			var d float64
			if x%8 == 0 {
				d += step
			}
			if y%8 == 0 {
				d += step
			}
			if d == 0 {
				continue
			}
			if (x/8+y/8)%2 == 0 {
				d = -d
			}
			for c := 0; c < 3; c++ {
				v := float64(fill[p*3+c]) + d*aw
				fill[p*3+c] = clamp8(v)
			}
		}
	}
}

// boundaryStep returns the mean absolute single-pixel difference at
// 8-aligned boundaries versus mid-block positions.
func boundaryStep(src []byte, ring []uint8, w, h int, vertical bool) (boundary, mid float64) {
	var nb, nm float64
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			p := y*w + x
			if ring[p] == 0 {
				continue
			}
			var q int
			var pos int
			if vertical {
				q = p - 1
				pos = x
			} else {
				q = p - w
				pos = y
			}
			var d float64
			for c := 0; c < 3; c++ {
				d += math.Abs(float64(src[p*3+c]) - float64(src[q*3+c]))
			}
			switch pos % 8 {
			case 0:
				boundary += d
				nb++
			case 4:
				mid += d
				nm++
			}
		}
	}
	if nb > 0 {
		boundary /= nb
	}
	if nm > 0 {
		mid /= nm
	}
	return
}

func clamp8(v float64) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5)
}
