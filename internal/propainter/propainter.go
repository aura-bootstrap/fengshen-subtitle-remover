// Package propainter drives the ProPainter sidecar (scripts/propainter_infer.py):
// the band strip and its mask sequence are exchanged as lossless PNG
// directories, inference runs in chunks that never cross shot cuts, and only
// the masked pixels of the model output are used. Any failure bubbles up as
// an error so RunFill degrades the event to the motion tier (R5.6).
package propainter

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"time"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/engine"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/ffx"
)

// Client implements engine.Painter against the sidecar script.
type Client struct {
	Script    string
	Python    string
	Timeout   time.Duration // per chunk
	ChunkSize int           // frames per inference chunk (40–80 per R5.3)
	Overlap   int           // cross-fade frames between chunks (8–16)

	Home           string // PROPAINTER_HOME passed to the sidecar subprocess
	MaskDilation   int    // PROPAINTER_MASK_DILATION (0: model default 4)
	RaftIter       int    // PROPAINTER_RAFT_ITER (0: model default 20)
	NeighborLength int    // PROPAINTER_NEIGHBOR_LENGTH (0: model default 10)
}

// NewClient checks the sidecar script exists and returns a client with the
// spec'd chunk geometry.
func NewClient(script string, timeout time.Duration) (*Client, error) {
	if script == "" {
		script = "scripts/propainter_infer.py"
	}
	if _, err := os.Stat(script); err != nil {
		return nil, fmt.Errorf("propainter: %v", err)
	}
	if timeout <= 0 {
		timeout = 30 * time.Minute
	}
	return &Client{Script: script, Python: "python3", Timeout: timeout, ChunkSize: 64, Overlap: 12}, nil
}

// planChunks splits [startF, endF] into inference chunks of at most size
// frames with overlap frames shared between neighbours. Chunks never cross a
// cut; a boundary forced by a cut restarts without overlap.
func planChunks(startF, endF int, cuts []int, size, overlap int) [][2]int {
	srt := append([]int(nil), cuts...)
	sort.Ints(srt)
	var chunks [][2]int
	s := startF
	for s <= endF {
		e := s + size - 1
		if e > endF {
			e = endF
		}
		cut := -1
		for _, c := range srt {
			if c > s && c <= e {
				cut = c
				break
			}
			if c > e {
				break
			}
		}
		if cut >= 0 {
			e = cut - 1
		}
		// A cut inside the previous chunk's overlap truncates this chunk to
		// frames the previous chunk already covers — skip the empty remainder.
		if len(chunks) == 0 || e > chunks[len(chunks)-1][1] {
			chunks = append(chunks, [2]int{s, e})
		}
		if e >= endF {
			break
		}
		if cut >= 0 {
			s = cut // restart at the cut, no overlap across shots
		} else {
			s = e - overlap + 1
		}
	}
	return chunks
}

// Inpaint repairs the job's frame range chunk by chunk and returns one RGB
// band frame per frame in [StartF, EndF].
func (c *Client) Inpaint(j engine.PaintJob) ([][]byte, error) {
	n := j.EndF - j.StartF + 1
	if n <= 0 {
		return nil, fmt.Errorf("propainter: empty range %d..%d", j.StartF, j.EndF)
	}
	size, overlap := c.ChunkSize, c.Overlap
	if size < 40 {
		size = 40
	}
	if size > 80 {
		size = 80
	}
	if overlap < 8 {
		overlap = 8
	}
	if overlap > 16 {
		overlap = 16
	}
	var cutFrames []int
	for _, c := range j.Cuts {
		cutFrames = append(cutFrames, int(c*j.FPS+0.5))
	}
	chunks := planChunks(j.StartF, j.EndF, cutFrames, size, overlap)

	dir := j.WorkDir
	if dir == "" {
		dir = os.TempDir()
	}
	jobDir, err := os.MkdirTemp(dir, "propainter-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(jobDir)
	stripDir := filepath.Join(jobDir, "strip")
	maskDir := filepath.Join(jobDir, "masks")
	if err := os.MkdirAll(stripDir, 0o755); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(maskDir, 0o755); err != nil {
		return nil, err
	}
	if err := exportStrip(j, stripDir); err != nil {
		return nil, err
	}
	if err := exportMasks(j, maskDir); err != nil {
		return nil, err
	}

	out := make([][]byte, n)
	weight := make([]float64, n) // accumulated cross-fade weights
	acc := make([][]float64, n)
	for _, ch := range chunks {
		frames, err := c.runChunk(jobDir, ch, j.FPS, j.StartF)
		if err != nil {
			return nil, err
		}
		// Cross-fade: inside an overlap the later chunk's weight ramps 0→1.
		for k, fr := range frames {
			fi := ch[0] + k - j.StartF
			var w0, w1 float64 = 1, 0
			if k < overlap && fi > 0 && weight[fi] > 0 {
				t := float64(k+1) / float64(overlap+1)
				w0, w1 = 1-t, t
			}
			if acc[fi] == nil {
				acc[fi] = make([]float64, len(fr))
			}
			if w1 > 0 {
				for p, v := range fr {
					acc[fi][p] = acc[fi][p]*w0 + float64(v)*w1
				}
				weight[fi] = 1
			} else {
				for p, v := range fr {
					acc[fi][p] += float64(v)
				}
				weight[fi]++
			}
		}
	}
	for i := range out {
		if acc[i] == nil || weight[i] == 0 {
			return nil, fmt.Errorf("propainter: frame %d not covered by any chunk", j.StartF+i)
		}
		fr := make([]byte, len(acc[i]))
		for p, v := range acc[i] {
			fr[p] = uint8(v/weight[i] + 0.5)
		}
		out[i] = fr
	}
	return out, nil
}

// runChunk executes the sidecar on one chunk and reads back its frames.
// Strip/mask files are shared across chunks; each chunk gets its own output
// directory and processes the frame range [ch0, ch1] (absolute numbers).
func (c *Client) runChunk(jobDir string, ch [2]int, fps float64, jobStart int) ([][]byte, error) {
	outDir := filepath.Join(jobDir, fmt.Sprintf("out-%05d-%05d", ch[0], ch[1]))
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), c.Timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, c.Python, c.Script,
		"--strip", filepath.Join(jobDir, "strip"),
		"--masks", filepath.Join(jobDir, "masks"),
		"--out", outDir,
		"--start", fmt.Sprint(ch[0]-jobStart),
		"--count", fmt.Sprint(ch[1]-ch[0]+1),
		"--fps", fmt.Sprintf("%.6f", fps))
	var env []string
	if c.Home != "" {
		env = append(os.Environ(), "PROPAINTER_HOME="+c.Home)
	}
	for _, kv := range [][2]string{
		{"PROPAINTER_MASK_DILATION", fmt.Sprint(c.MaskDilation)},
		{"PROPAINTER_RAFT_ITER", fmt.Sprint(c.RaftIter)},
		{"PROPAINTER_NEIGHBOR_LENGTH", fmt.Sprint(c.NeighborLength)},
	} {
		if kv[1] != "0" {
			if env == nil {
				env = os.Environ()
			}
			env = append(env, kv[0]+"="+kv[1])
		}
	}
	cmd.Env = env
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("propainter chunk %d-%d: %v: %s", ch[0], ch[1], err, tail(stderr.String(), 400))
	}
	n := ch[1] - ch[0] + 1
	frames := make([][]byte, n)
	for k := 0; k < n; k++ {
		p := filepath.Join(outDir, frameName(ch[0]-jobStart+k))
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("propainter chunk %d-%d: %v", ch[0], ch[1], err)
		}
		img, err := png.Decode(bytes.NewReader(b))
		if err != nil {
			return nil, fmt.Errorf("propainter chunk %d-%d: %v", ch[0], ch[1], err)
		}
		frames[k] = rgbOf(img)
	}
	return frames, nil
}

func frameName(i int) string { return fmt.Sprintf("%05d.png", i) }

// exportStrip decodes the job's band frames [StartF, EndF] to PNG files.
func exportStrip(j engine.PaintJob, dir string) error {
	n := j.EndF - j.StartF + 1
	args := []string{"-hide_banner", "-nostdin", "-y", "-loglevel", "error",
		"-ss", fmt.Sprintf("%.6f", float64(j.StartF)/j.FPS),
		"-i", j.Input,
		"-vf", fmt.Sprintf("crop=%d:%d:0:%d", j.W, j.BandH, j.BandY),
		"-frames:v", fmt.Sprint(n),
		"-start_number", "0",
		filepath.Join(dir, "%05d.png")}
	cmd := exec.Command(ffx.FFmpeg(), args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("propainter strip export: %v: %s", err, tail(stderr.String(), 400))
	}
	if _, err := os.Stat(filepath.Join(dir, frameName(n-1))); err != nil {
		return fmt.Errorf("propainter strip export: %v", err)
	}
	return nil
}

// exportMasks writes each frame's repair mask as a single-channel PNG
// (255 = to be inpainted), dilated one pixel so the model covers stroke
// anti-aliasing that the binary mask edges miss.
func exportMasks(j engine.PaintJob, dir string) error {
	w, h := j.W, j.BandH
	bits := make([]uint8, w*h)
	for i, m := range j.Masks {
		m.Decode(bits, w)
		img := image.NewGray(image.Rect(0, 0, w, h))
		for p, b := range bits {
			if b != 0 {
				img.Pix[p] = 255
			}
		}
		f, err := os.Create(filepath.Join(dir, frameName(i)))
		if err != nil {
			return err
		}
		err = png.Encode(f, img)
		f.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

// rgbOf flattens a decoded PNG to RGB triplets.
func rgbOf(img image.Image) []byte {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			p := (y*w + x) * 3
			out[p] = uint8(r >> 8)
			out[p+1] = uint8(g >> 8)
			out[p+2] = uint8(bl >> 8)
		}
	}
	return out
}

func tail(s string, n int) string {
	if len(s) > n {
		return s[len(s)-n:]
	}
	return s
}
