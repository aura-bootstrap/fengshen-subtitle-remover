// Package ffx wraps ffmpeg/ffprobe subprocesses. All media IO (decode,
// encode, probing, scene detection) happens inside ffmpeg; this package
// only pumps raw frames through pipes.
package ffx

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

func FFmpeg() string  { return binName("ffmpeg") }
func FFprobe() string { return binName("ffprobe") }

func binName(def string) string {
	if v := os.Getenv("DESUB_" + strings.ToUpper(def)); v != "" {
		return v
	}
	return def
}

type MediaInfo struct {
	Path            string  `json:"path"`
	W               int     `json:"width"`
	H               int     `json:"height"`
	FPS             float64 `json:"fps"`
	Duration        float64 `json:"duration"`
	Rotation        int     `json:"rotation"`
	VCodec          string  `json:"vcodec"`
	PixFmt          string  `json:"pix_fmt,omitempty"`
	ColorRange      string  `json:"color_range,omitempty"`
	ColorSpace      string  `json:"color_space,omitempty"`
	ColorTransfer   string  `json:"color_transfer,omitempty"`
	ColorPrimaries  string  `json:"color_primaries,omitempty"`
	HasAudio        bool    `json:"has_audio"`
	SubtitleStreams int     `json:"subtitle_streams"`
	NBytes          int64   `json:"bytes"`
}

// ColorEncodeArgs returns encoder flags reproducing the input's colour
// metadata on the output (R9.3). Unset or "unknown" fields are omitted so the
// muxer default applies.
func (mi *MediaInfo) ColorEncodeArgs() []string {
	var args []string
	add := func(flag, v string) {
		if v != "" && v != "unknown" {
			args = append(args, flag, v)
		}
	}
	add("-color_range", mi.ColorRange)
	add("-colorspace", mi.ColorSpace)
	add("-color_trc", mi.ColorTransfer)
	add("-color_primaries", mi.ColorPrimaries)
	return args
}

type rawProbe struct {
	Streams []struct {
		Index          int    `json:"index"`
		CodecType      string `json:"codec_type"`
		CodecName      string `json:"codec_name"`
		Width          int    `json:"width"`
		Height         int    `json:"height"`
		PixFmt         string `json:"pix_fmt"`
		ColorRange     string `json:"color_range"`
		ColorSpace     string `json:"color_space"`
		ColorTransfer  string `json:"color_transfer"`
		ColorPrimaries string `json:"color_primaries"`
		AvgFrameRate   string `json:"avg_frame_rate"`
		RFrameRate     string `json:"r_frame_rate"`
		Duration       string `json:"duration"`
		SideDataList   []struct {
			Rotation float64 `json:"rotation"`
		} `json:"side_data_list"`
	} `json:"streams"`
	Format struct {
		Duration string `json:"duration"`
		BitRate  string `json:"bit_rate"`
		Size     string `json:"size"`
	} `json:"format"`
}

func Probe(path string) (*MediaInfo, error) {
	args := []string{"-v", "error", "-print_format", "json", "-show_streams", "-show_format", path}
	out, err := exec.Command(FFprobe(), args...).Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("ffprobe: %s", strings.TrimSpace(string(ee.Stderr)))
		}
		return nil, fmt.Errorf("ffprobe: %w", err)
	}
	var rp rawProbe
	if err := json.Unmarshal(out, &rp); err != nil {
		return nil, fmt.Errorf("ffprobe json: %w", err)
	}
	mi := &MediaInfo{Path: path}
	for _, s := range rp.Streams {
		switch s.CodecType {
		case "video":
			if mi.W != 0 {
				continue
			}
			w, h, rot := s.Width, s.Height, 0
			for _, sd := range s.SideDataList {
				if sd.Rotation != 0 {
					rot = int(sd.Rotation)
				}
			}
			rot = ((rot % 360) + 360) % 360
			if rot == 90 || rot == 270 {
				w, h = h, w
			}
			mi.W, mi.H, mi.Rotation = w, h, rot
			mi.VCodec = s.CodecName
			mi.PixFmt = s.PixFmt
			mi.ColorRange = s.ColorRange
			mi.ColorSpace = s.ColorSpace
			mi.ColorTransfer = s.ColorTransfer
			mi.ColorPrimaries = s.ColorPrimaries
			mi.FPS = parseFPS(s.AvgFrameRate)
			if mi.FPS == 0 {
				mi.FPS = parseFPS(s.RFrameRate)
			}
		case "audio":
			mi.HasAudio = true
		case "subtitle":
			mi.SubtitleStreams++
		}
	}
	if mi.W == 0 {
		return nil, fmt.Errorf("no video stream in %s", path)
	}
	if v, err := strconv.ParseFloat(rp.Format.Duration, 64); err == nil {
		mi.Duration = v
	}
	if v, err := strconv.ParseInt(rp.Format.Size, 10, 64); err == nil {
		mi.NBytes = v
	}
	return mi, nil
}

func parseFPS(s string) float64 {
	if s == "" || s == "0/0" {
		return 0
	}
	num, den, ok := strings.Cut(s, "/")
	if !ok {
		v, _ := strconv.ParseFloat(s, 64)
		return v
	}
	n, err1 := strconv.ParseFloat(num, 64)
	d, err2 := strconv.ParseFloat(den, 64)
	if err1 != nil || err2 != nil || d == 0 {
		return 0
	}
	return n / d
}

// BytesPer returns the byte size of one pixel for a rawvideo pix_fmt.
func BytesPer(pixfmt string) int {
	switch pixfmt {
	case "gray":
		return 1
	case "rgb24":
		return 3
	case "rgba":
		return 4
	}
	return 1
}

// FrameReader streams raw frames out of ffmpeg's stdout.
type FrameReader struct {
	cmd    *exec.Cmd
	out    io.ReadCloser
	size   int
	errBuf *bytes.Buffer
}

func NewFrameReader(input, vf string, w, h int, pixfmt string) (*FrameReader, error) {
	args := []string{"-hide_banner", "-nostdin", "-loglevel", "error", "-i", input, "-an", "-sn", "-dn"}
	if vf != "" {
		args = append(args, "-vf", vf)
	}
	args = append(args, "-f", "rawvideo", "-pix_fmt", pixfmt, "pipe:1")
	cmd := exec.Command(FFmpeg(), args...)
	errBuf := &bytes.Buffer{}
	cmd.Stderr = errBuf
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("ffmpeg start: %w", err)
	}
	return &FrameReader{cmd: cmd, out: out, size: w * h * BytesPer(pixfmt), errBuf: errBuf}, nil
}

func (r *FrameReader) FrameSize() int { return r.size }

// Next fills buf (must be FrameSize long). ok=false means clean EOF.
func (r *FrameReader) Next(buf []byte) (bool, error) {
	n, err := io.ReadFull(r.out, buf[:r.size])
	if err == io.EOF {
		return false, nil
	}
	if err == io.ErrUnexpectedEOF {
		return false, fmt.Errorf("ffmpeg: truncated frame (%d/%d bytes): %s", n, r.size, strings.TrimSpace(r.errBuf.String()))
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

func (r *FrameReader) Close() error {
	r.out.Close()
	if err := r.cmd.Wait(); err != nil {
		return fmt.Errorf("ffmpeg decode: %v: %s", err, strings.TrimSpace(r.errBuf.String()))
	}
	return nil
}

// Encoder is an ffmpeg process whose stdin accepts raw frames.
type Encoder struct {
	cmd    *exec.Cmd
	in     io.WriteCloser
	errBuf *bytes.Buffer
}

func NewEncoder(args []string) (*Encoder, error) {
	cmd := exec.Command(FFmpeg(), args...)
	errBuf := &bytes.Buffer{}
	cmd.Stderr = errBuf
	cmd.Stdout = io.Discard
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("ffmpeg start: %w", err)
	}
	return &Encoder{cmd: cmd, in: in, errBuf: errBuf}, nil
}

func (e *Encoder) Write(p []byte) (int, error) { return e.in.Write(p) }

func (e *Encoder) Close() error {
	if err := e.in.Close(); err != nil {
		return err
	}
	if err := e.cmd.Wait(); err != nil {
		return fmt.Errorf("ffmpeg encode: %v: %s", err, strings.TrimSpace(e.errBuf.String()))
	}
	return nil
}

// Run executes ffmpeg to completion, capturing stderr in error messages.
func Run(args ...string) error {
	cmd := exec.Command(FFmpeg(), args...)
	errBuf := &bytes.Buffer{}
	cmd.Stderr = errBuf
	cmd.Stdout = io.Discard
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg: %v: %s", err, strings.TrimSpace(errBuf.String()))
	}
	return nil
}

// SceneCuts returns timestamps (seconds) where the scene score exceeds thr.
func SceneCuts(input string, threshold float64) ([]float64, error) {
	vf := fmt.Sprintf("select='gt(scene,%g)',metadata=print:file=-", threshold)
	args := []string{"-hide_banner", "-nostdin", "-loglevel", "error", "-i", input, "-an", "-sn", "-dn", "-vf", vf, "-f", "null", "-"}
	cmd := exec.Command(FFmpeg(), args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg scene: %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	var cuts []float64
	lastPTS := -1.0
	for _, line := range strings.Split(stdout.String(), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "lavfi.scene_score=") {
			if lastPTS >= 0 {
				cuts = append(cuts, lastPTS)
			}
			continue
		}
		if i := strings.Index(line, "pts_time:"); i >= 0 {
			fields := strings.Fields(line[i+len("pts_time:"):])
			if len(fields) > 0 {
				if v, err := strconv.ParseFloat(fields[0], 64); err == nil {
					lastPTS = v
				}
			}
		}
	}
	return cuts, nil
}
