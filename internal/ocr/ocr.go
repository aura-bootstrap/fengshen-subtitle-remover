// Package ocr is the client side of the OCR sidecar (scripts/ocr_boxes.py):
// the Go pipeline sends a JSON request (input path, band, frame numbers) to
// the script's stdin and reads back JSON detection boxes. The sidecar is an
// optional recall booster — any failure or timeout is reported as an error
// and the caller degrades to the classical detector alone.
package ocr

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
)

// Box is one OCR detection in band coordinates.
type Box struct {
	Frame int       `json:"frame"`
	Rect  imgx.Rect `json:"rect"`
	Score float64   `json:"score"`
}

// Request is the JSON payload sent to the sidecar's stdin.
type Request struct {
	Input  string `json:"input"`
	BandY  int    `json:"band_y"`
	BandH  int    `json:"band_h"`
	Frames []int  `json:"frames"`
}

// Response is the JSON payload read back from the sidecar's stdout.
type Response struct {
	Boxes []Box  `json:"boxes"`
	Error string `json:"error,omitempty"`
}

type Client struct {
	Script  string
	Python  string
	Timeout time.Duration
	// Exec runs one request against the sidecar; nil uses the real
	// subprocess. Tests inject a stub here.
	Exec func(ctx context.Context, python, script string, req Request) (Response, error)
}

// NewClient validates the script path and returns a client with the default
// subprocess transport.
func NewClient(script string, timeout time.Duration) (*Client, error) {
	if script == "" {
		return nil, errors.New("ocr: empty script path")
	}
	if _, err := os.Stat(script); err != nil {
		return nil, fmt.Errorf("ocr: %w", err)
	}
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	return &Client{Script: script, Python: "python3", Timeout: timeout}, nil
}

// NewStubClient returns a client whose transport is fn; used by tests and by
// callers that want to feed canned OCR results.
func NewStubClient(fn func(ctx context.Context, req Request) (Response, error)) *Client {
	return &Client{Timeout: time.Minute, Exec: func(_ context.Context, _, _ string, req Request) (Response, error) {
		return fn(context.Background(), req)
	}}
}

// DetectFrames runs the sidecar over the given frame numbers and returns the
// detections in band coordinates. A sidecar-reported error, a non-zero exit
// or a timeout all surface as errors.
func (c *Client) DetectFrames(input string, bandY, bandH int, frames []int) ([]Box, error) {
	if len(frames) == 0 {
		return nil, nil
	}
	req := Request{Input: input, BandY: bandY, BandH: bandH, Frames: frames}
	ctx, cancel := context.WithTimeout(context.Background(), c.Timeout)
	defer cancel()
	ex := c.Exec
	if ex == nil {
		ex = runSubprocess
	}
	resp, err := ex(ctx, c.Python, c.Script, req)
	if err != nil {
		return nil, err
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("ocr sidecar: %s", resp.Error)
	}
	return resp.Boxes, nil
}

func runSubprocess(ctx context.Context, python, script string, req Request) (Response, error) {
	var resp Response
	if python == "" {
		python = "python3"
	}
	cmd := exec.CommandContext(ctx, python, script)
	in, err := cmd.StdinPipe()
	if err != nil {
		return resp, err
	}
	var errBuf bytes.Buffer
	cmd.Stderr = &errBuf
	out, err := cmd.StdoutPipe()
	if err != nil {
		return resp, err
	}
	if err := cmd.Start(); err != nil {
		return resp, fmt.Errorf("ocr sidecar start: %w", err)
	}
	werr := make(chan error, 1)
	go func() {
		werr <- json.NewEncoder(in).Encode(req)
		in.Close()
	}()
	var decErr error
	if err := json.NewDecoder(out).Decode(&resp); err != nil {
		decErr = fmt.Errorf("ocr sidecar response: %w", err)
	}
	if err := <-werr; err != nil && decErr == nil {
		decErr = fmt.Errorf("ocr sidecar request: %w", err)
	}
	if err := cmd.Wait(); err != nil && decErr == nil {
		if ctx.Err() == context.DeadlineExceeded {
			decErr = fmt.Errorf("ocr sidecar: timeout")
		} else {
			decErr = fmt.Errorf("ocr sidecar exit: %v: %s", err, strings.TrimSpace(errBuf.String()))
		}
	}
	return resp, decErr
}

// Close is a no-op: the transport is one process per request, so there is no
// persistent process to shut down. It exists so callers can treat the client
// uniformly if a persistent mode is ever added.
func (c *Client) Close() error { return nil }
