// Package vlmqc drives the VLM QC sidecar (scripts/vlm_qc.py): the
// verify-stage residue boxes are re-judged by a vision language model
// behind an OpenAI-compatible endpoint (ollama / vLLM / LM Studio), so
// texture false positives can be told apart from true subtitle residue.
// Any sidecar failure bubbles up as an error; the pipeline degrades to
// threshold-only residue counts with a warning.
package vlmqc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// Box is one verify-stage residue box in absolute frame coordinates.
type Box struct {
	Frame int     `json:"frame"`
	Time  float64 `json:"time"`
	X     int     `json:"x"`
	Y     int     `json:"y"`
	W     int     `json:"w"`
	H     int     `json:"h"`
}

// Verdict is a Box plus the VLM's judgement. Residue is nil when the box
// could not be judged (per-box errors never abort the batch).
type Verdict struct {
	Box
	Residue    *bool   `json:"residue"`
	Confidence float64 `json:"confidence,omitempty"`
	Note       string  `json:"note,omitempty"`
}

// Client invokes the sidecar script.
type Client struct {
	Script   string
	Python   string
	Endpoint string
	Model    string
	APIKey   string
	Timeout  time.Duration
}

// NewClient checks the sidecar script exists and returns a client.
func NewClient(script, endpoint, model, apiKey string) (*Client, error) {
	if script == "" {
		script = "scripts/vlm_qc.py"
	}
	if _, err := os.Stat(script); err != nil {
		return nil, fmt.Errorf("vlm-qc: %v", err)
	}
	if endpoint == "" {
		endpoint = "http://host.docker.internal:11434/v1"
	}
	if model == "" {
		model = "qwen2.5vl:7b"
	}
	return &Client{
		Script: script, Python: "python3",
		Endpoint: endpoint, Model: model, APIKey: apiKey,
		Timeout: 10 * time.Minute,
	}, nil
}

// Review sends every residue box to the VLM and returns one verdict per
// box, in input order.
func (c *Client) Review(video string, boxes []Box) ([]Verdict, error) {
	if len(boxes) == 0 {
		return nil, nil
	}
	dir, err := os.MkdirTemp("", "vlmqc-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	inPath := filepath.Join(dir, "residue.json")
	outPath := filepath.Join(dir, "verdicts.json")
	raw, err := json.Marshal(boxes)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(inPath, raw, 0o644); err != nil {
		return nil, err
	}
	perBox := int(c.Timeout.Seconds())/len(boxes) + 30
	cmd := exec.Command(c.Python, c.Script,
		"--video", video,
		"--residue", inPath,
		"--out", outPath,
		"--endpoint", c.Endpoint,
		"--model", c.Model,
		"--api-key", c.APIKey,
		"--timeout", fmt.Sprint(perBox))
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("vlm-qc: %v: %s", err, tail(stderr.String(), 300))
	}
	data, err := os.ReadFile(outPath)
	if err != nil {
		return nil, fmt.Errorf("vlm-qc: %v", err)
	}
	var verdicts []Verdict
	if err := json.Unmarshal(data, &verdicts); err != nil {
		return nil, fmt.Errorf("vlm-qc: verdicts: %v", err)
	}
	if len(verdicts) != len(boxes) {
		return nil, fmt.Errorf("vlm-qc: got %d verdicts for %d boxes", len(verdicts), len(boxes))
	}
	return verdicts, nil
}

func tail(s string, n int) string {
	if len(s) > n {
		return s[len(s)-n:]
	}
	return s
}
