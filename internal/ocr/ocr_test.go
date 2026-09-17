package ocr

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/imgx"
)

func TestDetectFramesRoundTrip(t *testing.T) {
	var gotReq Request
	c := NewStubClient(func(_ context.Context, req Request) (Response, error) {
		gotReq = req
		return Response{Boxes: []Box{
			{Frame: 12, Rect: imgx.Rect{X: 100, Y: 90, W: 200, H: 50}, Score: 0.9},
			{Frame: 24, Rect: imgx.Rect{X: 120, Y: 92, W: 180, H: 48}, Score: 0.8},
		}}, nil
	})
	boxes, err := c.DetectFrames("/in.mp4", 742, 538, []int{12, 24})
	if err != nil {
		t.Fatal(err)
	}
	if gotReq.Input != "/in.mp4" || gotReq.BandY != 742 || gotReq.BandH != 538 {
		t.Fatalf("request garbled: %+v", gotReq)
	}
	if len(gotReq.Frames) != 2 || gotReq.Frames[0] != 12 || gotReq.Frames[1] != 24 {
		t.Fatalf("frames garbled: %v", gotReq.Frames)
	}
	if len(boxes) != 2 || boxes[0].Rect.W != 200 || boxes[1].Frame != 24 {
		t.Fatalf("boxes garbled: %+v", boxes)
	}
}

func TestDetectFramesSidecarError(t *testing.T) {
	c := NewStubClient(func(_ context.Context, _ Request) (Response, error) {
		return Response{Error: "import: no module named easyocr"}, nil
	})
	if _, err := c.DetectFrames("/in.mp4", 0, 100, []int{0}); err == nil {
		t.Fatal("sidecar error not surfaced")
	}
}

func TestDetectFramesExecFailure(t *testing.T) {
	c := NewStubClient(func(_ context.Context, _ Request) (Response, error) {
		return Response{}, errors.New("exit status 2")
	})
	if _, err := c.DetectFrames("/in.mp4", 0, 100, []int{0}); err == nil {
		t.Fatal("exec failure not surfaced")
	}
}

func TestDetectFramesTimeout(t *testing.T) {
	c := &Client{Timeout: 30 * time.Millisecond}
	c.Exec = func(ctx context.Context, _, _ string, _ Request) (Response, error) {
		<-ctx.Done()
		return Response{}, ctx.Err()
	}
	if _, err := c.DetectFrames("/in.mp4", 0, 100, []int{0}); err == nil {
		t.Fatal("timeout not surfaced")
	}
}

func TestDetectFramesEmpty(t *testing.T) {
	c := NewStubClient(func(_ context.Context, _ Request) (Response, error) {
		t.Error("sidecar called for an empty frame list")
		return Response{}, nil
	})
	boxes, err := c.DetectFrames("/in.mp4", 0, 100, nil)
	if err != nil || len(boxes) != 0 {
		t.Fatalf("empty request: boxes=%v err=%v", boxes, err)
	}
}
