package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/manifest"
	"github.com/aura-bootstrap/fengshen-subtitle-remover/internal/report"
)

// cmdRisk prints the high-risk segment list from a run manifest (R6): JSON by
// default, HH:MM:SS.mmm timecode text with --timecodes (R6.4).
func cmdRisk(args []string) error {
	fs := flag.NewFlagSet("risk", flag.ExitOnError)
	fs.Usage = usage
	manifestPath := fs.String("manifest", "", "run manifest written by remove --manifest (required)")
	timecodes := fs.Bool("timecodes", false, "print HH:MM:SS.mmm text instead of JSON")
	covThr := fs.Float64("risk-coverage", report.DefaultCoverage, "coverage threshold below which a segment is listed")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 0 || *manifestPath == "" {
		return fmt.Errorf("usage: desub risk --manifest m.json [--timecodes] [--risk-coverage 0.35]")
	}
	m, err := manifest.Read(*manifestPath)
	if err != nil {
		return err
	}
	items := report.RiskList(m, nil, *covThr)
	if *timecodes {
		return report.WriteTimecodes(os.Stdout, items)
	}
	if items == nil {
		items = []report.RiskItem{}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(items)
}
