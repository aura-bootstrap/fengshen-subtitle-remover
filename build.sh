#!/bin/sh
set -e
cd "$(dirname "$0")"
mkdir -p bin
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o bin/desub-linux-amd64 ./cmd/desub
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o bin/desub-windows-amd64.exe ./cmd/desub
echo "built bin/desub-linux-amd64 and bin/desub-windows-amd64.exe"
