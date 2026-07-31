#!/usr/bin/env sh

set -e

conda run -n waft python custom.py --cfg config/a1/tar.json --ckpt weights/tar-c-t.pth --device cuda