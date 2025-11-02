#!/bin/sh
# Entry point: install any wheel files found under /app/requirements into /app/vendor
# and then exec the main process. This keeps large wheel files out of the image.

set -e

VENDOR_DIR="/app/vendor"
REQ_DIR="/app/requirements"

mkdir -p "$VENDOR_DIR"
chown -R $(id -u):$(id -g) "$VENDOR_DIR" || true

echo "[entrypoint] PYTHONPATH=/app/vendor:/usr/local/lib/python3.11/site-packages"
export PYTHONPATH="$VENDOR_DIR:/usr/local/lib/python3.11/site-packages:${PYTHONPATH:-}"

if [ -d "$REQ_DIR" ]; then
  for whl in "$REQ_DIR"/*.whl; do
    [ -e "$whl" ] || continue
    echo "[entrypoint] Found wheel: $whl"
    # Determine a target marker file for installed wheel (simple heuristic)
    pkgname=$(basename "$whl" .whl)
    marker="$VENDOR_DIR/.installed_${pkgname}"
    if [ -f "$marker" ]; then
      echo "[entrypoint] Wheel $pkgname already installed, skipping."
      continue
    fi

    echo "[entrypoint] Installing $pkgname into $VENDOR_DIR"
    # Install into vendor dir without dependencies (assume deps are satisfied by main image)
    pip install --no-deps --upgrade --target "$VENDOR_DIR" "$whl" || {
      echo "[entrypoint] pip install failed for $whl" >&2
      # Don't exit - continue to start the app, but note failure
      continue
    }
    # Create marker to avoid reinstalling
    touch "$marker" || true
  done
else
  echo "[entrypoint] No requirements directory mounted at $REQ_DIR"
fi

echo "[entrypoint] Starting application: $@"
exec "$@"
