#!/bin/bash

echo "[runtime] Checking OpenCV dependencies..."

# Check if we're in a Nix environment (Nixpacks) or standard Debian (Railpack)
if [ -d "/nix/store" ]; then
  # Nixpacks environment - find libraries in Nix store
  echo "Detected Nixpacks environment"
  GL=$(find /nix/store -name 'libGL.so.1' 2>/dev/null | head -n 1)
  GLIB=$(find /nix/store -name 'libglib-2.0.so.0' 2>/dev/null | head -n 1)
  GTHREAD=$(find /nix/store -name 'libgthread-2.0.so.0' 2>/dev/null | head -n 1)
  
  if [ -n "$GL" ]; then
    mkdir -p /usr/lib
    ln -sf "$GL" /usr/lib/libGL.so.1 2>/dev/null || true
    echo "Linked libGL.so.1 from Nix store"
  fi
  
  if [ -n "$GLIB" ]; then
    mkdir -p /usr/lib
    ln -sf "$GLIB" /usr/lib/libglib-2.0.so.0 2>/dev/null || true
    echo "Linked libglib-2.0.so.0 from Nix store"
  fi
  
  if [ -n "$GTHREAD" ]; then
    mkdir -p /usr/lib
    ln -sf "$GTHREAD" /usr/lib/libgthread-2.0.so.0 2>/dev/null || true
    echo "Linked libgthread-2.0.so.0 from Nix store"
  fi
else
  # Railpack/Debian environment - libraries should be in standard paths
  echo "Detected Railpack/Debian environment"
  
  # Check if libraries exist in standard locations
  if [ ! -f "/usr/lib/x86_64-linux-gnu/libGL.so.1" ] && [ ! -f "/usr/lib/libGL.so.1" ]; then
    echo "Warning: libGL.so.1 not found in standard locations"
    echo "Note: For Railpack, system libraries should be installed via apt-get"
    echo "If you see libGL errors, you may need to switch to Nixpacks or use Dockerfile"
  else
    echo "OpenGL libraries found in standard locations"
  fi
fi

echo "[runtime] Running app..."
exec "$@"

