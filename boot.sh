#!/bin/bash

echo "[runtime] Linking missing shared libs..."

GL=$(find /nix/store -name 'libGL.so.1' | head -n 1)
GLIB=$(find /nix/store -name 'libglib-2.0.so.0' | head -n 1)
GTHREAD=$(find /nix/store -name 'libgthread-2.0.so.0' | head -n 1)

if [ -n "$GL" ]; then
  ln -sf "$GL" /usr/lib/libGL.so.1
  echo "Linked libGL.so.1"
else
  echo "Warning: libGL.so.1 not found"
fi

if [ -n "$GLIB" ]; then
  ln -sf "$GLIB" /usr/lib/libglib-2.0.so.0
  echo "Linked libglib-2.0.so.0"
else
  echo "Warning: libglib-2.0.so.0 not found"
fi

if [ -n "$GTHREAD" ]; then
  ln -sf "$GTHREAD" /usr/lib/libgthread-2.0.so.0
  echo "Linked libgthread-2.0.so.0"
else
  echo "Warning: libgthread-2.0.so.0 not found"
fi

echo "[runtime] Running app..."
exec "$@"

