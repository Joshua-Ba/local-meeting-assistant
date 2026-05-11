#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [[ "$BUILD_DIR" = /* ]]; then
  BUILD_DIR_ABS="$BUILD_DIR"
else
  BUILD_DIR_ABS="$ROOT_DIR/$BUILD_DIR"
fi

if [[ -d "$BUILD_DIR_ABS" ]]; then
  stale_cache=""
  while IFS= read -r cache_file; do
    cached_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache_file" | head -n 1)"
    cached_build="$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$cache_file" | head -n 1)"
    if [[ -n "$cached_source" && "$cached_source" != "$ROOT_DIR" && "$cached_source" != "$ROOT_DIR"/* ]]; then
      stale_cache="$cache_file"
      break
    fi
    if [[ -n "$cached_build" && "$cached_build" != "$BUILD_DIR_ABS" && "$cached_build" != "$BUILD_DIR_ABS"/* ]]; then
      stale_cache="$cache_file"
      break
    fi
  done < <(find "$BUILD_DIR_ABS" -name CMakeCache.txt -print)

  if [[ -n "$stale_cache" ]]; then
    echo "Existing CMake cache belongs to another checkout: $stale_cache"
    echo "Removing $BUILD_DIR so CMake can configure this copy cleanly..."
    rm -rf "$BUILD_DIR_ABS"
  fi
fi

echo "Configuring tests..."
cmake -S "$ROOT_DIR" -B "$BUILD_DIR_ABS" -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" -DBUILD_TESTING=ON

echo "Building test target..."
cmake --build "$BUILD_DIR_ABS" --target local-meeting-assistant-tests

echo "Running tests..."
ctest --test-dir "$BUILD_DIR_ABS" --output-on-failure "$@"
