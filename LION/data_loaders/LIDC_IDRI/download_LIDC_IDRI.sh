#!/usr/bin/env bash
# Download the LIDC-IDRI dataset using the NBIA Data Retriever command-line interface.
# Primary download logic and was hand-written with Singularity support, before robustness improvements were added using Codex & GPT 5.4.
set -Eeuo pipefail

NBIA_DEB_URL="${NBIA_DEB_URL:-https://github.com/CBIIT/NBIA-TCIA/releases/download/DR-4_4_3-TCIA-20240916-1/nbia-data-retriever_4.4.3-1_amd64.deb}"
DIAGNOSIS_XLS_URL="${DIAGNOSIS_XLS_URL:-https://www.cancerimagingarchive.net/wp-content/uploads/tcia-diagnosis-data-2012-04-20.xls}"
JAVA_CONTAINER_IMAGE="${JAVA_CONTAINER_IMAGE:-eclipse-temurin:17-jre}"
# NBIA prompts with A/M/E when resuming: A = all, M = missing series, E = exit.
# Default to M so interrupted downloads resume without overwriting completed series.
NBIA_RESUME_CHOICE="${NBIA_RESUME_CHOICE:-M}"

die() {
  echo "Error: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

normalize_path() {
  if have python3; then
    python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$1"
  else
    # Good enough for systems without Python, provided the parent exists.
    local path="$1"
    local dir
    dir="$(dirname "$path")"
    local base
    base="$(basename "$path")"
    (cd "$dir" && printf "%s/%s\n" "$(pwd -P)" "$base")
  fi
}

default_raw_path_from_lion_paths() {
  local paths_py="$1"

  have python3 || return 1
  [[ -f "$paths_py" ]] || return 1

  python3 - "$paths_py" <<'PY'
import importlib.util
import pathlib
import sys

paths_py = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("lion_paths", paths_py)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.LION_DATA_PATH.joinpath("raw"))
PY
}

download_file() {
  local url="$1"
  local dest="$2"
  local tmp="${dest}.tmp"

  rm -f "$tmp"
  if have curl; then
    curl --fail --location --retry 3 --output "$tmp" "$url"
  elif have wget; then
    wget -O "$tmp" "$url"
  else
    die "Neither curl nor wget is available to download $url"
  fi
  mv "$tmp" "$dest"
}

extract_deb() {
  local deb_path="$1"
  local install_root="$2"
  local extract_dir="$3"

  rm -rf -- "$install_root" "$extract_dir"
  mkdir -p "$install_root" "$extract_dir"

  if have dpkg-deb; then
    dpkg-deb -x "$deb_path" "$install_root"
    return
  fi

  have ar || die "Need either dpkg-deb or ar to extract $deb_path"
  have tar || die "Need tar to extract the data archive from $deb_path"

  (cd "$extract_dir" && ar x "$deb_path")
  local data_archive
  data_archive="$(find "$extract_dir" -maxdepth 1 -type f -name 'data.tar.*' | head -n 1)"
  [[ -n "$data_archive" ]] || die "Could not find data.tar.* inside $deb_path"
  tar -xf "$data_archive" -C "$install_root"
}

java_major_version() {
  java -version 2>&1 | awk -F '"' '
    /version/ {
      split($2, parts, ".")
      if (parts[1] == "1") {
        print parts[2]
      } else {
        print parts[1]
      }
      exit
    }
  '
}

run_downloader_command() {
  local choice="${NBIA_RESUME_CHOICE^^}"

  case "$choice" in
    A|M|E)
      printf "%s\n" "$choice" | "$@"
      ;;
    PROMPT|INTERACTIVE)
      "$@"
      ;;
    *)
      die "NBIA_RESUME_CHOICE must be M, A, E, prompt, or interactive"
      ;;
  esac
}

run_downloader_with_host_java() {
  run_downloader_command java -jar "$APP_JAR" "${DOWNLOADER_ARGS[@]}"
}

run_downloader_with_singularity_like() {
  local runner="$1"
  run_downloader_command "$runner" exec \
    --bind "$RAW_PATH:$RAW_PATH" \
    "docker://${JAVA_CONTAINER_IMAGE}" \
    java -jar "$APP_JAR" "${DOWNLOADER_ARGS[@]}"
}

run_downloader_with_docker() {
  run_downloader_command docker run -i --rm \
    -v "$RAW_PATH:$RAW_PATH" \
    "$JAVA_CONTAINER_IMAGE" \
    java -jar "$APP_JAR" "${DOWNLOADER_ARGS[@]}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PATHS_PY="$SCRIPT_DIR/../../utils/paths.py"

if [[ -z "${RAW_PATH:-}" ]]; then
  RAW_PATH="$(default_raw_path_from_lion_paths "$PATHS_PY")" || die "RAW_PATH environment variable is not set, and could not read LION_DATA_PATH from $PATHS_PY"
fi

RAW_PATH="$(normalize_path "$RAW_PATH")"
[[ "$RAW_PATH" != "/" ]] || die "Refusing to use / as RAW_PATH"
mkdir -p "$RAW_PATH"
LIDC_IDRI_RAW_PATH="$RAW_PATH/LIDC-IDRI"
mkdir -p "$LIDC_IDRI_RAW_PATH"

DEFAULT_MANIFEST="$SCRIPT_DIR/LIDC-IDRI.tcia"
if [[ -n "${TCIA_MANIFEST_PATH:-}" ]]; then
  SOURCE_MANIFEST="$(normalize_path "$TCIA_MANIFEST_PATH")"
elif [[ -f "$DEFAULT_MANIFEST" ]]; then
  SOURCE_MANIFEST="$DEFAULT_MANIFEST"
elif [[ -f "$RAW_PATH/LIDC-IDRI.tcia" ]]; then
  SOURCE_MANIFEST="$RAW_PATH/LIDC-IDRI.tcia"
else
  die "Could not find LIDC-IDRI.tcia. Set TCIA_MANIFEST_PATH to its location."
fi
[[ -f "$SOURCE_MANIFEST" ]] || die "Manifest not found: $SOURCE_MANIFEST"

TOOLS_DIR="$RAW_PATH/tools"
NBIA_DIR="$TOOLS_DIR/nbia-data-retriever"
INSTALL_ROOT="$NBIA_DIR/root"
EXTRACT_DIR="$NBIA_DIR/deb-extract"
DEB_PATH="$NBIA_DIR/nbia-data-retriever.deb"
APP_JAR="$INSTALL_ROOT/opt/nbia-data-retriever/lib/app/StandaloneDM.jar"

mkdir -p "$NBIA_DIR"

MANIFEST_FOR_RUN="$TOOLS_DIR/$(basename "$SOURCE_MANIFEST")"
mkdir -p "$TOOLS_DIR"
if [[ "$SOURCE_MANIFEST" != "$MANIFEST_FOR_RUN" ]]; then
  cp "$SOURCE_MANIFEST" "$MANIFEST_FOR_RUN"
fi

if [[ ! -f "$APP_JAR" ]]; then
  echo "Installing NBIA Data Retriever into $NBIA_DIR"
  download_file "$NBIA_DEB_URL" "$DEB_PATH"
  extract_deb "$DEB_PATH" "$INSTALL_ROOT" "$EXTRACT_DIR"
fi
[[ -f "$APP_JAR" ]] || die "Could not find StandaloneDM.jar after extracting $DEB_PATH"

DIAGNOSIS_XLS_PATH="$LIDC_IDRI_RAW_PATH/tcia-diagnosis-data-2012-04-20.xls"
if [[ ! -f "$DIAGNOSIS_XLS_PATH" ]]; then
  echo "Downloading LIDC diagnosis file to $DIAGNOSIS_XLS_PATH"
  download_file "$DIAGNOSIS_XLS_URL" "$DIAGNOSIS_XLS_PATH"
else
  echo "LIDC diagnosis file already exists at $DIAGNOSIS_XLS_PATH"
fi

DOWNLOADER_ARGS=(--agree-to-license --cli "$MANIFEST_FOR_RUN" -d "$RAW_PATH" -v -m)

echo "Raw data path: $RAW_PATH"
echo "LIDC-IDRI dataset path: $LIDC_IDRI_RAW_PATH"
echo "TCIA manifest: $MANIFEST_FOR_RUN"
echo "NBIA Data Retriever jar: $APP_JAR"
echo "Resume choice: $NBIA_RESUME_CHOICE (M = missing series, A = all/overwrite, prompt = ask)"
echo "MD5 verification: enabled"

if have java && [[ "${NBIA_FORCE_CONTAINER:-0}" != "1" ]]; then
  JAVA_MAJOR="$(java_major_version || true)"
  if [[ "$JAVA_MAJOR" =~ ^[0-9]+$ && "$JAVA_MAJOR" -ge 17 ]]; then
    echo "Using host Java $(java -version 2>&1 | head -n 1)"
    run_downloader_with_host_java
    exit 0
  fi
  echo "Host Java is unavailable or older than 17; trying a container runtime."
fi

if have apptainer; then
  echo "Using Apptainer with $JAVA_CONTAINER_IMAGE"
  run_downloader_with_singularity_like apptainer
elif have singularity; then
  echo "Using Singularity with $JAVA_CONTAINER_IMAGE"
  run_downloader_with_singularity_like singularity
elif have docker; then
  echo "Using Docker with $JAVA_CONTAINER_IMAGE"
  run_downloader_with_docker
else
  die "Need Java 17+, Apptainer, Singularity, or Docker to run NBIA Data Retriever."
fi
