ROOT="/mnt/store_ext4/GIT/LATEX"
# ROOT="/Users/t/Documents/GIT/Cambridge/PROJECT/LATEX"
SLIDES_ROOT="$ROOT/A3SmartML-month-18-meeting"
PCM_OUTPUT_ROOT="pcm_slide_results"

# SUB_DIR="cia_pnp_slide_data"
SUB_DIR="cia_pnp_slide_CIGS_0288"

cp -R \
  "$PCM_OUTPUT_ROOT/$SUB_DIR/slide_export/." \
  "$SLIDES_ROOT/"
