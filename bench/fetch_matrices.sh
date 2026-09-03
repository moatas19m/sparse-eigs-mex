#!/usr/bin/env bash
# Download the benchmark set from the SuiteSparse Matrix Collection.
# Real problems, not sprandsym: random sparse matrices have no exploitable
# structure and would misrepresent both codes.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/../data" && pwd)"
BASE="https://suitesparse-collection-website.herokuapp.com/MM"

# group/name pairs: SPD, spanning ~3 orders of magnitude in n
MATS=(
  HB/nos3                     # n=960     small, SPD
  HB/bcsstk16                 # n=4884    structural
  UTEP/Dubcova2               # n=65025   2D FEM
  Wissgott/parabolic_fem      # n=525825  CFD, SPD
  McRae/ecology2              # n=999999  2D landscape, SPD
  Schmid/thermal2             # n=1228045 thermal FEM
  AMD/G3_circuit              # n=1585478 circuit sim, SPD
  Schenk_AFE/af_shell3        # n=504855  sheet metal
)
mkdir -p "$DEST"
for m in "${MATS[@]}"; do
  name="${m##*/}"
  if [ -f "$DEST/$name.mtx" ] || [ -f "$DEST/$name/$name.mtx" ]; then
    echo "have  $name"; continue
  fi
  echo "fetch $m"
  curl -fL --retry 3 -o "$DEST/$name.tar.gz" "$BASE/$m.tar.gz" \
    && tar -xzf "$DEST/$name.tar.gz" -C "$DEST" && rm -f "$DEST/$name.tar.gz"
done
echo "Matrices in $DEST"
