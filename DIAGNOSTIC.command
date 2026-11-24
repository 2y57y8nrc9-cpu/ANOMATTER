#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1
printf '\nANOMATTER 0.5 — DIAGNOSTIC MAC\n'
uname -srm
for py in "$ROOT/.venv/bin/python" /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 python3.11;do
 if command -v "$py" >/dev/null 2>&1;then
   echo "Python : $py";"$py" --version
   "$py" -c 'import torch; print("Torch",torch.__version__, "MPS",torch.backends.mps.is_available())' 2>&1 || true
   "$py" -c 'import diffusers, transformers;print("Diffusers",diffusers.__version__, "Transformers",transformers.__version__)' 2>&1 || true
 fi
done
for port in 8770 8771 8772 8773 8774;do lsof -nP -iTCP:"$port" -sTCP:LISTEN | tail -2;done
echo 'En cas de problème, transmets les dernières lignes du Terminal.'
read -r -p 'Entrée pour fermer… ' _
