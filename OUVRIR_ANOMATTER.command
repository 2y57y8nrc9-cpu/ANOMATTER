#!/bin/bash
# ANOMATTER v0.5 — local-only launcher; no global pip changes.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1
printf '\n=============================================\n ANOMATTER STUDIO v0.5 — DÉMARRAGE\n=============================================\n\n'
if [ "$(uname -s)" != 'Darwin' ]; then
  echo 'Ce lanceur est prévu pour macOS.'
  read -r -p 'Appuie sur Entrée pour fermer… ' _
  exit 1
fi
if [ "$(uname -m)" != 'arm64' ]; then
  echo 'ATTENTION : la génération GPU locale demande Apple Silicon (M1 ou ultérieur).'
fi
usable(){
  [ -x "$1" ] || return 1
  "$1" -c 'import sys; assert (3,11)<=sys.version_info[:2]<=(3,12); import PIL, numpy, torch, diffusers, transformers, accelerate' >/dev/null 2>&1
}
ENGINE=""
for candidate in \
 "$ROOT/.venv/bin/python" \
 "$HOME/Downloads/ANOMATTER_v0.4_PRET_A_L_EMPLOI/.venv/bin/python" \
 "$HOME/Downloads/ANOMATTER_v0.4_LEARNING_MAC/.venv/bin/python" \
 "$HOME/Downloads/ANOMATTER_Network_Bending_Mac_v0.3.2/.venv/bin/python" \
 "$HOME/Downloads/ANOMATTER_Network_Bending_Mac_v0.3.1/.venv/bin/python" \
 "$ROOT/../ANOMATTER_v0.4_PRET_A_L_EMPLOI/.venv/bin/python" \
 "$ROOT/../ANOMATTER_Network_Bending_Mac_v0.3.1/.venv/bin/python"; do
  if usable "$candidate";then ENGINE="$candidate";break;fi
done
if [ -n "$ENGINE" ];then
  echo "✓ Dépendances existantes retrouvées : $ENGINE"
else
  BASE=""
  for c in /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
           /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
           /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.12 \
           /usr/local/bin/python3.11 /usr/local/bin/python3.12 python3.11 python3.12;do
    if command -v "$c" >/dev/null 2>&1;then
      p="$(command -v "$c")"
      if "$p" -c 'import sys;assert (3,11)<=sys.version_info[:2]<=(3,12)' >/dev/null 2>&1;then BASE="$p";break;fi
    fi
  done
  if [ -z "$BASE" ];then
    echo 'Python 3.11 ou 3.12 absent. Installe-le depuis https://www.python.org/downloads/macos/'
    read -r -p 'Appuie sur Entrée pour fermer… ' _
    exit 1
  fi
  echo "Python compatible : $BASE"
  [ -x "$ROOT/.venv/bin/python" ] || "$BASE" -m venv "$ROOT/.venv" || exit 1
  ENGINE="$ROOT/.venv/bin/python"
  echo 'Installation isolée des dépendances : une connexion Internet est nécessaire…'
  "$ENGINE" -m pip install -r "$ROOT/requirements.txt" || {
    echo 'Échec des dépendances. Copie les dernières lignes du Terminal.'
    read -r -p 'Appuie sur Entrée pour fermer… ' _
    exit 1
  }
  if ! usable "$ENGINE";then
    echo 'Environnement Python incomplet.'
    read -r -p 'Appuie sur Entrée pour fermer… ' _
    exit 1
  fi
fi
PORT=""
for port in 8770 8771 8772 8773 8774;do
  if ! lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1;then PORT="$port";break;fi
done
if [ -z "$PORT" ];then
  echo 'Les ports 8770–8774 sont occupés. Ferme d’anciennes instances ANOMATTER.'
  read -r -p 'Appuie sur Entrée pour fermer… ' _
  exit 1
fi
export ANOMATTER_PORT="$PORT"
export PYTORCH_ENABLE_MPS_FALLBACK=1
printf '\n✓ Interface : http://127.0.0.1:%s\n' "$PORT"
echo '  Conserve cette fenêtre Terminal ouverte pendant la session.'
echo '  Stable Diffusion, IP-Adapter et ControlNet sont téléchargés seulement si nécessaires.'
echo '  Les fichiers restent dans ce dossier (sauf poids modèles téléchargés).'
printf '\n'
"$ENGINE" -u "$ROOT/server.py"
code="$?"
echo "Le moteur s’est arrêté (code $code)."
read -r -p 'Appuie sur Entrée pour fermer… ' _
exit "$code"
