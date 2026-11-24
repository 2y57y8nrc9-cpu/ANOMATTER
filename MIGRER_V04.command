#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1
printf '\nANOMATTER / IMPORT DES ARCHIVES v0.4\n'
echo 'Sélectionne dans le Finder le dossier ANOMATTER v0.4 contenant archive_learning.'
SOURCE="$(osascript -e 'POSIX path of (choose folder with prompt "Sélectionner ton dossier ANOMATTER v0.4")' 2>/dev/null)"
if [ -z "$SOURCE" ];then echo 'Migration annulée.';exit 0;fi
PY=""
for p in /usr/bin/python3 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 python3;do
 if command -v "$p" >/dev/null 2>&1;then PY="$(command -v "$p")";break;fi
done
if [ -z "$PY" ];then echo 'Python introuvable';exit 1;fi
"$PY" "$ROOT/migrer_archives.py" "$SOURCE" --destination "$ROOT"
read -r -p 'Appuie sur Entrée pour fermer… ' _
