# ANOMATTER Studio v0.5 — Mac

**Version de recherche opérationnelle, non signée par Apple.**

## Démarrer

1. Décompresse le ZIP en entier dans Téléchargements (ou un dossier de ton choix).
2. Ouvre le dossier `ANOMATTER_v0.5_STUDIO_MAC`.
3. Double-clique sur **`OUVRIR_ANOMATTER.command`**. Garde le Terminal ouvert.
4. ANOMATTER s'ouvre dans Safari/Chrome sur `http://127.0.0.1:8770` (ou le port affiché dans le Terminal).
5. Importe une image **MORPHOLOGY**. Tu peux ajouter **MATERIAL**, **PATTERN**, **CHROMATIC**, **PARASITE**.
6. Clique **PRÉVISUALISER LA FUSION** — cet aperçu ne télécharge aucun modèle IA.
7. Régle le bending, sélectionne un mode, puis **GÉNÉRER AVEC NETWORK BENDING**.
8. Note les spécimens, sélectionne une image et clique **Réinjecter → EVOLVE** pour produire quatre descendants.

### Pour réintégrer les notes et générations de la v0.4

Lance **`MIGRER_V04.command`** et choisis ton ancien dossier ANOMATTER v0.4 dans le Finder. Le script copie les images/archives, fusionne les notes et préserve les données existantes. Il ne copie aucun environnement Python ni programme provenant de la v0.4. Effectue cette opération avant d'évaluer de nouveaux spécimens pour simplifier l'organisation.

Les **13 images de référence** précédemment partagées sont déjà incluses. Elles ne portent aucune note inventée.

### Compatibilité / installation

- Mac Apple Silicon M1 ou ultérieur recommandé. Le GPU Metal/MPS est nécessaire au mode génératif rapide.
- Python 3.11 ou 3.12. La v0.5 recherche les environnements Python installés avec les anciennes versions avant d'installer ses propres dépendances.
- Les modèles IA préentraînés **ne sont pas inclus dans le ZIP**. Le premier lancement d'une génération Stable Diffusion 1.5 télécharge plusieurs gigaoctets et nécessite une connexion Internet. Des modèles supplémentaires sont téléchargés uniquement si **IP-Adapter** ou **ControlNet** sont activés.
- Environ **16 Go de mémoire unifiée** conseillés pour SD 1.5 ; les modes optionnels peuvent demander davantage. Aucun niveau de mémoire n'est garanti sur tous les Mac.
- L'interface est locale et se lance uniquement via `127.0.0.1` : **ce n'est pas encore une application iPhone ou une application Mac signée**.
- L'avertissement Gatekeeper de macOS est normal pour ce prototype non signé ; vérifie sa provenance et inspecte les scripts avant de décider de l'autoriser. Ne désactive pas la sécurité globale du Mac.

## Ce qui fonctionne sans IA

- Import d'images dans **cinq rôles** ; prévisualisation du support image-to-image ; détails et cartes de couleur.
- Traitements différenciés par rôle : MORPHOLOGY = image structurelle de départ ; MATERIAL = relief de haute fréquence ; PATTERN = répétitions de contours ; CHROMATIC = transfert statistique de couleur ; PARASITE = masques régionaux. C'est un **prétraitement de pixels** : ce système ne garantit pas une séparation sémantique des objets.
- Archivage, notes 0–15, choix de favoris, comparaison de deux images, propositions de paramètres heuristiques (Préserver / Explorer / Dériver / Contradiction).
- Généalogie des images dérivées, reprise des sessions et export des PNG.

## Génération neuronale

- Stable Diffusion 1.5 image-to-image avec témoin neutre et même graine initiale.
- Perturbation réelle des activations du UNet pendant l'inférence (6 modes), six perturbations par canaux, séquenceur temporel (début/fin) et pondération Early/Middle/Late, masque spatial, interruption de la génération, 4 descendants de modes différents en EVOLVE.
- **IP-Adapter (option)** : conditionnement neuronal de l'image associée au rôle sélectionné. Téléchargement de poids `h94/IP-Adapter` ; mémoire GPU supplémentaire.
- **ControlNet (option)** : guidage des contours de Morphology avec un ControlNet pour SD 1.5. L'aperçu des contours est construit par seuillage des arêtes (approximation de Canny), ce qui peut réduire la fidélité. Téléchargement supplémentaire du modèle `lllyasviel/control_v11p_sd15_canny` ; mémoire GPU supplémentaire.
- IP-Adapter et ControlNet ne peuvent **pas** être activés ensemble dans cette version, pour limiter les contraintes mémoire et les risques de compatibilité. Le mode par défaut n'utilise aucun modèle optionnel.

**Transparence :** les transformations ont été validées par tests unitaires et une simulation d'intégration avec un réseau témoin ; la génération complète SD 1.5 et les modèles optionnels n'ont pas été exécutés ici avec leurs poids sur un Mac physique. Aucun résultat généré par IA n'est promis ou préenregistré. Les courbes, niveaux de nouveauté dHash et suggestions de paramètres sont expérimentaux et ne reflètent pas un apprentissage profond de tes goûts.

## Fichiers

- `OUVRIR_ANOMATTER.command` : démarre le serveur et ton navigateur.
- `server.py` : serveur HTTP local, traitements, jobs, traces, gestion des modèles.
- `source_roles.py` : préparation des cinq rôles.
- `bending.py` : hooks et séquenceur de perturbations des activations.
- `learning.py` : archive, évaluations, recommandations et comparaisons.
- `studio.html` : interface responsive.
- `MIGRER_V04.command` / `migrer_archives.py` : migration de tes anciennes archives.
- `DIAGNOSTIC.command` : renseignements utiles si le moteur ne démarre pas.
- `tests/` : tests sans téléchargement de modèles ; commande : `python3.11 -m pytest -q` (si pytest est installé).

### Données et confidentialité

Le moteur ne met en ligne aucune interface, car il écoute uniquement `127.0.0.1`. Les sources sont stockées sur ce Mac dans `resultats/<identifiant>` après lancement des générations. Le code peut télécharger des poids depuis Hugging Face. Il n'envoie pas volontairement tes images à un service externe. Évite de partager ton dossier complet si des photographies privées y figurent.

## Pistes ultérieures

- Séparation sémantique réelle des matériaux et motifs, conditionnement multi-adapter par régions.
- Plus grande stabilité des modèles optionnels sur Metal, presets mémoire selon le Mac.
- App native signée pour macOS, suivi mobile sécurisé, moteur distant et synchronisation choisie par l'utilisateur.

## Documentation des technologies

- Documentation Diffusers image-to-image : https://huggingface.co/docs/diffusers/v0.39.0/en/using-diffusers/img2img
- Documentation IP-Adapter : https://huggingface.co/docs/diffusers/v0.39.0/en/using-diffusers/ip_adapter
- ControlNet pour Stable Diffusion : https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/controlnet
- Terence Broad, network bending (référence conceptuelle ; code original non redistribué) : https://github.com/terrybroad/network-bending
