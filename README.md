# Collection Équinoxe — hausse de loyer 2026
CodeML 2026 · Défi « Clés en main » (JADCO)

## Résultat

**+3,2 %** de hausse du loyer **encaissé** (`sRentEffective`, après concessions) · **+5,8 %** au bail (`sRent`).
En 2025, sur la même mesure : 4,0 % encaissé, 7,4 % au bail.

Scénarios selon les concessions : 2,4 % (elles montent comme en 2025) / **3,2 %** (rythme moyen 2024–2025) / 4,1 % (montée divisée par 2) / 5,8 % (leur part cesse de monter). Erreur moyenne historique (MAE 2023–2025) : 1,6 pt, pire année 2,4 pt (ce n'est pas un intervalle de confiance).

**Définition.** Hausse du loyer encaissé d'un **même logement** (`sPropCode + sUnitCode`) par rapport à son bail précédent, pour les baux **débutant en 2026** : médiane dans chacun des quatre segments (Québec / Ontario × renouvellement / nouveau locataire), puis moyenne pondérée par les baux attendus en 2026. The Met (Ottawa) suit les règles ontariennes (exempté de la ligne directrice : premiers baux en 2023).

| Segment 2026 | Au bail | Encaissé |
|---|---|---|
| Québec, renouvellement | 4,7 % | 2,2 % |
| Québec, nouveau locataire | 7,6 % | 4,9 % |
| The Met, renouvellement | 4,8 % | 2,1 % |
| The Met, nouveau locataire | 7,3 % | 4,7 % |

## Livrables

| Livrable | Où | Contenu |
|---|---|---|
| Estimation 2026 et définition | ce README (ci-dessus) et `solution.ipynb`, section 7.6 | +3,2 % encaissé, +5,8 % au bail, scénarios 2,4 % / 3,2 % / 4,1 % / 5,8 % |
| `estimate_2026()` et `backtest()` | `solution.ipynb`, section 7.5 | fonctions complétées ; backtest 2023, 2024, 2025 (erreur moyenne 1,6 pt) |
| Prédictions 2026 | `predictions_2026.csv`, créé par le notebook (section 8) | une ligne par immeuble × nombre de chambres × statut (renouvellement / nouveau locataire), hausse au bail et encaissée, plus lignes agrégées |
| Tableau de bord | `dashboard_2026.html`, créé par le notebook (annexe C) | page autonome par immeuble : hausse 2026, historique, baux attendus, écart au marché (Kaggle), **choix du scénario** (bas, central, haut, concessions stables) |
| Modèle | `models/two_part_2026.json`, créé par le notebook (section 7.10) | paramètres du modèle retenu par segment ; le notebook vérifie qu'il redonne 3,2 % |
| Figures | `figures/*.png`, créées par le notebook | graphiques de la présentation |
| Présentation | remise séparément (Devpost) | 28 diapositives, en anglais |

Les prédictions, le tableau de bord, le modèle et les figures sont calculés à partir des données du CRM : conformément aux consignes, ils ne sont pas publiés ici et sont recréés en exécutant le notebook (voir Exécution).

## Fichiers

| Fichier | Contenu |
|---|---|
| `solution.ipynb` | Livrable principal, organisé selon les 7 tâches des consignes : question, données et effet de composition, unité constante, concessions, renouvellements / Ontario, données publiques, prévision et backtest (`estimate_2026()`, `backtest()`, scénarios, hypothèses) ; bonus par immeuble ; annexes. **Publié sans aucune sortie.** |
| `public_data.py` | Téléchargement et nettoyage des données publiques, une fonction par source, appelé par le notebook (section 2.1). |
| `external/clean/*.csv` | Données publiques déjà nettoyées (StatCan, TAL, Ontario, SCHL, FPI, résumé Kaggle par ville), lues par défaut par le notebook. |
| `requirements.txt` | Versions des librairies. |

**Confidentialité.** Ce dépôt ne contient **aucune donnée du CRM** (`equinoxe_*.csv`, exclus par `.gitignore`) et **aucune sortie** de notebook. Les fichiers produits à partir de ces données (`predictions_2026.csv`, `dashboard_2026.html`, `models/two_part_2026.json`, `figures/*.png`) ne sont pas publiés : ils sont recréés en exécutant le notebook avec les fichiers CRM.

## Exécution

Python 3.10.

```bash
pip install -r requirements.txt
# Placer les 4 fichiers CRM (equinoxe_*.csv) à la racine du dépôt (jamais commités), puis :
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 solution.ipynb
```

Un seul notebook à exécuter (~40 s). Il crée `predictions_2026.csv` (prévision par immeuble × chambres × statut), `dashboard_2026.html` (tableau de bord par immeuble avec choix du scénario), `models/two_part_2026.json` (paramètres du modèle) et `figures/` (graphiques de la présentation). Pour retélécharger les données publiques depuis les sources, mettre `REBUILD_PUBLIC_DATA = True` dans la section 2.1 (internet requis, ~1 min) ; Kaggle : placer le zip du jeu de données dans `external/raw/kaggle/` (compte Kaggle requis), les annonces ne sont pas redistribuées.

**Ne pas commiter le notebook après l'avoir exécuté** : ses sorties contiendraient des données du CRM. Vider les sorties d'abord : `jupyter nbconvert --clear-output --inplace solution.ipynb`.

**Versions** : pandas 2.3.3 · numpy 2.2.6 · matplotlib 3.10.9 · scikit-learn 1.7.2 · statsmodels 0.15.0 · xgboost 3.2.0 · requests 2.34.2 · openpyxl 3.1.5 · pypdf 6.18.0.

## Méthode

1. **Même logement** : chaque bail comparé au précédent du même logement, jamais `sSite + sUnitCode`. La médiane naïve mesure surtout le mélange d'immeubles.
2. **Concessions** : `sRentEffective` = `sRent` + concessions ÷ durée (reconstruit pour 91 % des baux). L'écart bail − encaissé = la hausse de la part des concessions.
3. **Données publiques** : SCHL, IPC, TAL et ligne directrice ontarienne, FPI concurrentes servent de repères et sont réconciliés avec la tendance interne ; Kaggle (photo de juin 2024) sert à entraîner un modèle de prix sur les autres propriétaires et à mesurer l'écart de chaque immeuble au marché (pas une hausse, donc pas une entrée de la prévision).
4. **Constat** : la hausse est une décision annuelle commune à tout le portefeuille ; immeuble, chambres, superficie fixent le *prix*, pas la *hausse*.
5. **Peu de données → modèle simple** : la hausse est une décision annuelle (~20 observations). Le meilleur des 133 modèles choisi après coup semble excellent (0,97 pt) mais, choisi sans voir l'avenir, il perd son avance (1,58 pt) : c'est du surapprentissage. Le modèle retenu n'a qu'un paramètre, fixé d'avance. Testé aussi : prévoir le prix de chaque logement puis en déduire la hausse (modèle hédonique, XGBoost, hybride prix du marché) : pas mieux.
6. **Validation** : prévision au 31 décembre de l'année précédente, données publiques filtrées sur leur date de publication, sauf le taux du TAL de l'année visée (publié en janvier, connu avant les avis de renouvellement ; sans lui, 2026 donne 4,4 %), années testées 2023–2025. Référence à battre : « même hausse que l'an dernier ».
7. **Modèle retenu** : hausse au bail = ½ × hausse de l'an dernier + ½ × (repère de l'année + écart habituel), repère = taux du TAL (renouvellements Québec) ou SCHL (nouveaux locataires, The Met) ; encaissé = bail − hausse moyenne des concessions des 2 dernières années. À égalité avec la référence sur Équinoxe (1,61 contre 1,49 pt), meilleur sur 35 villes canadiennes (2,38 contre 2,48 pt), et seul à intégrer ce qu'on sait de 2026 (TAL 3,1 % contre 5,9 %).

**Hypothèses** : année = début du bail ; immeubles québécois non chauffés ; The Met exempté (à confirmer) ; part de renouvellements 2026 = 2025 ; écart habituel au repère moyenné sur 3 ans, fixé d'avance (sensibilité : 1 an à toutes les années donne 3,2 % à 4,1 %, section 7.8) ; données publiées après le 31 déc. 2025 utilisées seulement pour vérifier, sauf le taux du TAL 2026 (3,1 %, publié en janvier 2026).

## Sources

- Statistique Canada, tableau 18-10-0004-01 (IPC, loyers) — https://www150.statcan.gc.ca/t1/tbl1/fr/tv.action?pid=1810000401
- Tribunal administratif du logement — https://www.tal.gouv.qc.ca ; taux 2026 : https://ici.radio-canada.ca/nouvelle/2221781/loyers-tal-logement-augmentation-locataires
- Ontario, ligne directrice et exemption post-2018 — https://www.ontario.ca/page/rent-increase-guideline
- SCHL, Enquête sur les logements locatifs 2021–2025 — https://www.cmhc-schl.gc.ca/professionals/housing-markets-data-and-research/housing-data/data-tables/rental-market/rental-market-report-data-tables
- FPI : InterRent, Killam, CAPREIT, Minto (liens dans `external/clean/reit_metrics.csv`)
- Kaggle, *25000+ Canadian rental housing market (June 2024)* — https://www.kaggle.com/datasets/sergiygavrylov/25000-canadian-rental-housing-market-june-2024 (annonces non redistribuées ; seul un résumé par ville est fourni)
- Méthodes : Bailey, Muth et Nourse (1963) ; Case et Shiller (1987) ; Rosen (1974) ; Lundberg et Lee (2017).

**Outil d'IA** : Claude (Anthropic), modèle `claude-opus-5-5`, utilisé pour rechercher les sources publiques, écrire le code et rédiger la documentation. Les choix de modélisation ont été validés par l'équipe.
