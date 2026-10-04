"""Données publiques : téléchargement et nettoyage, une fonction par source.

Appelé par solution.ipynb (section 1c) quand REBUILD_PUBLIC_DATA = True ; sinon le notebook lit
les fichiers déjà nettoyés de external/clean/. Les fichiers bruts sont gardés dans external/raw/.
Internet requis, sauf Kaggle (téléchargement manuel, compte requis).
"""
import contextlib
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

EXTERNAL_DIR = Path(__file__).resolve().parent / "external"


@contextlib.contextmanager
def _in_external_dir():
    # Les chemins des sources sont relatifs à external/ (raw/..., clean/...).
    previous = os.getcwd()
    os.chdir(EXTERNAL_DIR)
    try:
        yield
    finally:
        plt.close("all")
        os.chdir(previous)


def build_cpi():
    """IPC de Statistique Canada (tableau 18-10-0004-01) : loyers, logement locatif, ensemble.
    Sorties : clean/statcan_cpi_monthly.csv, clean/statcan_cpi_annual.csv (avec dates de publication)."""
    with _in_external_dir():
        import json
        from pathlib import Path

        import pandas as pd
        import requests
        import matplotlib.pyplot as plt

        RAW_DIR = Path("raw/statcan")
        CLEAN_DIR = Path("clean")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        CLEAN_DIR.mkdir(parents=True, exist_ok=True)

        WDS_URL = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"
        FIRST_YEAR = 2015                    # a few years before the first Équinoxe lease (2017)
        MONTHS_TO_FETCH = 12 * 13            # covers 2014 onward, enough for a 2015 yoy
        MONTHS_PER_YEAR = 12

        # Vector IDs looked up in the table metadata (18100004_MetaData.csv)
        SERIES = {
            "rent_canada":           ("Canada",   "Rent",                 41691052),
            "rent_quebec":           ("Quebec",   "Rent",                 41691818),
            "rent_ontario":          ("Ontario",  "Rent",                 41691954),
            "rented_accom_montreal": ("Montreal", "Rented accommodation", 41692878),
            "rented_accom_ottawa":   ("Ottawa",   "Rented accommodation", 41692884),
            "allitems_canada":       ("Canada",   "All-items",            41690973),
            "allitems_quebec":       ("Quebec",   "All-items",            41691783),
            "allitems_ontario":      ("Ontario",  "All-items",            41691919),
            "allitems_montreal":     ("Montreal", "All-items",            41692876),
            "allitems_ottawa":       ("Ottawa",   "All-items",            41692882),
        }

        payload = [{"vectorId": vid, "latestN": MONTHS_TO_FETCH} for _, _, vid in SERIES.values()]
        response = requests.post(WDS_URL, json=payload, timeout=60)
        response.raise_for_status()
        raw = response.json()

        raw_path = RAW_DIR / "cpi_vectors.json"
        raw_path.write_text(json.dumps(raw, indent=1))
        print(f"saved {raw_path}  ({raw_path.stat().st_size/1e3:.0f} kB)")
        assert all(item["status"] == "SUCCESS" for item in raw), "a vector failed to download"

        vector_to_series = {vid: name for name, (_, _, vid) in SERIES.items()}
        rows = []
        for item in raw:
            obj = item["object"]
            name = vector_to_series[obj["vectorId"]]
            geo, product, _ = SERIES[name]
            for point in obj["vectorDataPoint"]:
                rows.append({"series": name, "geo": geo, "product": product,
                             "month": pd.Timestamp(point["refPer"]),
                             "index_value": point["value"],
                             "released_at": pd.Timestamp(point["releaseTime"])})
        cpi_monthly = pd.DataFrame(rows).sort_values(["series", "month"]).reset_index(drop=True)
        cpi_monthly["year"] = cpi_monthly.month.dt.year
        cpi_monthly.to_csv(CLEAN_DIR / "statcan_cpi_monthly.csv", index=False)
        print(cpi_monthly.groupby("series").month.agg(["min", "max", "size"]))

        annual = (cpi_monthly.groupby(["series", "geo", "product", "year"])
                  .agg(index_avg=("index_value", "mean"),
                       months_available=("index_value", "size"),
                       last_release=("released_at", "max"))
                  .reset_index())
        december = (cpi_monthly[cpi_monthly.month.dt.month == MONTHS_PER_YEAR]
                    .set_index(["series", "year"]).index_value.rename("index_dec"))
        annual = annual.join(december, on=["series", "year"])

        is_full_year = annual.months_available == MONTHS_PER_YEAR
        annual["yoy_avg_pct"] = (annual.where(is_full_year).groupby("series").index_avg.pct_change(fill_method=None) * 100)
        annual["yoy_dec_pct"] = annual.groupby("series").index_dec.pct_change(fill_method=None) * 100
        annual = annual[annual.year >= FIRST_YEAR].round(3)

        annual.to_csv(CLEAN_DIR / "statcan_cpi_annual.csv", index=False)
        annual.pivot(index="year", columns="series", values="yoy_avg_pct").round(2)

        rent_cols = ["rent_quebec", "rent_ontario", "rented_accom_montreal", "rented_accom_ottawa"]
        wide = annual.pivot(index="year", columns="series", values="yoy_avg_pct")[rent_cols]
        ax = wide.plot(marker="o", figsize=(9, 4.5))
        ax.set_ylabel("year-over-year change, annual average (%)")
        ax.set_title("StatCan CPI — rent components")
        ax.grid(alpha=0.3)
        plt.tight_layout(); plt.close()

        print("Partial years:")
        print(annual[annual.months_available < MONTHS_PER_YEAR][["series", "year", "months_available"]].drop_duplicates("year"))


def build_regulation():
    """Taux du TAL (non chauffé, vérifiés dans les PDF) et ligne directrice ontarienne + exemption post-2018.
    Sorties : clean/rent_regulation_annual.csv, clean/ontario_exemption.csv."""
    with _in_external_dir():
        import re
        import html
        from io import BytesIO
        from pathlib import Path

        import pandas as pd
        import requests
        from pypdf import PdfReader

        RAW_TAL = Path("raw/tal");      RAW_TAL.mkdir(parents=True, exist_ok=True)
        RAW_ON  = Path("raw/ontario");  RAW_ON.mkdir(parents=True, exist_ok=True)
        CLEAN_DIR = Path("clean");      CLEAN_DIR.mkdir(parents=True, exist_ok=True)

        HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (research; CodeML 2026)"}
        TAL_PDF_URL = "https://www.tal.gouv.qc.ca/sites/default/files/COMMUNIQUE_FIXATION_{year}_FR.pdf"
        TAL_PDF_YEARS = [2016, 2022, 2023, 2024, 2025]           # years whose communiqué PDF is still online
        ONTARIO_URL = "https://www.ontario.ca/page/rent-increase-guideline"
        ONTARIO_EXEMPTION_DATE = pd.Timestamp("2018-11-15")
        FIRST_YEAR, LAST_YEAR = 2015, 2026

        def download(url, path):
            if not path.exists():
                r = requests.get(url, headers=HTTP_HEADERS, timeout=60)
                r.raise_for_status()
                path.write_bytes(r.content)
            return path

        def pdf_text(path):
            return " ".join(page.extract_text() for page in PdfReader(path).pages)

        tal_text = {}
        for year in TAL_PDF_YEARS:
            path = download(TAL_PDF_URL.format(year=year), RAW_TAL / f"COMMUNIQUE_FIXATION_{year}_FR.pdf")
            tal_text[year] = re.sub(r"\s+", " ", pdf_text(path))
            print(year, path, f"{len(tal_text[year]):,} chars")

        tal_reference = pd.DataFrame([
            # year, rate %, source, how it was checked
            (2016, 0.4,  TAL_PDF_URL.format(year=2016), "pdf"),
            (2017, None, "not retrieved", "missing"),
            (2018, 0.5,  "https://www.latribune.ca/2018/01/28/des-augmentations-modestes-des-loyers-en-vue-522ca029115c073a968f7b5c7c345b90/", "press, quoting the Régie"),
            (2019, None, "not retrieved", "missing"),
            (2020, 1.2,  "https://www.quebec.ca/nouvelles/actualites/details/le-calcul-de-laugmentation-des-loyers-en-2020", "official news page"),
            (2021, 0.8,  "https://www.newswire.ca/fr/news-releases/le-calcul-de-l-augmentation-des-loyers-en-2021-873582110.html", "official communiqué (Newswire)"),
            (2022, 1.28, TAL_PDF_URL.format(year=2022), "pdf"),
            (2023, 2.3,  TAL_PDF_URL.format(year=2023), "pdf"),
            (2024, 4.0,  TAL_PDF_URL.format(year=2024), "pdf"),
            (2025, 5.9,  "https://morneausenechal-avocat.ca/2025/12/29/nouveau-reglement-sur-les-criteres-de-fixation-de-loyer-recul-amelioration-ou-reforme-necessaire/", "law-firm summary citing TAL"),
            (2026, 3.1,  "https://ici.radio-canada.ca/nouvelle/2221781/loyers-tal-logement-augmentation-locataires", "press, TAL announcement (new CPI-based regulation)"),
        ], columns=["year", "tal_unheated_pct", "source", "verification"])

        def appears_in_pdf(year, rate):
            text = tal_text[year]
            # the PDFs write 4.0 as "4,0" and 1.28 as "1,28"
            candidates = {f"{rate:.1f}", f"{rate:.2f}".rstrip("0")}
            return any(re.search(r"non chauff\w*\s+" + re.escape(c.replace(".", ",")) + r"\s?%", text) for c in candidates)

        for row in tal_reference[tal_reference.verification == "pdf"].itertuples():
            ok = appears_in_pdf(row.year, row.tal_unheated_pct)
            print(row.year, row.tal_unheated_pct, "found in PDF" if ok else "NOT FOUND")
            assert ok, f"{row.year}: rate not found in the communiqué"
        tal_reference

        PERCENT = r"(-?\d,\d)\s?%"
        row_pattern = re.compile(r"\b(20[12]\d)\s+" + r"\s+".join([PERCENT] * 6) + r"\s+([\d ]{3,6}?)\s+" + PERCENT)
        granted_cols = ["year", "granted_heated_electric_pct", "granted_heated_gas_pct", "granted_heated_oil_pct",
                        "granted_unheated_pct", "granted_all_cases_pct", "granted_with_capex_pct",
                        "n_decisions", "inflation_pct"]

        def to_float(fr_number):
            return float(fr_number.replace(" ", "").replace(",", "."))

        granted = pd.DataFrame([[int(m[0])] + [to_float(x) for x in m[1:]] for m in row_pattern.findall(tal_text[2024])],
                               columns=granted_cols)
        granted["n_decisions"] = granted.n_decisions.astype(int)
        granted

        page = requests.get(ONTARIO_URL, headers=HTTP_HEADERS, timeout=60).text
        (RAW_ON / "rent-increase-guideline.html").write_text(page)
        page_text = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"(?s)<script.*?</script>|<style.*?</style>", "", page)))
        page_text = re.sub(r"\s+", " ", page_text)

        chart = page_text[page_text.index("Year guideline (%)"):]
        pairs = re.findall(r"\b((?:19|20)\d\d) (\d+(?:\.\d)?)\b", chart[:2000])
        ontario = (pd.DataFrame(pairs, columns=["year", "ontario_guideline_pct"]).astype({"year": int, "ontario_guideline_pct": float})
                   .drop_duplicates("year").sort_values("year"))
        ontario = ontario[ontario.year.between(FIRST_YEAR, LAST_YEAR + 1)]

        assert "occupied for the first time for residential purposes after November 15, 2018" in page_text
        print("Exemption clause found on the page.")
        ontario.set_index("year").T

        quebec = (tal_reference[["year", "tal_unheated_pct", "source"]]
                  .rename(columns={"tal_unheated_pct": "reference_rate_pct"})
                  .merge(granted[["year", "granted_unheated_pct", "inflation_pct"]], on="year", how="outer")
                  .assign(province="Quebec", rate_type="tal_estimate"))
        quebec["cpi_regime"] = quebec.year >= 2026

        ontario_tidy = (ontario.rename(columns={"ontario_guideline_pct": "reference_rate_pct"})
                        .assign(province="Ontario", rate_type="ontario_cap", source=ONTARIO_URL, cpi_regime=False))

        regulation = (pd.concat([quebec, ontario_tidy], ignore_index=True)
                      .query("@FIRST_YEAR <= year <= @LAST_YEAR + 1")
                      .sort_values(["province", "year"])
                      [["province", "year", "reference_rate_pct", "rate_type", "granted_unheated_pct",
                        "inflation_pct", "cpi_regime", "source"]])
        regulation.to_csv(CLEAN_DIR / "rent_regulation_annual.csv", index=False)

        pd.DataFrame({"ontario_exemption_date": [ONTARIO_EXEMPTION_DATE.date()],
                      "rule": ["units first occupied for residential purposes after this date are exempt from the guideline"],
                      "source": [ONTARIO_URL]}).to_csv(CLEAN_DIR / "ontario_exemption.csv", index=False)

        regulation.pivot(index="year", columns="province", values="reference_rate_pct")


def build_cmhc():
    """SCHL, Enquête sur les logements locatifs (rapports 2021-2025) : zones de Montréal et Ottawa, panel des villes,
    logements avec / sans rotation, zone SCHL de chaque immeuble. Sorties : clean/cmhc_*.csv."""
    with _in_external_dir():
        from pathlib import Path
        import re
        import numpy as np
        import pandas as pd
        import requests
        import matplotlib.pyplot as plt

        RAW_DIR = Path("raw/cmhc");  RAW_DIR.mkdir(parents=True, exist_ok=True)
        CLEAN_DIR = Path("clean");   CLEAN_DIR.mkdir(parents=True, exist_ok=True)

        BASE_URL = ("https://assets.cmhc-schl.gc.ca/sites/cmhc/professional/housing-markets-data-and-research/"
                    "housing-data-tables/rental-market/rental-market-report-data-tables/{year}/rmr-{geo}-{year}-en.xlsx")
        REPORT_YEARS = range(2021, 2026)          # earlier years are not published in this format
        GEOGRAPHIES = ["canada", "montreal", "ottawa"]
        HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (research; CodeML 2026)"}
        QUALITY_USABLE = {"a", "b", "c"}           # CMHC grade d = 'poor, use with caution'

        for year in REPORT_YEARS:
            for geo in GEOGRAPHIES:
                path = RAW_DIR / f"rmr-{geo}-{year}-en.xlsx"
                if not path.exists():
                    r = requests.get(BASE_URL.format(year=year, geo=geo), headers=HTTP_HEADERS, timeout=60)
                    r.raise_for_status()
                    path.write_bytes(r.content)
        print(sorted(p.name for p in RAW_DIR.glob("*.xlsx")))

        import re
        import numpy as np
        import pandas as pd
        import openpyxl

        ROW_ANCHORS = {"Zone", "Centre", "Year of Construction", "Structure Size", "Rent Quartile"}
        NOTE_PREFIXES = ("§", "*", "Quality", "a —", "**", "++", "Source", "©", "Note", "↑", "↓", "Y/N", "Data")
        QUALITY_CODES = {"a", "b", "c", "d"}
        SIGNIFICANCE_CODES = {"↑", "↓", "-", "Y", "N"}
        FLAG_CODES = {"**": "suppressed", "++": "not_significant", "-": "none"}

        def _clean(x):
            if x is None:
                return None
            s = str(x).replace("\n", " ").strip()
            return s if s else None

        def _grid(ws):
            grid = [[_clean(c) for c in row] for row in ws.iter_rows(values_only=True)]
            width = max((len(r) for r in grid), default=0)
            return [r + [None] * (width - len(r)) for r in grid]

        def _span_label(header_row, j, is_lowest, row_below=None):
            """Label of column j in one header row.

            CMHC places a group label (e.g. '2 Bedroom', 'Turnover units') in a single cell
            roughly centred over its columns, with no merged-cell information. A column takes the
            label whose position is nearest, ties going to the left label."""
            if is_lowest:
                return header_row[j]
            positions = [p for p, v in enumerate(header_row) if v is not None and p > 0]
            if not positions:
                return None
            below = {p for p, v in enumerate(row_below or []) if v is not None and p > 0}
            if set(positions) <= below:          # labels left-aligned on their group: forward fill
                left = [p for p in positions if p <= j]
                return header_row[left[-1]] if left else None
            left = [p for p in positions if p <= j]
            right = [p for p in positions if p > j]
            if not left:
                return header_row[right[0]]
            if not right:
                return header_row[left[-1]]
            pl, pr = left[-1], right[0]
            return header_row[pl] if j - pl <= pr - j else header_row[pr]

        def _is_data_row(row):
            label = row[0]
            return (label is not None and label not in ROW_ANCHORS and not label.startswith(NOTE_PREFIXES)
                    and not re.match(r"^\d+(\.\d+)*\s", label) and any(v is not None for v in row[1:]))

        def _to_number(s):
            try:
                return float(s.replace(",", ""))
            except (AttributeError, ValueError):
                return np.nan

        def parse_sheet(ws):
            grid = _grid(ws)
            title = next((r[0] for r in grid if r[0] and re.match(r"^\d+(\.\d+)*\s", r[0])), ws.title)
            title_idx = next(i for i, r in enumerate(grid) if r[0] == title) if title != ws.title else 0
            data_idx = [i for i, r in enumerate(grid) if i > title_idx and _is_data_row(r)]
            if not data_idx:
                return pd.DataFrame()
            first, last = data_idx[0], data_idx[-1]
            # stop at the first note row after data starts
            block = []
            for i in range(first, len(grid)):
                if grid[i][0] and grid[i][0].startswith(NOTE_PREFIXES):
                    break
                if _is_data_row(grid[i]):
                    block.append(i)
            header_rows = [i for i in range(title_idx + 1, first) if any(v is not None for v in grid[i][1:])]
            ncol = len(grid[0])

            def col_values(j):
                return {grid[i][j] for i in block if grid[i][j] is not None}

            rows = []
            for j in range(1, ncol):
                vals = col_values(j)
                if not vals or vals <= QUALITY_CODES or vals <= SIGNIFICANCE_CODES | {"**"}:
                    continue
                labels = []
                for k, i in enumerate(header_rows):
                    is_lowest = k == len(header_rows) - 1
                    below = None if is_lowest else grid[header_rows[k + 1]]
                    h = _span_label(grid[i], j, is_lowest, below)
                    if h and h not in labels:
                        labels.append(h)
                if not labels:
                    continue
                nxt = col_values(j + 1) if j + 1 < ncol else set()
                has_quality = bool(nxt) and nxt <= QUALITY_CODES
                for i in block:
                    raw = grid[i][j]
                    rows.append({"table_title": title, "row_label": grid[i][0], "column_label": " | ".join(labels),
                                 "raw": raw, "value": _to_number(raw),
                                 "flag": FLAG_CODES.get(raw),
                                 "quality": grid[i][j + 1] if has_quality else None})
            return pd.DataFrame(rows)

        def parse_workbook(path, sheets=None):
            wb = openpyxl.load_workbook(path)
            out = []
            for ws in wb.worksheets:
                if sheets and ws.title not in sheets:
                    continue
                df = parse_sheet(ws)
                if len(df):
                    df.insert(0, "sheet", ws.title)
                    out.append(df)
            return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

        def period_end_year(label):
            # 'Oct-23 to Oct-24' -> 2024 ; 'Oct-24' -> 2024 ; '2021-10-01 00:00:00' -> 2021
            m = re.findall(r"Oct-(\d{2})", label)
            if m:
                return 2000 + int(m[-1])
            m = re.findall(r"(20\d\d)-10-01", label)
            return int(m[-1]) if m else None

        long_frames = []
        for path in sorted(RAW_DIR.glob("rmr-*-en.xlsx")):
            geo, year = re.match(r"rmr-(\w+)-(\d{4})-en", path.stem).groups()
            df = parse_workbook(path)
            df.insert(0, "report_year", int(year))
            df.insert(0, "geo_file", geo)
            long_frames.append(df)
        cmhc_long = pd.concat(long_frames, ignore_index=True)
        cmhc_long["table_id"] = cmhc_long.sheet.str.replace("Table ", "", regex=False)
        cmhc_long["year"] = cmhc_long.column_label.map(period_end_year)
        cmhc_long.to_csv(CLEAN_DIR / "cmhc_rms_long.csv", index=False)
        print(f"{len(cmhc_long):,} cells from {cmhc_long[['geo_file','report_year']].drop_duplicates().shape[0]} workbooks")
        cmhc_long.groupby(["geo_file", "report_year"]).table_id.nunique().unstack()

        ZONE_TABLES = {"1.1.1": "vacancy_pct", "1.1.2": "avg_rent", "1.1.5": "fixed_sample_change_pct",
                       "1.1.6": "turnover_rate_pct", "1.1.9": None, "1.2.2": "avg_rent_by_construction"}
        BEDROOM_ALIASES = {"Bachelor": "Studio"}

        zone = cmhc_long[cmhc_long.geo_file.isin(["montreal", "ottawa"]) & cmhc_long.table_id.isin(ZONE_TABLES)].copy()
        parts = zone.column_label.str.split(" | ", regex=False)
        zone["bedroom"] = parts.str[0].replace(BEDROOM_ALIASES)
        zone["measure"] = zone.table_id.map(ZONE_TABLES)
        is_vacant_table = zone.table_id == "1.1.9"
        zone.loc[is_vacant_table, "measure"] = "avg_rent_" + parts[is_vacant_table].str[1].str.lower().str.replace(" ", "_")
        zone.loc[is_vacant_table, "year"] = zone.loc[is_vacant_table, "report_year"]
        zone = zone.rename(columns={"row_label": "zone"})

        zone_panel = (zone.dropna(subset=["year"])
                      .sort_values("report_year")
                      .drop_duplicates(["geo_file", "zone", "bedroom", "measure", "year"], keep="last")
                      [["geo_file", "zone", "bedroom", "year", "measure", "value", "flag", "quality", "report_year"]])
        zone_panel["year"] = zone_panel.year.astype(int)
        zone_panel.to_csv(CLEAN_DIR / "cmhc_zone_panel.csv", index=False)
        zone_panel.groupby(["geo_file", "measure"]).year.agg(["min", "max", "size"])

        building_zone_map = pd.DataFrame([
            ("Daniel-Johnson", "montreal", "Zone 19 - Chomedey/Sainte-Dorothée", "Laval (Zones 19-24)", "Montréal CMA"),
            ("Levesque",       "montreal", "Zone 19 - Chomedey/Sainte-Dorothée", "Laval (Zones 19-24)", "Montréal CMA"),
            ("Saint-Elzear",   "montreal", "Zone 19 - Chomedey/Sainte-Dorothée", "Laval (Zones 19-24)", "Montréal CMA"),
            ("Le Carlyle",     "montreal", "Zone 5 - Ct-des-Neiges/Mt-Royal/Outremont", "Montréal Island (Zones 1-18)", "Montréal CMA"),
            ("Westpark",       "montreal", "Zone 15 - Baie-d'Urfé/Beaconsfield etc.", "Montréal Island (Zones 1-18)", "Montréal CMA"),
            ("The Met",        "ottawa",   "Zone 1 - Downtown", "Former City of Ottawa (Zones 1-9)", "Ottawa CMA"),
        ], columns=["building", "geo_file", "zone", "sub_area", "cma"])

        known_zones = set(zone_panel.zone)
        missing = {z for col in ["zone", "sub_area", "cma"] for z in building_zone_map[col]} - known_zones
        assert not missing, f"zone names not found in CMHC tables: {missing}"
        building_zone_map.to_csv(CLEAN_DIR / "cmhc_building_zone_map.csv", index=False)
        building_zone_map

        CENTRE_MEASURES = {"Vacancy Rates": "vacancy_pct", "Turnover Rates": "turnover_rate_pct",
                           "Average Rent": "avg_rent_2bed", "Percentage Change": "fixed_sample_change_pct_2bed"}

        centre = cmhc_long[(cmhc_long.geo_file == "canada") & (cmhc_long.table_id == "1.0")].copy()
        centre["measure"] = centre.column_label.map(
            lambda label: next((v for k, v in CENTRE_MEASURES.items() if label.startswith(k)), None))
        centre = centre.rename(columns={"row_label": "centre"}).dropna(subset=["measure", "year"])
        centre["centre"] = centre.centre.str.replace(r"\s+\d+(,\d+)?\+?$", "", regex=True).str.strip()
        centre_panel = (centre.sort_values("report_year")
                        .drop_duplicates(["centre", "measure", "year"], keep="last")
                        .pivot_table(index=["centre", "year"], columns="measure", values="value")
                        .reset_index())
        centre_panel["year"] = centre_panel.year.astype(int)
        centre_panel["is_cma"] = centre_panel.centre.str.contains("CMA")
        centre_panel.to_csv(CLEAN_DIR / "cmhc_centre_panel.csv", index=False)
        print(centre_panel.centre.nunique(), "centres,", centre_panel.year.min(), "-", centre_panel.year.max())
        centre_panel[centre_panel.centre.isin(["Montréal CMA", "Ottawa-Gatineau CMA (Ont. part)", "Quebec", "Ontario"])]

        tvs = cmhc_long[(cmhc_long.geo_file == "canada") & (cmhc_long.table_id == "6.0")].copy()
        labels = tvs.column_label.str.split(" | ", regex=False)
        tvs["unit_status"] = np.where(labels.str[0].str.startswith("Turnover"), "turnover", "non_turnover")
        tvs["bedroom"] = labels.map(lambda p: next((x for x in p if x in {"Studio", "1 Bedroom", "2 Bedroom", "3 Bedroom +", "Total"}), "2 Bedroom"))
        tvs["is_level"] = tvs.column_label.str.contains("Average rent")
        tvs = tvs.rename(columns={"row_label": "centre"})
        tvs["centre"] = tvs.centre.str.replace(r"\s+\d+(,\d+)?\+?$", "", regex=True).str.strip()

        levels = (tvs[tvs.is_level].sort_values("report_year")
                  .drop_duplicates(["centre", "bedroom", "unit_status", "year"], keep="last")
                  .pivot_table(index=["centre", "bedroom", "year"], columns="unit_status", values="value")
                  .rename(columns={"turnover": "turnover_rent", "non_turnover": "non_turnover_rent"}))
        changes = (tvs[~tvs.is_level].sort_values("report_year")
                   .drop_duplicates(["centre", "bedroom", "unit_status", "year"], keep="last")
                   .pivot_table(index=["centre", "bedroom", "year"], columns="unit_status", values="value")
                   .rename(columns={"turnover": "turnover_change_pct", "non_turnover": "non_turnover_change_pct"}))
        turnover_vs_sitting = levels.join(changes, how="outer").reset_index()
        turnover_vs_sitting["turnover_premium_pct"] = (turnover_vs_sitting.turnover_rent / turnover_vs_sitting.non_turnover_rent - 1) * 100
        turnover_vs_sitting["year"] = turnover_vs_sitting.year.astype(int)
        turnover_vs_sitting.to_csv(CLEAN_DIR / "cmhc_turnover_vs_sitting.csv", index=False)
        turnover_vs_sitting[turnover_vs_sitting.centre.isin(["Montréal CMA", "Ottawa-Gatineau CMA (Ont. part)"])
                            & (turnover_vs_sitting.bedroom == "2 Bedroom")].round(1)

        fixed = zone_panel[(zone_panel.measure == "fixed_sample_change_pct") & (zone_panel.bedroom == "Total")]
        watch = list(dict.fromkeys(building_zone_map.zone.tolist() + building_zone_map.cma.tolist()))
        table = fixed[fixed.zone.isin(watch)].pivot_table(index="year", columns="zone", values="value")
        ax = table.plot(marker="o", figsize=(10, 4.8))
        ax.set_ylabel("fixed-sample change in average rent (%)")
        ax.set_title("CMHC — same-sample rent growth, all bedroom types, October to October")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        plt.tight_layout(); plt.close()
        table.round(1)


def build_reits():
    """FPI concurrentes (InterRent, Killam, CAPREIT, Minto) : hausses à périmètre constant, renouvellements, relocations,
    chaque chiffre vérifié dans son document source. Sortie : clean/reit_metrics.csv."""
    with _in_external_dir():
        import re
        import html
        from pathlib import Path

        import pandas as pd
        import requests
        from pypdf import PdfReader
        import matplotlib.pyplot as plt

        RAW_DIR = Path("raw/reits");  RAW_DIR.mkdir(parents=True, exist_ok=True)
        CLEAN_DIR = Path("clean");    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (research; CodeML 2026)"}
        PDF_PAGES_TO_READ = 80        # metrics live in the first part of each MD&A

        SOURCES = {
            # key: (url, local file)
            "interrent_2019": ("https://assets.irent.com/image/upload/Reports/Quarterly/2019_Q4_MDA.pdf", "interrent_2019_Q4_MDA.pdf"),
            "interrent_2020": ("https://assets.irent.com/image/upload/Reports/Quarterly/2020_Q4_MDA.pdf", "interrent_2020_Q4_MDA.pdf"),
            "interrent_2021": ("https://assets.irent.com/image/upload/Reports/Quarterly/2021_Q4_MDA.pdf", "interrent_2021_Q4_MDA.pdf"),
            "interrent_2022": ("https://assets.irent.com/image/upload/Reports/Quarterly/2022_Q4_MDA.pdf", "interrent_2022_Q4_MDA.pdf"),
            "interrent_2023": ("https://assets.irent.com/image/upload/Reports/Quarterly/2023_Q4_MDA.pdf", "interrent_2023_Q4_MDA.pdf"),
            "interrent_2024": ("https://assets.irent.com/image/upload/Reports/Quarterly/2024_Q4_MDA.pdf", "interrent_2024_Q4_MDA.pdf"),
            "killam_2023":    ("https://investors.killamreit.com/image/Killam+Q4+12-31-2023+MDA.pdf", "killam_2023_Q4_MDA.pdf"),
            "killam_2025":    ("https://investors.killamreit.com/image/Killam+Q4+12-31-2025+MDA.pdf", "killam_2025_Q4_MDA.pdf"),
            "capreit_2021":   ("https://s206.q4cdn.com/217547279/files/doc_news/2022/02/1/2021-Press-Release.pdf", "capreit_2021_press_release.pdf"),
            "capreit_2023":   ("https://www.capreit.ca/wp-content/uploads/2024/02/CAPREIT-2023-Annual-Report-Web.pdf", "capreit_2023_annual_report.pdf"),
            "capreit_2025":   ("https://s206.q4cdn.com/217547279/files/doc_financials/2025/ar/CAPREIT-Conference-Call-Q4-2025.pdf", "capreit_2025_q4_conference_call.pdf"),
            "capreit_2022_web": ("https://www.globenewswire.com/news-release/2023/02/22/2613800/0/en/CAPREIT-Reports-Fourth-Quarter-and-Year-End-2022-Results.html", None),
            "capreit_2024_web": ("https://www.globenewswire.com/news-release/2025/02/13/3026342/0/en/CAPREIT-Reports-Fourth-Quarter-and-Year-End-2024-Results.html", None),
            "capreit_2025_web": ("https://www.globenewswire.com/news-release/2026/02/12/3237776/0/en/CAPREIT-Reports-Fourth-Quarter-and-Year-End-2025-Results.html", None),
            "minto_2023":     ("https://www.newswire.ca/news-releases/minto-apartment-reit-reports-2023-fourth-quarter-and-year-end-financial-results-805060321.html", "minto_2023_q4_release.html"),
            "minto_2024":     ("https://www.newswire.ca/news-releases/minto-apartment-reit-reports-2024-fourth-quarter-and-year-end-financial-results-835695342.html", "minto_2024_q4_release.html"),
            "minto_2025":     ("https://www.newswire.ca/news-releases/minto-apartment-reit-reports-2025-fourth-quarter-and-year-end-financial-results-834259940.html", "minto_2025_q4_release.html"),
        }

        def fetch(key):
            url, filename = SOURCES[key]
            if filename is None:
                return None
            path = RAW_DIR / filename
            if not path.exists():
                try:
                    r = requests.get(url, headers=HTTP_HEADERS, timeout=90)
                    r.raise_for_status()
                    path.write_bytes(r.content)
                except requests.RequestException as err:
                    print(f"  could not download {key}: {err}")
                    return None
            return path

        def document_text(path):
            if path is None:
                return ""
            if path.suffix == ".pdf":
                pages = PdfReader(path).pages[:PDF_PAGES_TO_READ]
                text = " ".join((p.extract_text() or "") for p in pages)
            else:
                raw = re.sub(r"(?s)<script.*?</script>|<style.*?</style>", "", path.read_text(errors="ignore"))
                text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
            return re.sub(r"\s+", " ", text)

        texts = {key: document_text(fetch(key)) for key in SOURCES}
        pd.Series({k: len(v) for k, v in texts.items()}, name="characters_of_text")

        MONEY, PCT = r"\$\s?([\d,]+)", r"([+\-−]?\s?\d+\.\d)\s?%"
        ROW = r"(?:\s?\(\d\))?\s+" + r"\s+".join([MONEY, MONEY, PCT, MONEY, MONEY, PCT])
        INTERRENT_REGIONS = {"Ottawa": ["National Capital Region", "Ottawa"],
                             "Montreal": ["Greater Montr[eé]al Area", "Montreal"],
                             "Total": ["Total"]}

        def to_float(s):
            return float(s.replace(",", "").replace("−", "-").replace(" ", ""))

        interrent_rows = []
        for year in range(2019, 2025):
            text = texts[f"interrent_{year}"]
            for region, aliases in INTERRENT_REGIONS.items():
                match = next((m for alias in aliases for m in re.findall(alias + ROW, text)), None)
                if match is None:
                    print(f"InterRent {year} {region}: not found")
                    continue
                sp_now, sp_prev, sp_change = to_float(match[3]), to_float(match[4]), to_float(match[5])
                interrent_rows.append({"reit": "InterRent", "region": region, "year": year,
                                       "metric": "same_property_amr_growth_pct", "value": sp_change,
                                       "amr_now": sp_now, "amr_prev": sp_prev,
                                       "source": SOURCES[f"interrent_{year}"][0], "verified": True})
        interrent = pd.DataFrame(interrent_rows)
        # self-check: the published % change must match the two published rents
        interrent["recomputed_pct"] = (interrent.amr_now / interrent.amr_prev - 1) * 100
        assert (interrent.recomputed_pct - interrent.value).abs().max() < 0.15, "parsed rents and % disagree"
        interrent.pivot(index="year", columns="region", values="value")

        transcribed = pd.DataFrame([
            # reit,    region,   year, metric,                         value, source key,     keyword near the number
            ("Killam",  "Total", 2021, "renewal_increase_pct",          1.5, "killam_2025", "Historical Annual Same Property Rental Rate Growth"),
            ("Killam",  "Total", 2022, "renewal_increase_pct",          1.8, "killam_2023", "Lease renewal"),
            ("Killam",  "Total", 2023, "renewal_increase_pct",          2.8, "killam_2023", "Lease renewal"),
            ("Killam",  "Total", 2024, "renewal_increase_pct",          4.3, "killam_2025", "Lease renewal"),
            ("Killam",  "Total", 2025, "renewal_increase_pct",          3.5, "killam_2025", "Lease renewal"),
            ("Killam",  "Total", 2021, "turnover_increase_pct",         7.8, "killam_2025", "Historical Annual Same Property Rental Rate Growth"),
            ("Killam",  "Total", 2022, "turnover_increase_pct",        10.0, "killam_2023", "Unit turn"),
            ("Killam",  "Total", 2023, "turnover_increase_pct",        16.4, "killam_2023", "Unit turn"),
            ("Killam",  "Total", 2024, "turnover_increase_pct",        19.8, "killam_2025", "Unit turn"),
            ("Killam",  "Total", 2025, "turnover_increase_pct",        10.9, "killam_2025", "Unit turn"),
            ("Killam",  "Total", 2021, "weighted_increase_pct",         3.0, "killam_2025", "Historical Annual Same Property Rental Rate Growth"),
            ("Killam",  "Total", 2022, "weighted_increase_pct",         3.5, "killam_2023", "Rental increase (weighted average)"),
            ("Killam",  "Total", 2023, "weighted_increase_pct",         5.4, "killam_2023", "Rental increase (weighted average)"),
            ("Killam",  "Total", 2024, "weighted_increase_pct",         7.0, "killam_2025", "Rental increase (weighted average)"),
            ("Killam",  "Total", 2025, "weighted_increase_pct",         5.0, "killam_2025", "Rental increase (weighted average)"),
            ("CAPREIT", "Canada", 2020, "turnover_increase_pct",        7.9, "capreit_2021", "urnover"),
            ("CAPREIT", "Canada", 2021, "turnover_increase_pct",        5.9, "capreit_2021", "urnover"),
            ("CAPREIT", "Canada", 2020, "renewal_increase_pct",         1.3, "capreit_2021", "enewal"),
            ("CAPREIT", "Canada", 2021, "renewal_increase_pct",         1.4, "capreit_2021", "enewal"),
            ("CAPREIT", "Canada", 2021, "same_property_amr_growth_pct", 1.9, "capreit_2021", "AMR"),
            ("CAPREIT", "Canada", 2022, "turnover_increase_pct",       14.5, "capreit_2023", "urnover"),
            ("CAPREIT", "Canada", 2022, "renewal_increase_pct",         1.4, "capreit_2023", "enewal"),
            ("CAPREIT", "Canada", 2022, "same_property_amr_growth_pct", 4.3, "capreit_2022_web", "AMR"),
            ("CAPREIT", "Canada", 2023, "turnover_increase_pct",       27.7, "capreit_2023", "urnover"),
            ("CAPREIT", "Canada", 2023, "renewal_increase_pct",         2.7, "capreit_2023", "enewal"),
            ("CAPREIT", "Canada", 2024, "turnover_increase_pct",       18.8, "capreit_2025", "urnover"),
            ("CAPREIT", "Canada", 2024, "renewal_increase_pct",         3.6, "capreit_2025", "enewal"),
            ("CAPREIT", "Canada", 2024, "same_property_amr_growth_pct", 6.0, "capreit_2024_web", "AMR"),
            ("CAPREIT", "Canada", 2025, "turnover_increase_pct",        4.2, "capreit_2025", "Blended total"),
            ("CAPREIT", "Canada", 2025, "turnover_increase_tenure_lt2y_pct", -6.3, "capreit_2025", "ess than two years"),
            ("CAPREIT", "Canada", 2025, "turnover_increase_tenure_ge2y_pct", 16.0, "capreit_2025", "wo years or longer"),
            ("CAPREIT", "Canada", 2025, "renewal_increase_pct",         3.2, "capreit_2025", "enewal"),
            ("CAPREIT", "Canada", 2025, "same_property_amr_growth_pct", 3.8, "capreit_2025_web", "AMR"),
            ("Minto",   "Total", 2023, "same_property_amr_growth_pct",  6.8, "minto_2023", "average monthly rent"),
            ("Minto",   "Total", 2024, "same_property_amr_growth_pct",  5.5, "minto_2024", "average monthly rent"),
            ("Minto",   "Total", 2025, "same_property_amr_growth_pct",  3.9, "minto_2025", "average monthly rent"),
            ("Minto",   "Total", 2023, "q4_gain_on_lease_pct",         16.1, "minto_2023", "expiring rents"),
            ("Minto",   "Total", 2024, "q4_gain_on_lease_pct",         11.2, "minto_2024", "expiring rents"),
            ("Minto",   "Total", 2025, "q4_gain_on_lease_pct",          0.9, "minto_2025", "expiring rents"),
        ], columns=["reit", "region", "year", "metric", "value", "source_key", "keyword"])

        NEARBY_CHARS = 600

        def verify(row):
            text = texts.get(row.source_key, "")
            if not text:
                return False
            magnitude = re.escape(f"{abs(row.value):.1f}") + r"\s?%"
            number = (r"[-−(]\s?" + magnitude) if row.value < 0 else magnitude
            for hit in re.finditer(re.escape(row.keyword), text):
                window = text[max(0, hit.start() - NEARBY_CHARS): hit.end() + NEARBY_CHARS]
                if re.search(number, window):
                    return True
            return False

        transcribed["verified"] = transcribed.apply(verify, axis=1)
        transcribed["source"] = transcribed.source_key.map(lambda k: SOURCES[k][0])
        print(transcribed.groupby("reit").verified.agg(["sum", "size"]))
        transcribed[~transcribed.verified][["reit", "year", "metric", "value", "source"]]

        COLUMNS = ["reit", "region", "year", "metric", "value", "verified", "source"]
        reit_metrics = (pd.concat([interrent[COLUMNS], transcribed[COLUMNS]], ignore_index=True)
                        .sort_values(["reit", "region", "metric", "year"]))
        reit_metrics.to_csv(CLEAN_DIR / "reit_metrics.csv", index=False)
        reit_metrics.pivot_table(index=["reit", "region", "metric"], columns="year", values="value")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
        for metric, ax in zip(["turnover_increase_pct", "renewal_increase_pct"], axes):
            data = reit_metrics[reit_metrics.metric == metric].pivot_table(index="year", columns="reit", values="value")
            data.plot(marker="o", ax=ax)
            ax.set_title(metric.replace("_", " "))
            ax.set_ylabel("%"); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.close()

        amr = reit_metrics[reit_metrics.metric == "same_property_amr_growth_pct"].pivot_table(
            index="year", columns=["reit", "region"], values="value")
        amr.round(1)


def build_kaggle():
    """Kaggle, 25000+ Canadian rental housing market (juin 2024, RentFaster). Zip à placer dans external/raw/kaggle/.
    Sorties : clean/kaggle_listings_clean.csv (annonces nettoyées, sans adresse ni lien ; non redistribué)
    et clean/kaggle_rent_by_city.csv (résumé par ville et nombre de chambres)."""
    import re
    import zipfile
    import numpy as np
    import pandas as pd
    with _in_external_dir():
        raw_dir, clean_dir = Path("raw/kaggle"), Path("clean")
        zip_path, csv_name = raw_dir / "kaggle_25000_canadian_rentals_june2024.zip", "rentfaster.csv"
        snapshot_date = pd.Timestamp("2024-06-15")            # date du fichier = date de collecte
        min_rent, max_rent, min_sqft, max_sqft = 300, 15_000, 150, 6_000   # bornes de plausibilité
        residential = {"Apartment", "Condo Unit", "Townhouse", "House", "Basement", "Main Floor", "Duplex", "Loft", "Acreage"}
        cities = ["Montréal", "Laval", "Pointe-Claire", "Ottawa"]
        csv_path = raw_dir / csv_name
        if csv_path.exists():
            raw = pd.read_csv(csv_path)
        else:
            with zipfile.ZipFile(zip_path) as zf:
                raw = pd.read_csv(zf.open(csv_name))

        def first_number(value):
            if pd.isna(value):
                return np.nan
            match = re.search(r"\d+(?:[.,]\d+)?", str(value).replace(",", ""))
            return float(match.group(0)) if match else np.nan

        def parse_beds(value):
            return 0.0 if str(value).lower().startswith("studio") else first_number(value)

        listings = raw.drop_duplicates().rename(columns={"price": "asking_rent", "sq_feet": "sqft_raw", "type": "property_type"})
        listings["beds"] = listings.beds.map(parse_beds)
        listings["baths"] = listings.baths.map(first_number)
        listings["sqft"] = listings.sqft_raw.map(first_number)
        listings.loc[~listings.sqft.between(min_sqft, max_sqft), "sqft"] = np.nan
        listings = listings[listings.property_type.isin(residential) & listings.asking_rent.between(min_rent, max_rent)].copy()
        listings["rent_per_sqft"] = listings.asking_rent / listings.sqft
        listings["is_long_term"] = listings.lease_term.eq("Long Term")
        listings["is_furnished"] = listings.furnishing.str.startswith("Furnished", na=False)
        listings["snapshot_date"] = snapshot_date
        keep = ["city", "province", "latitude", "longitude", "property_type", "is_long_term", "is_furnished",
                "beds", "baths", "sqft", "asking_rent", "rent_per_sqft", "snapshot_date"]
        listings[keep].to_csv(clean_dir / "kaggle_listings_clean.csv", index=False)
        comparable = listings[listings.property_type.isin({"Apartment", "Condo Unit"}) & listings.is_long_term
                              & ~listings.is_furnished & listings.city.isin(cities) & listings.beds.between(0, 3)]
        (comparable.groupby(["city", "beds"])
         .agg(annonces=("asking_rent", "size"), loyer_median=("asking_rent", "median"),
              superficie_mediane=("sqft", "median"), loyer_median_pi2=("rent_per_sqft", "median"))
         .reset_index().round(2).to_csv(clean_dir / "kaggle_rent_by_city.csv", index=False))
        return len(raw), len(listings)


SOURCES = {"IPC (StatCan)": build_cpi, "TAL et Ontario": build_regulation, "SCHL": build_cmhc,
           "FPI concurrentes": build_reits, "Kaggle": build_kaggle}


def build_all(include_kaggle=True):
    """Reconstruit toutes les sources ; Kaggle seulement si le zip est présent."""
    for name, build in SOURCES.items():
        if build is build_kaggle and not (include_kaggle and (EXTERNAL_DIR / "raw/kaggle").exists()):
            print(f"{name} : ignoré (zip Kaggle absent)")
            continue
        build()
        print(f"{name} : reconstruit")
