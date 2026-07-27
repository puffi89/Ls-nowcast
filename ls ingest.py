"""
LUS-Ingest - liest die Umsatztabelle und erzeugt forecast.json fuer das Frontend.

Quelle: entweder die veroeffentlichte CSV-URL des Sheets oder eine lokale xlsx.
    python ls_ingest.py LUS.xlsx
    python ls_ingest.py "https://docs.google.com/.../pub?gid=0&single=true&output=csv"
"""
import sys, json, datetime as dt
import pandas as pd, numpy as np

COLS = ['datum','lsx_aktien','lsx_etf','lsx_anleihen','lsx_fonds','lsx_summe',
        'tc_aktien','tc_etf','tc_anleihen','tc_fonds',
        'tc_wikifolio','tc_turbo','tc_zertifikate','tc_optionen']

# Sheet untererfasst das gemeldete Volumen um rund 5 % - L&S quotiert auch in
# Frankfurt, Wien, Stuttgart und Bern. Kalibriert auf 20 Quartale, R2 = 0,9986.
BRUECKE = dict(achse=778.0, steigung=1.0383)

# Ertragsmodell, kalibriert auf die Abschluesse 2022-2025
CAL = dict(kg_fix=5.86, kg_var=0.324, ag_fix=8.14, ag_var=0.416,
           ag_tax=0.39, ag_other=0.58, konz_off=-3.96, konz_var=0.1683,
           konz_tax=0.687, aktien=9.438)

# Margenmodell: der Produktmix erklaert die Marge, nicht die Volatilitaet.
# log(bp) = -0,960 + 1,575 x log(Aktien/ETF).  R2 = 0,37 auf 20 Quartalen,
# out-of-sample MAPE 41 % - grobe Richtung, keine Punktschaetzung.
# Die Potenz 1,575 explodiert ausserhalb des Stuetzbereichs, deshalb geklemmt.
MIX = dict(exponent=1.575, referenz=3.677, unten=2.98, oben=6.34)


def lade(quelle: str) -> pd.DataFrame:
    """Liest alle Jahresblaetter plus das laufende Blatt und dedupliziert."""
    if quelle.startswith("http"):
        teile = [pd.read_csv(quelle, skiprows=3, header=None, usecols=range(14))]
    else:
        xl = pd.ExcelFile(quelle)
        blaetter = [s for s in xl.sheet_names if s.isdigit()] + \
                   [s for s in xl.sheet_names if s == 'TagesStatistiken']
        teile = [pd.read_excel(xl, sheet_name=s, skiprows=3, header=None,
                               usecols=range(14)) for s in blaetter]
    df = pd.concat(teile)
    df.columns = COLS
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    df = df.dropna(subset=['datum'])
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.drop_duplicates('datum', keep='last').sort_values('datum').reset_index(drop=True)

    # Zwei Bloecke: Market Making der KG, Eigenemissionen der AG
    df['kassa'] = df[['lsx_aktien','lsx_etf','lsx_anleihen','lsx_fonds',
                      'tc_aktien','tc_etf','tc_anleihen','tc_fonds']].sum(axis=1)
    df['lsx']   = df[['lsx_aktien','lsx_etf','lsx_anleihen','lsx_fonds']].sum(axis=1)
    df['tc']    = df[['tc_aktien','tc_etf','tc_anleihen','tc_fonds']].sum(axis=1)
    df['deriv'] = df[['tc_wikifolio','tc_turbo','tc_zertifikate','tc_optionen']].sum(axis=1)
    # Handelstag = Werktag mit Umsatz. Wochenendhandel laeuft mit, zaehlt aber
    # nicht als eigener Handelstag - sonst verzerrt die Normierung.
    df['ht'] = (df.datum.dt.weekday < 5) & (df.kassa > 0)
    # Produktmix: Aktienumsatz je Euro ETF-Umsatz. Aktienhandel traegt breite
    # Spreads, ETF- und Sparplanflow ist margenarm.
    akt = df[['lsx_aktien', 'tc_aktien']].sum(axis=1)
    etf = df[['lsx_etf', 'tc_etf']].sum(axis=1)
    df['mix'] = np.where(etf > 0, akt / etf.replace(0, np.nan), np.nan)
    return df


def margenfaktor(mix: float):
    """Mix -> Multiplikator auf die Referenzmarge. Klemmt am Stuetzbereich."""
    roh = max(MIX['unten'], min(MIX['oben'], mix))
    faktor = (roh / MIX['referenz']) ** MIX['exponent']
    return faktor, (mix < MIX['unten'] or mix > MIX['oben'])


def pruefe(df: pd.DataFrame) -> list:
    """Datenqualitaet. Gibt Warnungen zurueck, bricht nicht ab."""
    w = []
    luecken = df[df[COLS[1:]].isna().any(axis=1)]
    if len(luecken):
        w.append(f"{len(luecken)} Tage mit fehlenden Zellen, zuletzt {luecken.datum.max().date()}")
    letzte = df.datum.max().date()
    alter = (dt.date.today() - letzte).days
    if alter > 4:
        w.append(f"Daten sind {alter} Tage alt (letzter Satz {letzte})")
    werktage = pd.bdate_range(df.datum.min(), df.datum.max())
    fehlend = len(set(werktage.date) - set(df.datum.dt.date))
    if fehlend > 15:
        w.append(f"{fehlend} Werktage ohne Zeile - Feiertage oder Ausfaelle")
    return w


def _mixwarnung(mix, geklemmt):
    if not geklemmt:
        return []
    seite = "unter" if mix < MIX['unten'] else "ueber"
    return [f"Produktmix {mix:.2f} liegt {seite} dem Stuetzbereich "
            f"({MIX['unten']}-{MIX['oben']}); Margenfaktor wurde geklemmt"]


def run_rate(df: pd.DataFrame, seit: str, halbwertszeit: int = 10) -> float:
    """Exponentiell gewichteter Tagesdurchschnitt seit dem Strukturbruch."""
    g = df[(df.datum >= seit) & df.ht]
    if g.empty:
        return 0.0
    gew = 0.5 ** ((len(g) - 1 - np.arange(len(g))) / halbwertszeit)
    return float(np.average(g.kassa.values, weights=gew))


def jahresvolumen(tagesrate: float, handelstage: int = 255) -> float:
    """Sheet-Tagesrate -> gemeldetes Jahresvolumen in Mio EUR."""
    return BRUECKE['achse'] + BRUECKE['steigung'] * tagesrate * handelstage


def ertrag(tc_volumen_mio: float, bp: float, sp_he: float, payout: float = 0.40) -> dict:
    c = CAL
    tc_he = tc_volumen_mio * bp / 1e4
    kg_erg = tc_he - (c['kg_fix'] + c['kg_var'] * tc_he)
    ag_op = sp_he - (c['ag_fix'] + c['ag_var'] * sp_he)
    ag_ju = (ag_op + kg_erg) * (1 - c['ag_tax']) - c['ag_other']
    konz_he = tc_he + sp_he
    egt = ag_op + kg_erg + c['konz_off'] + c['konz_var'] * konz_he
    return dict(
        tc_handelsergebnis=round(tc_he, 1), kg_ergebnis=round(kg_erg, 1),
        ag_jahresueberschuss=round(ag_ju, 1),
        dividende_je_aktie=round(max(0, ag_ju) * payout / c['aktien'], 2),
        egt=round(egt, 1),
        eps=round(egt * c['konz_tax'] / c['aktien'], 2),
    )


def aktueller_mix(df: pd.DataFrame, seit: str) -> float:
    g = df[(df.datum >= seit) & df.ht]
    akt = g[['lsx_aktien', 'tc_aktien']].sum().sum()
    etf = g[['lsx_etf', 'tc_etf']].sum().sum()
    return float(akt / etf) if etf else float('nan')


def main(quelle: str, bruch: str = "2026-07-02", bp_ref: float = 5.4, sp_he: float = 50.0):
    df = lade(quelle)
    rate = run_rate(df, bruch)
    vol = jahresvolumen(rate)
    mix = aktueller_mix(df, bruch)
    faktor, geklemmt = margenfaktor(mix)
    bp = bp_ref * faktor
    out = dict(
        stand=str(df.datum.max().date()),
        warnungen=pruefe(df) + _mixwarnung(mix, geklemmt),
        tagesrate_mio=round(rate, 1),
        handelstage_seit_bruch=int(df[(df.datum >= bruch) & df.ht].shape[0]),
        volumen_jahresrate_mio=round(vol, 0),
        anteil_von_2025=round(vol / 331690, 4),
        mix_aktien_je_etf=round(mix, 2),
        margenfaktor=round(faktor, 2),
        mix_ausserhalb_stuetzbereich=geklemmt,
        annahmen=dict(referenzmarge_bp=bp_ref, abgeleitete_marge_bp=round(bp, 2),
                      sp_handelsergebnis_mio=sp_he),
        prognose=ertrag(vol, bp, sp_he),
        verlauf=[dict(datum=str(r.datum.date()), kassa=round(r.kassa, 1),
                      deriv=round(r.deriv, 2))
                 for r in df[df.datum >= '2026-01-01'].itertuples()],
    )
    with open("forecast.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'verlauf'},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/LUS.xlsx")
