"""
Financial Statement Puller — SEC EDGAR Only
"""
from __future__ import annotations
from datetime import datetime

import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.set_page_config(page_title="SEC Financial Puller", layout="wide")
st.title("SEC Financial Statement Puller")
st.caption("Income Statement · Balance Sheet · Cash Flow — Annual & Quarterly · Download as Excel")

HEADERS = {"User-Agent": "financial-research-app contact@example.com"}

INCOME_STMT_CONCEPTS = [
    ("Revenue",             ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                             "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"]),
    ("Cost of Revenue",     ["CostOfRevenue", "CostOfGoodsSold"]),
    ("Gross Profit",        ["GrossProfit"]),
    ("R&D Expense",         ["ResearchAndDevelopmentExpense"]),
    ("SG&A Expense",        ["SellingGeneralAndAdministrativeExpense"]),
    ("Operating Expenses",  ["OperatingExpenses"]),
    ("Operating Income",    ["OperatingIncomeLoss"]),
    ("Interest Expense",    ["InterestExpense", "InterestAndDebtExpense"]),
    ("Other Inc/Exp",       ["NonoperatingIncomeExpense"]),
    ("Pre-Tax Income",      ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"]),
    ("Income Tax",          ["IncomeTaxExpenseBenefit"]),
    ("Net Income",          ["NetIncomeLoss", "ProfitLoss"]),
    ("EPS Basic",           ["EarningsPerShareBasic"]),
    ("EPS Diluted",         ["EarningsPerShareDiluted"]),
    ("Shares Basic",        ["WeightedAverageNumberOfSharesOutstandingBasic"]),
    ("Shares Diluted",      ["WeightedAverageNumberOfDilutedSharesOutstanding"]),
]

BALANCE_SHEET_CONCEPTS = [
    ("Cash & Equivalents",        ["CashAndCashEquivalentsAtCarryingValue", "Cash"]),
    ("Short-Term Investments",    ["ShortTermInvestments", "AvailableForSaleSecuritiesCurrent"]),
    ("Accounts Receivable",       ["AccountsReceivableNetCurrent"]),
    ("Inventory",                 ["InventoryNet"]),
    ("Total Current Assets",      ["AssetsCurrent"]),
    ("PP&E, Net",                 ["PropertyPlantAndEquipmentNet"]),
    ("Goodwill",                  ["Goodwill"]),
    ("Intangible Assets",         ["IntangibleAssetsNetExcludingGoodwill"]),
    ("Total Assets",              ["Assets"]),
    ("Accounts Payable",          ["AccountsPayableCurrent"]),
    ("Short-Term Debt",           ["ShortTermBorrowings", "NotesPayableCurrent"]),
    ("Total Current Liabilities", ["LiabilitiesCurrent"]),
    ("Long-Term Debt",            ["LongTermDebt", "LongTermDebtNoncurrent"]),
    ("Total Liabilities",         ["Liabilities"]),
    ("Retained Earnings",         ["RetainedEarningsAccumulatedDeficit"]),
    ("Total Equity",              ["StockholdersEquity", "StockholdersEquityAttributableToParent"]),
    ("Total Liab. & Equity",      ["LiabilitiesAndStockholdersEquity"]),
]

CASHFLOW_CONCEPTS = [
    ("Net Income",           ["NetIncomeLoss", "ProfitLoss"]),
    ("D&A",                  ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"]),
    ("Stock-Based Comp",     ["ShareBasedCompensation"]),
    ("Cash from Operations", ["NetCashProvidedByUsedInOperatingActivities"]),
    ("CapEx",                ["PaymentsToAcquirePropertyPlantAndEquipment"]),
    ("Acquisitions",         ["PaymentsToAcquireBusinessesNetOfCashAcquired"]),
    ("Cash from Investing",  ["NetCashProvidedByUsedInInvestingActivities"]),
    ("Debt Issued",          ["ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromIssuanceOfDebt"]),
    ("Debt Repaid",          ["RepaymentsOfLongTermDebt", "RepaymentsOfDebt"]),
    ("Stock Repurchased",    ["PaymentsForRepurchaseOfCommonStock"]),
    ("Dividends Paid",       ["PaymentsOfDividends"]),
    ("Cash from Financing",  ["NetCashProvidedByUsedInFinancingActivities"]),
    ("Net Change in Cash",   ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect"]),
]


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_cik(ticker: str) -> tuple[str, str]:
    resp = requests.get("https://www.sec.gov/files/company_tickers.json",
                        headers=HEADERS, timeout=10)
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10), entry["title"]
    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR.")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_facts(cik: str) -> dict:
    resp = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                        headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_filings_list(cik: str) -> dict:
    resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                        headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _parse_date(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def _annual_label(date_str: str) -> str:
    d = _parse_date(date_str)
    return f"FY{d.year}" if d else date_str


def _quarter_label(date_str: str) -> str:
    d = _parse_date(date_str)
    if not d:
        return date_str
    q = (d.month - 1) // 3 + 1
    return f"Q{q}'{d.strftime('%y')}"


def extract_series(facts: dict, concepts: list[str], period_type: str,
                   fact_type: str = "duration") -> pd.Series | None:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        if concept not in gaap:
            continue
        units = gaap[concept].get("units", {})
        unit_data = (units.get("USD") or units.get("shares") or
                     units.get("USD/shares") or next(iter(units.values()), []))
        rows = []
        for item in unit_data:
            form = item.get("form", "")
            if period_type == "annual" and form not in ("10-K", "10-K/A"):
                continue
            if period_type == "quarterly" and form not in ("10-Q", "10-Q/A"):
                continue
            end = item.get("end", "")
            val = item.get("val")
            if val is None or not end:
                continue
            if fact_type == "instant":
                if item.get("start"):
                    continue
            else:
                start = item.get("start", "")
                if not start:
                    continue
                d_s, d_e = _parse_date(start), _parse_date(end)
                if d_s and d_e:
                    days = (d_e - d_s).days
                    if period_type == "annual" and not (330 <= days <= 400):
                        continue
                    if period_type == "quarterly" and not (60 <= days <= 105):
                        continue
            rows.append({"end": end, "val": val, "accn": item.get("accn", "")})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = df.sort_values("accn").drop_duplicates(subset=["end"], keep="last")
        return df.set_index("end")["val"]
    return None


def compute_ltm(facts: dict, concept_map: list, fact_type: str) -> pd.Series:
    """LTM = sum of last 4 quarters (IS/CF) or most recent snapshot (BS/instant)."""
    result = {}
    for label, concepts in concept_map:
        s = extract_series(facts, concepts, "quarterly", fact_type)
        if s is None or s.empty:
            continue
        s_sorted = s[sorted(s.index, reverse=True)]
        if fact_type == "instant":
            result[label] = float(s_sorted.iloc[0])
        else:
            last4 = s_sorted.iloc[:4]
            if not last4.empty:
                result[label] = float(last4.sum())
    return pd.Series(result)


def build_statement(facts: dict, concept_map: list, period_type: str,
                    n_periods: int, fact_type: str = "duration",
                    include_ltm: bool = False) -> pd.DataFrame:
    rows: dict = {}
    for label, concepts in concept_map:
        s = extract_series(facts, concepts, period_type, fact_type)
        if s is not None and not s.empty:
            rows[label] = s
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).T
    df = df[sorted(df.columns, reverse=True)]
    df = df.dropna(axis=1, how="all")
    df = df.iloc[:, :n_periods]
    label_fn = _annual_label if period_type == "annual" else _quarter_label
    df.columns = [label_fn(c) for c in df.columns]
    if include_ltm and period_type == "annual":
        ltm = compute_ltm(facts, concept_map, fact_type)
        if not ltm.empty:
            df.insert(0, "LTM", ltm)
    return df


EPS_ROWS = {"EPS Basic", "EPS Diluted"}
SHARE_ROWS = {"Shares Basic", "Shares Diluted"}


def fmt_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for col in df.columns:
        for idx in df.index:
            val = df.loc[idx, col]
            if pd.isna(val):
                out.loc[idx, col] = "—"
                continue
            try:
                v = float(val)
                if idx in EPS_ROWS:
                    out.loc[idx, col] = f"${v:.2f}"
                elif idx in SHARE_ROWS:
                    out.loc[idx, col] = f"{v/1e6:.0f}M shs"
                elif abs(v) >= 1e9:
                    out.loc[idx, col] = f"${v/1e9:.2f}B"
                elif abs(v) >= 1e6:
                    out.loc[idx, col] = f"${v/1e6:.1f}M"
                elif abs(v) >= 1e3:
                    out.loc[idx, col] = f"${v/1e3:.0f}K"
                else:
                    out.loc[idx, col] = f"${v:.2f}"
            except Exception:
                out.loc[idx, col] = str(val) if pd.notna(val) else "—"
    return out


def get_filings_df(cik: str, form_type: str, count: int = 8) -> pd.DataFrame:
    try:
        data = fetch_filings_list(cik)
        recent = data.get("filings", {}).get("recent", {})
        rows = []
        for form, date, acc in zip(recent.get("form", []),
                                    recent.get("filingDate", []),
                                    recent.get("accessionNumber", [])):
            if form == form_type:
                acc_fmt = acc.replace("-", "")
                link = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_fmt}/"
                rows.append({"Form": form, "Filed": date, "Accession #": acc, "Link": link})
                if len(rows) >= count:
                    break
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def to_excel(sheets: dict[str, pd.DataFrame], ticker: str) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        hdr = wb.add_format({"bold": True, "bg_color": "#1F4E79",
                              "font_color": "#FFFFFF", "border": 1, "align": "center"})
        num = wb.add_format({"num_format": '#,##0', "border": 1})
        neg = wb.add_format({"num_format": '#,##0', "border": 1, "font_color": "#C00000"})
        idx_fmt = wb.add_format({"border": 1, "bg_color": "#D6E4F0"})
        dec = wb.add_format({"num_format": '0.00', "border": 1})
        for name, df in sheets.items():
            if df is None or df.empty:
                continue
            ws = wb.add_worksheet(name[:31])
            writer.sheets[name[:31]] = ws
            for c, col in enumerate(["Metric"] + list(df.columns)):
                ws.write(0, c, str(col), hdr)
            for r, (metric, row) in enumerate(df.iterrows(), 1):
                ws.write(r, 0, str(metric), idx_fmt)
                for c, val in enumerate(row, 1):
                    if pd.isna(val):
                        ws.write(r, c, "—", idx_fmt)
                    else:
                        try:
                            fv = float(val)
                            fmt = neg if fv < 0 else (dec if abs(fv) < 100 else num)
                            ws.write_number(r, c, fv, fmt)
                        except Exception:
                            ws.write(r, c, str(val), idx_fmt)
            ws.set_column(0, 0, 40)
            ws.set_column(1, len(df.columns), 18)
            ws.freeze_panes(1, 1)
    return buf.getvalue()


c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    ticker_input = st.text_input("Ticker Symbol",
                                  placeholder="e.g. AAPL, MSFT, TSLA, JPM").strip().upper()
with c2:
    n_annual = st.selectbox("Annual Periods", [5, 10, 15, 20], index=0)
with c3:
    n_quarterly = st.selectbox("Quarterly Periods", [8, 12, 16, 20], index=0)

if st.button("Fetch Financial Statements", type="primary", use_container_width=True):
    if not ticker_input:
        st.error("Please enter a ticker symbol.")
        st.stop()

    with st.spinner(f"Resolving {ticker_input} on SEC EDGAR..."):
        try:
            cik, company_name = resolve_cik(ticker_input)
        except ValueError as e:
            st.error(str(e)); st.stop()
        except Exception as e:
            st.error(f"SEC EDGAR error: {e}"); st.stop()

    st.subheader(f"{company_name}  ({ticker_input})")
    st.caption(f"CIK: {int(cik):,}  ·  Data source: SEC EDGAR XBRL")

    with st.spinner("Fetching XBRL data from SEC EDGAR..."):
        try:
            facts = fetch_company_facts(cik)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}"); st.stop()

    with st.spinner("Building statements..."):
        sheets = {
            "IS — Annual":    build_statement(facts, INCOME_STMT_CONCEPTS,   "annual",    n_annual,    "duration", include_ltm=True),
            "IS — Quarterly": build_statement(facts, INCOME_STMT_CONCEPTS,   "quarterly", n_quarterly, "duration"),
            "BS — Annual":    build_statement(facts, BALANCE_SHEET_CONCEPTS, "annual",    n_annual,    "instant",  include_ltm=True),
            "BS — Quarterly": build_statement(facts, BALANCE_SHEET_CONCEPTS, "quarterly", n_quarterly, "instant"),
            "CF — Annual":    build_statement(facts, CASHFLOW_CONCEPTS,      "annual",    n_annual,    "duration", include_ltm=True),
            "CF — Quarterly": build_statement(facts, CASHFLOW_CONCEPTS,      "quarterly", n_quarterly, "duration"),
        }

    tabs = st.tabs(list(sheets.keys()) + ["SEC Filings"])

    for tab, (name, df) in zip(tabs[:-1], sheets.items()):
        with tab:
            if df.empty:
                st.warning(f"No data available for {name}.")
            else:
                st.dataframe(fmt_display(df), use_container_width=True, height=560)

    with tabs[-1]:
        st.write("### SEC Filings")
        base = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type="
        ca, cb = st.columns(2)
        with ca:
            st.write("**10-K (Annual Reports)**")
            d = get_filings_df(cik, "10-K", n_annual)
            if d.empty:
                st.info("None found.")
            else:
                for _, row in d.iterrows():
                    st.markdown(f"- [{row['Filed']}]({row['Link']})  `{row['Accession #']}`")
            st.markdown(f"[All 10-K filings on EDGAR]({base}10-K&owner=include&count=40)")
        with cb:
            st.write("**10-Q (Quarterly Reports)**")
            d = get_filings_df(cik, "10-Q", n_quarterly)
            if d.empty:
                st.info("None found.")
            else:
                for _, row in d.iterrows():
                    st.markdown(f"- [{row['Filed']}]({row['Link']})  `{row['Accession #']}`")
            st.markdown(f"[All 10-Q filings on EDGAR]({base}10-Q&owner=include&count=40)")

    st.divider()
    excel_bytes = to_excel(sheets, ticker_input)
    st.download_button(
        label=f"Download {ticker_input} Financials (.xlsx)",
        data=excel_bytes,
        file_name=f"{ticker_input}_financials.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )
