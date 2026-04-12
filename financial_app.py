"""
Financial Statement Puller — SEC EDGAR Only
Pulls IS, BS, Cash Flow (Annual + Quarterly) directly from SEC EDGAR XBRL API.
"""

import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.set_page_config(page_title="SEC Financial Puller", layout="wide")
st.title("SEC Financial Statement Puller")
st.caption("Income Statement · Balance Sheet · Cash Flow — Annual & Quarterly · Download as Excel")

HEADERS = {"User-Agent": "financial-research-app contact@example.com"}

INCOME_STMT_CONCEPTS = [
    ("Revenue",                          ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                                          "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"]),
    ("Cost of Revenue",                  ["CostOfRevenue", "CostOfGoodsSold", "CostOfGoodsSoldAndServicesSold"]),
    ("Gross Profit",                     ["GrossProfit"]),
    ("R&D Expense",                      ["ResearchAndDevelopmentExpense"]),
    ("SG&A Expense",                     ["SellingGeneralAndAdministrativeExpense"]),
    ("Operating Expenses",               ["OperatingExpenses"]),
    ("Operating Income (Loss)",          ["OperatingIncomeLoss"]),
    ("Interest Expense",                 ["InterestExpense", "InterestAndDebtExpense"]),
    ("Other Income (Expense), Net",      ["NonoperatingIncomeExpense"]),
    ("Pre-Tax Income",                   ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                                          "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]),
    ("Income Tax Expense",               ["IncomeTaxExpenseBenefit"]),
    ("Net Income",                       ["NetIncomeLoss", "ProfitLoss"]),
    ("EPS Basic",                        ["EarningsPerShareBasic"]),
    ("EPS Diluted",                      ["EarningsPerShareDiluted"]),
    ("Shares Outstanding (Basic)",       ["CommonStockSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"]),
    ("Shares Outstanding (Diluted)",     ["WeightedAverageNumberOfDilutedSharesOutstanding"]),
]

BALANCE_SHEET_CONCEPTS = [
    ("Cash & Equivalents",               ["CashAndCashEquivalentsAtCarryingValue", "Cash"]),
    ("Short-Term Investments",           ["ShortTermInvestments", "AvailableForSaleSecuritiesCurrent"]),
    ("Accounts Receivable",              ["AccountsReceivableNetCurrent"]),
    ("Inventory",                        ["InventoryNet"]),
    ("Other Current Assets",             ["OtherAssetsCurrent"]),
    ("Total Current Assets",             ["AssetsCurrent"]),
    ("PP&E, Net",                        ["PropertyPlantAndEquipmentNet"]),
    ("Goodwill",                         ["Goodwill"]),
    ("Intangible Assets",                ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"]),
    ("Other Non-Current Assets",         ["OtherAssetsNoncurrent"]),
    ("Total Non-Current Assets",         ["AssetsNoncurrent"]),
    ("Total Assets",                     ["Assets"]),
    ("Accounts Payable",                 ["AccountsPayableCurrent"]),
    ("Accrued Liabilities",              ["AccruedLiabilitiesCurrent"]),
    ("Short-Term Debt",                  ["ShortTermBorrowings", "NotesPayableCurrent"]),
    ("Deferred Revenue (Current)",       ["DeferredRevenueCurrent", "ContractWithCustomerLiabilityCurrent"]),
    ("Total Current Liabilities",        ["LiabilitiesCurrent"]),
    ("Long-Term Debt",                   ["LongTermDebt", "LongTermDebtNoncurrent"]),
    ("Deferred Tax Liabilities",         ["DeferredIncomeTaxLiabilitiesNet", "DeferredTaxLiabilitiesNoncurrent"]),
    ("Other Non-Current Liabilities",    ["OtherLiabilitiesNoncurrent"]),
    ("Total Non-Current Liabilities",    ["LiabilitiesNoncurrent"]),
    ("Total Liabilities",                ["Liabilities"]),
    ("Common Stock",                     ["CommonStockValue"]),
    ("Additional Paid-In Capital",       ["AdditionalPaidInCapital"]),
    ("Retained Earnings",                ["RetainedEarningsAccumulatedDeficit"]),
    ("Treasury Stock",                   ["TreasuryStockValue"]),
    ("Total Stockholders' Equity",       ["StockholdersEquity", "StockholdersEquityAttributableToParent"]),
    ("Total Liab. & Equity",             ["LiabilitiesAndStockholdersEquity"]),
]

CASHFLOW_CONCEPTS = [
    ("Net Income",                       ["NetIncomeLoss", "ProfitLoss"]),
    ("D&A",                              ["DepreciationDepletionAndAmortization", "Depreciation",
                                          "DepreciationAndAmortization"]),
    ("Stock-Based Compensation",         ["ShareBasedCompensation"]),
    ("Changes in Working Capital",       ["IncreaseDecreaseInOperatingCapital"]),
    ("Other Operating Activities",       ["OtherOperatingActivitiesCashFlowStatement"]),
    ("Cash from Operations",             ["NetCashProvidedByUsedInOperatingActivities"]),
    ("CapEx",                            ["PaymentsToAcquirePropertyPlantAndEquipment",
                                          "CapitalExpendituresIncurringObligation"]),
    ("Acquisitions",                     ["PaymentsToAcquireBusinessesNetOfCashAcquired"]),
    ("Purchases of Investments",         ["PaymentsToAcquireInvestments",
                                          "PaymentsToAcquireAvailableForSaleSecurities"]),
    ("Sales of Investments",             ["ProceedsFromSaleAndMaturityOfMarketableSecurities",
                                          "ProceedsFromSaleOfAvailableForSaleSecurities"]),
    ("Cash from Investing",              ["NetCashProvidedByUsedInInvestingActivities"]),
    ("Debt Issued",                      ["ProceedsFromIssuanceOfLongTermDebt",
                                          "ProceedsFromIssuanceOfDebt"]),
    ("Debt Repaid",                      ["RepaymentsOfLongTermDebt", "RepaymentsOfDebt"]),
    ("Stock Issued",                     ["ProceedsFromIssuanceOfCommonStock"]),
    ("Stock Repurchased",                ["PaymentsForRepurchaseOfCommonStock"]),
    ("Dividends Paid",                   ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"]),
    ("Cash from Financing",              ["NetCashProvidedByUsedInFinancingActivities"]),
    ("FX Effect on Cash",                ["EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]),
    ("Net Change in Cash",               ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect"]),
]

@st.cache_data(ttl=3600, show_spinner=False)
def resolve_cik(ticker: str) -> tuple[str, str]:
    resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10), entry["title"]
    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR.")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_filings_list(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()

def extract_series(facts: dict, concepts: list[str], period_type: str) -> pd.Series | None:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        if concept not in gaap:
            continue
        units = gaap[concept].get("units", {})
        unit_data = units.get("USD") or units.get("shares") or units.get("USD/shares") or next(iter(units.values()), [])
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
            rows.append({"end": end, "val": val, "accn": item.get("accn", "")})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = df.sort_values("accn").drop_duplicates(subset=["end"], keep="last")
        df = df.set_index("end")["val"].sort_index(ascending=False)
        return df
    return None

def build_statement(facts: dict, concept_map: list, period_type: str, n_periods: int) -> pd.DataFrame:
    rows = {}
    for label, concepts in concept_map:
        series = extract_series(facts, concepts, period_type)
        if series is not None and not series.empty:
            rows[label] = series
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).T
    df = df.iloc[:, :n_periods]
    return df

def get_filings_df(cik: str, form_type: str, count: int = 8) -> pd.DataFrame:
    try:
        data = fetch_filings_list(cik)
        recent = data.get("filings", {}).get("recent", {})
        forms      = recent.get("form", [])
        dates      = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        rows = []
        for form, date, acc in zip(forms, dates, accessions):
            if form == form_type:
                acc_fmt = acc.replace("-", "")
                direct = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_fmt}/"
                rows.append({"Form": form, "Filed Date": date, "Accession #": acc, "Filing Index": direct})
                if len(rows) >= count:
                    break
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def to_excel(sheets: dict[str, pd.DataFrame], ticker: str) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        hdr_fmt = wb.add_format({"bold": True, "bg_color": "#1F4E79", "font_color": "#FFFFFF", "border": 1, "align": "center"})
        num_fmt = wb.add_format({"num_format": '#,##0', "border": 1})
        neg_fmt = wb.add_format({"num_format": '#,##0', "border": 1, "font_color": "#C00000"})
        idx_fmt = wb.add_format({"bold": False, "border": 1, "bg_color": "#D6E4F0"})
        dec_fmt = wb.add_format({"num_format": '0.00', "border": 1})
        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue
            ws = wb.add_worksheet(sheet_name[:31])
            writer.sheets[sheet_name[:31]] = ws
            cols = ["Metric"] + list(df.columns)
            for c, col in enumerate(cols):
                ws.write(0, c, col, hdr_fmt)
            for r, (metric, row) in enumerate(df.iterrows(), start=1):
                ws.write(r, 0, metric, idx_fmt)
                for c, val in enumerate(row, start=1):
                    if pd.isna(val):
                        ws.write(r, c, "—", idx_fmt)
                    else:
                        try:
                            fval = float(val)
                            fmt = neg_fmt if fval < 0 else (dec_fmt if abs(fval) < 100 else num_fmt)
                            ws.write_number(r, c, fval, fmt)
                        except (TypeError, ValueError):
                            ws.write(r, c, str(val), idx_fmt)
            ws.set_column(0, 0, 42)
            ws.set_column(1, len(df.columns), 20)
            ws.freeze_panes(1, 1)
    return buf.getvalue()

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ticker_input = st.text_input("Ticker Symbol", placeholder="e.g. AAPL, MSFT, TSLA, JPM").strip().upper()
with col2:
    n_annual = st.selectbox("Annual Periods", [5, 10, 15, 20], index=0)
with col3:
    n_quarterly = st.selectbox("Quarterly Periods", [8, 12, 16, 20], index=0)

fetch = st.button("Fetch Financial Statements", type="primary", use_container_width=True)

if fetch and not ticker_input:
    st.error("Please enter a ticker symbol.")

if fetch and ticker_input:
    with st.spinner(f"Resolving CIK for {ticker_input}..."):
        try:
            cik, company_name = resolve_cik(ticker_input)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Error contacting SEC EDGAR: {e}")
            st.stop()

    st.subheader(f"{company_name} ({ticker_input})")
    st.caption(f"SEC CIK: {int(cik):,}")

    with st.spinner("Downloading XBRL financial data from SEC EDGAR..."):
        try:
            facts = fetch_company_facts(cik)
        except Exception as e:
            st.error(f"Failed to fetch company facts: {e}")
            st.stop()

    with st.spinner("Parsing statements..."):
        annual_sheets = {
            "IS (Annual)":  build_statement(facts, INCOME_STMT_CONCEPTS,  "annual",    n_annual),
            "BS (Annual)":  build_statement(facts, BALANCE_SHEET_CONCEPTS, "annual",    n_annual),
            "CF (Annual)":  build_statement(facts, CASHFLOW_CONCEPTS,      "annual",    n_annual),
        }
        quarterly_sheets = {
            "IS (Quarterly)": build_statement(facts, INCOME_STMT_CONCEPTS,  "quarterly", n_quarterly),
            "BS (Quarterly)": build_statement(facts, BALANCE_SHEET_CONCEPTS, "quarterly", n_quarterly),
            "CF (Quarterly)": build_statement(facts, CASHFLOW_CONCEPTS,      "quarterly", n_quarterly),
        }

    all_sheets = {**annual_sheets, **quarterly_sheets}
    tab_labels = list(all_sheets.keys()) + ["SEC Filings"]
    tabs = st.tabs(tab_labels)

    for tab, (name, df) in zip(tabs[:-1], all_sheets.items()):
        with tab:
            if df is None or df.empty:
                st.warning(f"No XBRL data found for {name}.")
            else:
                st.dataframe(df, use_container_width=True, height=600)

    with tabs[-1]:
        st.write("### Recent SEC Filings")
        sec_base = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type="
        c1, c2 = st.columns(2)
        with c1:
            st.write("**10-K Annual Reports**")
            df_10k = get_filings_df(cik, "10-K", count=n_annual)
            if df_10k.empty:
                st.info("No 10-K filings found.")
            else:
                for _, row in df_10k.iterrows():
                    st.markdown(f"- [{row['Filed Date']}]({row['Filing Index']}) — `{row['Accession #']}`")
            st.markdown(f"[Browse all 10-K on EDGAR]({sec_base}10-K&dateb=&owner=include&count=40)")
        with c2:
            st.write("**10-Q Quarterly Reports**")
            df_10q = get_filings_df(cik, "10-Q", count=n_quarterly)
            if df_10q.empty:
                st.info("No 10-Q filings found.")
            else:
                for _, row in df_10q.iterrows():
                    st.markdown(f"- [{row['Filed Date']}]({row['Filing Index']}) — `{row['Accession #']}`")
            st.markdown(f"[Browse all 10-Q on EDGAR]({sec_base}10-Q&dateb=&owner=include&count=40)")

    st.divider()
    with st.spinner("Preparing Excel file..."):
        excel_bytes = to_excel(all_sheets, ticker_input)

    st.download_button(
        label=f"Download {ticker_input} Financial Statements (.xlsx)",
        data=excel_bytes,
        file_name=f"{ticker_input}_SEC_financials.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )
