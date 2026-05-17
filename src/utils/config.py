'''utils/config.py'''
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env desde la raíz del proyecto
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# DATA_ROOT viene del .env, si no existe usa "data" como fallback
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data")).resolve()


# Subcarpetas fijas del proyecto
RAW_DIR = DATA_ROOT / "dataraw"
PROCESSED_DIR = DATA_ROOT / "processed"
GRAPHS_DIR = DATA_ROOT / "graphs"

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

# Archivo de failures (siempre dentro de RAW_DIR)
FAILURES_FILE = RAW_DIR / "failures_07_25.csv"

# Parámetros globales del pipeline
ENTITY_COL = "CERT"
PERIOD_COL = "period"

# ── Panel temporal (TabPFN + T-GCN) ──────────────────────────────────────
# Variables seleccionadas tras mapeo CAMELS + filtro correlación >0.95

# FTS — 236 variables
FTS_CAMELS = {
    "C": [
        "APCDLNLS",  # RBC ACL PCD loans
        "APCDOFA",   # RBC ACL PCD other financial assets
        "APCDSCHM",  # RBC ACL PCD securities HTM
        "AVASSETW",  # Total assets leverage capital
        "AVTANEQ",   # Avg tangible equity assessment
        "CCED1L",    # Ctrl clr equity der <1yr
        "CCED1T5",   # Ctrl clr equity der 1-5yr
        "CCEDOV5",   # Ctrl clr equity der >5yr
        "CT1BADJ",   # Common equity T1 before adj
        "CT1MIN",    # Common equity T1 min interest
        "ED1LES",    # Equity derivative <1yr
        "ED1T5",     # Equity derivative 1-5yr
        "EDOVR5",    # Equity derivative >5yr
        "EQCSTKRX",  # Sale of capital stock
        "EQOTHCC",   # Other equity capital components
        "INTAN",     # Intangible assets
        "INTANTO",   # Intangible assets ex goodwill
        "LIABEQ",    # Total liabilities & capital
        "RB2LNRES",  # Allowance credit losses in tier 2
        "RBASTDL",   # Leverage capital asset deduct
        "RBCT1ADD",  # Additional tier 1 capital
        "RBOTHDL",   # Other leverage capital add/deduct
        "RWATOTW",   # Total RWA reported
        "SCDBEQ",    # Debt & equity securities
        "SCEQ",      # Equity securities
        "SCFDEQ",    # Foreign debt & equity
        "T1ADCPIN",  # Add tier 1 cap instruments & surplus
        "T2AFSGN",   # AFS unrealized gain tier 2
        "T2CAPINS",  # Tier 2 capital instruments & surplus
        "T2DED",     # Tier 2 capital deductions
        "T2MIN",     # Minority interest not incl tier 1
        "T2NQCAP",   # Tier 2 non qualifying capital
        "UCLOC",     # Unused commit home equity lines
    ],
    "A": [
        "DRAGSM",    # Ag loan charge-offs small banks
        "DRAUTO",    # Auto loans charge-offs
        "DRCI",      # Commercial loan charge-offs
        "DRCOMRE",   # Commercial RE loan charge-offs
        "DRCON",     # Consumer loan charge-offs
        "DRCONOTH",  # Other consumer loan charge-offs
        "DRCRCD",    # Credit card loan charge-offs
        "DRLNLS",    # Total loans & leases charge-offs
        "DRLS",      # Lease charge-offs
        "DROTHER",   # All other loan charge-offs
        "DRRE",      # Real estate loan charge-offs
        "DRREAG",    # Farmland RE loan charge-offs
        "DRRECNFM",  # 1-4 fam construct loan charge-offs
        "DRRECNOT",  # Other construct loan charge-offs
        "DRRECONS",  # Construction RE loan charge-offs
        "DRRELOC",   # Line of credit RE loan charge-offs
        "DRREMULT",  # Multifamily RE loan charge-offs
        "DRRENRES",  # Nonfarm nonres RE loan charge-offs
        "DRREOT",    # RE loans other charge-offs
        "DRREOTH",   # Residential RE loan charge-offs
        "DRRERES",   # RE loans 1-4 family charge-offs
        "ELNATR",    # Provisions for credit losses
        "LNLSRES",   # Allowance for credit losses on loans
        "NCGTYGNM",  # Noncurrent rebooked GNMA loans
        "NTAGSM",    # Ag loan net charge-offs small banks
        "NTCI",      # Commercial loan net charge-offs
        "NTCOMRE",   # Commercial RE loan net charge-offs
        "NTCON",     # Consumer loan net charge-offs
        "NTCONOTH",  # Other consumer loan net charge-offs
        "NTLNLS",    # Total loans & leases net charge-offs
        "NTLS",      # Lease net charge-offs
        "NTOTHER",   # All other loan net charge-offs
        "NTRE",      # Real estate loan net charge-offs
        "NTREAG",    # Farmland RE loan net charge-offs
        "NTREOT",    # RE loans other net charge-offs
    ],
    "M": [
        "ECD100",    # Interest expense time CD >$250k
        "EDEP",      # Deposit interest expense
        "EFREPP",    # Fed funds & repos interest expense
        "EOTHINT",   # Other interest expense
        "EOTHNINT",  # All other noninterest expense
        "EOTHTIME",  # Interest expense time CD <=$250k
        "EPREMAGG",  # Premises & fixed assets expense
        "ESAL",      # Salaries and employee benefits
        "ETRANDEP",  # Transaction accounts interest expense
        "NETNIX",    # Net noninterest expense
        "NONIX",     # Total noninterest expense
        "NUMEMP",    # Number of full time employees
        "OLINT",     # Expenses accrued & unpaid
        "OLINTOTH",  # Other expenses accrued & unpaid
    ],
    "E": [
        "EQCCOMPI",  # Other comprehensive income
        "EQCOMINC",  # Accum other comprehensive income
        "EQUP",      # Undivided profits net
        "EQUPGR",    # Undivided profits gross
        "IBEFTAX",   # Income before income taxes & disc
        "IFIDUC",    # Fiduciary activities income
        "IFREPO",    # Fed funds & repo interest income
        "ILN",       # Loan income
        "ILNAG",     # Agricultural loan income
        "ILNCI",     # Commercial loan income
        "ILNCON",    # Consumer loan income
        "ILNCRCD",   # Credit card loan income
        "ILNOTH",    # All other loan income
        "ILNRE",     # Real estate loan income
        "ILNREOTH",  # Other real estate loan income
        "ILNRERES",  # 1-4 family RE loan income
        "ILS",       # Lease income
        "INTINC",    # Total interest income
        "IOTHII",    # Other interest income
        "IOTNII",    # Other noninterest income
        "ISC",       # Total security income
        "ISCMTGBK",  # Mortgage-backed securities income
        "ISCMUNIX",  # Tax-exempt municipal sec income
        "ISCOTTIE",  # Other temp impaired loss sec earnings
        "ITAX",      # Applicable income taxes
        "NONII",     # Total noninterest income
        "OADEFTAX",  # Net deferred income taxes
        "OAIENC",    # Income earned not collected
        "OLDEFTAX",  # Net deferred income taxes
        "RBCERI",    # Eligible retained income
        "REM100",    # Consumer remittances abroad >100yr
        "REMACH",    # Consumer remit abroad ACH
        "REMAMT",    # Consumer remittances abroad amount
        "REMEXCPT",  # Consumer remit abroad num exceptions
        "REMIWT",    # Consumer remit abroad wire
        "REMMECH",   # Consumer remit abroad mechanism
        "REMNUM",    # Consumer remittances abroad number
        "REMPS",     # Consumer remit abroad prop institution
        "REMPSO",    # Consumer remit abroad prop other
        "RPTRI",     # Report of income flag
        "UNINC",     # Unearned income
    ],
    "L": [
        "AVOTHBOR",  # Avg other borrowed money
        "AVSAVDP",   # Avg savings deposits
        "BRORECIP",  # Reciprocal brokered deposits
        "CBLRSCLB",  # Securities lent and borrowed CBLR
        "CHBAL",     # Cash & due from depository inst
        "CHBAL0",    # Cash & bal 0% risk weight
        "CHBAL100",  # Cash & bal 100% risk weight
        "CHBAL150",  # Cash & bal 150% risk weight
        "CHBAL20",   # Cash & bal 20% risk weight
        "CHBAL50",   # Cash & bal 50% risk weight
        "CHBALI",    # Interest-bearing cash & due
        "CHBALNI",   # Noninterest-bearing cash & due
        "CHBALNRW",  # Cash and bal due adj to col A
        "CHBALRCR",  # Cash & bal BS RC-R col A
        "CT1CFHGL",  # Accum gain/loss cash flow hedge
        "DEP",       # Total deposits
        "DEPLGRA",   # Large deposit retirement acc amount
        "DEPRECP",   # Deposits received
        "DEPSMB",    # Small deposit accounts number
        "DEPSMRA",   # Small deposit retirement acc amount
        "DEPSMRN",   # Small deposit retirement acc number
        "DEPUNA",    # Estimated uninsured deposits
        "EQCDIV",    # Cash dividends on comm & pref
        "EQCDIVP",   # Cash dividends on pref stock
        "ICHBAL",    # Depository institutions interest inc
        "ILNREFNG",  # Noncash inc 1-4 fam neg amort
        "IRAKEOGH",  # IRAs and Keogh plans deposits
        "ISERCHG",   # Service charge on deposit accounts
        "NTRTIME",   # Time deposits total
        "NTRTMLG",   # Time deposits over $100M
        "NTRTMMED",  # Time deposits $100-$250M
        "OTBOT1L",   # Other borrowings mat/repr <1yr
        "OTBOT1T3",  # Other borrowings mat/repr 1-3yr
        "OTBOT3T5",  # Other borrowings mat/repr 3-5yr
        "OTBOTJ",    # Other borrowings excl FHLB
        "OTBOTOV5",  # Other borrowings mat/repr >5yr
        "OTHB1LES",  # Other borrowed money <1yr
        "OTHBOR",    # Other borrowed money
        "OTHBOVR1",  # Other borrowed money >1yr
        "SCBORROW",  # Securities borrowed
        "SECLBOTH",  # Secured liabilities other borrowings
        "TS",        # Time & savings deposits total
    ],
    "S": [
        "AVSCMTGB",  # Avg mortgage backed securities
        "CCRT1L",    # Ctrl clear int rate <1yr
        "CCRT1T5",   # Ctrl clear int rate 1-5yr
        "CCRTOV5",   # Ctrl clear int rate >5yr
        "DIRRCR",    # Derivative contracts CE amount
        "DRSCHTMX",  # Securities HA net allowance
        "FGN",       # Historical FRBH foreign flag
        "IGLSEC",    # Securities gains and losses
        "ISCOTH",    # All other securities income
        "NASCDEBT",  # Nonaccrual debt securities
        "NCSCDEBT",  # Total noncurrent debt securities
        "OALIFSEP",  # Life insurance assets separate acc
        "OCRT1L",    # OTC interest rate contracts <1yr
        "OCRT1T5",   # OTC interest rate contracts 1-5yr
        "OCRTOV5",   # OTC interest rate contracts >5yr
        "ONVMK",     # Other derivatives marked to market
        "OTCD0",     # OTC derivatives 0% risk weight
        "OTCD10",    # OTC derivatives 10% risk weight
        "OTCD100",   # OTC derivatives 100% risk weight
        "OTCD150",   # OTC derivatives 150% risk weight
        "OTCD2",     # OTC derivatives 2% risk weight
        "OTCD20",    # OTC derivatives 20% risk weight
        "OTCD4",     # OTC derivatives 4% risk weight
        "OTCD50",    # OTC derivatives 50% risk weight
        "OTCDOCE",   # OTC derivatives other approach CE
        "P3SCDEBT",  # 30-89 days past due debt securities
        "P9SCDEBT",  # 90+ days past due debt securities
        "RBCCX",     # Credit exposure derivatives
        "RTNVMK",    # Interest rate marked to market
        "RTNVTR",    # Interest rate total trade
        "RTOVR1",    # Interest rate contracts >1yr
        "RWAMKTRK",  # RWA standardized market risk assets
        "SC",        # Securities total
        "SCAA",      # Securities available for sale
        "SCAF0",     # Securities AF 0% risk weight
        "SCAF100",   # Securities AF 100% risk weight
        "SCAF150",   # Securities AF 150% risk weight
        "SCAF2",     # Securities AF 2% risk weight
        "SCAF20",    # Securities AF 20% risk weight
        "SCAF300",   # Securities AF 300% risk weight
        "SCAF4",     # Securities AF 4% risk weight
        "SCAF50",    # Securities AF 50% risk weight
        "SCAF600",   # Securities AF 600% risk weight
        "SCAFNRW",   # Securities AF adj col A
        "SCAFRCR",   # Securities AF BS RC-R col A
        "SCDEBT",    # Debt securities
        "SCFORDAA",  # Foreign debt securities AA
        "SCFORDHA",  # Foreign debt securities HA
        "SCHA",      # Securities held to maturity
        "SCHA0",     # Securities HA 0% risk weight
        "SCHA100",   # Securities HA 100% risk weight
        "SCHA150",   # Securities HA 150% risk weight
        "SCHA2",     # Securities HA 2% risk weight
        "SCHA20",    # Securities HA 20% risk weight
        "SCHA4",     # Securities HA 4% risk weight
        "SCHA50",    # Securities HA 50% risk weight
        "SCHANRW",   # Securities HA adj col A
        "SCHARCR",   # Securities HA BS RC-R col A
        "SCLENT",    # Securities lent
        "SCMUNAFD",  # Municipal securities AF dom
        "SCMUNHAD",  # Municipal securities HA dom
        "SCMUNI",    # Municipal securities
        "SCPLEDGE",  # Pledged securities
        "SCTOTALL",  # Total HTM securities allowance
        "SCUST",     # US treasury securities
        "SCUSTAA",   # US treasury securities AA
        "SCUSTHA",   # US treasury securities HA
        "TRADE",     # Trading accounts
        "TRADEL",    # Trading liabilities
        "TRNFG",     # Transaction foreign government
        "VALACRCD",  # Separate val all uncoll CC fees
    ],
}

# CDI — 100 variables
CDI_CAMELS = {
    "C": [
        "ASSETQT",   # Average assets tangible PCA
        "EQ2",     # Total bank equity capital CAVG2
        "EQ5",        # Total bank equity capital CAVG5
        "EQCDIVQ",   # Cash dividends on capital stock QTR
        "EQCSSTXQ",  # Capital stock transactions QTR
        "SCEQ2",     # Equity securities CAVG2
    ],
    "A": [
        "DRLNLSQ",   # Loans & leases charge-offs QTR
        "DRLSQ",     # Lease charge-offs QTR
        "DRREAOT",   # Other RE loans charge-offs
        "ELNATRA",   # Provision for credit losses annual
        "NCAGTEST",  # Noncurrent ag loan test
        "NCORE",     # Nonperforming RE loans & ORE
        "NPERF",     # Nonperforming assets
        "NPERFPP",   # Nonperforming assets prior period
        "NTAGSMA",   # Ag loan net charge-offs ann small banks
        "NTAUTO",    # Auto loans net charge-offs
        "NTCOMRQA",  # Commercial RE net charge-off QTR ann
        "NTLNLSQ",   # Loans & leases net charge-offs QTR
        "NTLSQ",     # Lease net charge-offs QTR
        "OFFBSRES",  # Allowance for off-BS credit losses
    ],
    "M": [
        "EDEPA",     # Deposit interest expense annual
        "EEFFQ",     # Efficiency ratio expense QTR
        "EFREPPA",   # Fed funds & repo int expense annual
        "EFREPPQ",   # Fed funds & repo int expense QTR
        "EOTHINQA",  # Other interest expense QTR ann
        "EOTHINTA",  # Other interest expense annual
        "EOTHNINA",  # All other noninterest expense annual
        "EOTHNINQ",  # All other noninterest expense QTR
        "ESALA",     # Salaries and employee benefits annual
        "ESALQ",     # Salaries and employee benefits QTR
        "IEFF",      # Efficiency ratio income
        "IEFFQ",     # Efficiency ratio income QTR
        "NETNIXA",   # Net noninterest expense annual
        "NETNIXQ",   # Net noninterest expense QTR
        "NONIXA",    # Total noninterest expense annual
    ],
    "E": [
        "EQUP2",     # Undivided profits CAVG2
        "EQUP5",     # Undivided profits CAVG5
        "IBEFXTQA",  # Income before disc operations QTR ann
        "IBEFXTRA",  # Income before disc operations annual
        "IFIDUCA",   # Fiduciary activities income annual
        "ILNA",      # Loan income annual
        "ILNAGA",    # Agricultural loan income annual
        "ILNCIA",    # Commercial loan income annual
        "ILNCONA",   # Consumer loan income annual
        "ILNCRCDA",  # Credit card loan income annual
        "ILNDOMQ",   # Loan income domestic QTR
        "ILNOTHA",   # All other loan income annual
        "ILNREA",    # Real estate loan income annual
        "ILNREOTA",  # Other RE loan income annual
        "ILNRERSA",  # 1-4 family RE loan income annual
        "ILSA",      # Lease income annual
        "INTINCA",   # Total interest income annual
        "INTINQ",    # Total interest income QTR
        "IOTHIIA",   # Other interest income annual
        "IOTNIIA",   # Other noninterest income annual
        "IOTNIIQ",   # Other noninterest income QTR
        "ISCA",      # Total security income annual
        "ITAXA",     # Applicable income taxes annual
        "ITAXQ",     # Applicable income taxes QTR
        "NETINCPP",  # Net income bank prior period
        "NETINQPP",  # Net income bank QTR prior period
        "NIMPP",     # Net interest income prior period
        "NIMQPP",    # Net interest income QTR prior period
        "NONIIA",    # Total noninterest income annual
        "NONIIQ",    # Total noninterest income QTR
        "NTIRT",     # Retained earnings bank
        "NTIRTA",    # Retained earnings bank annual
        "NTIRTQ",    # Retained earnings bank QTR
        "TAXMAR",    # Applicable income taxes rate
        "TAXRAW",    # Applicable income taxes raw
    ],
    "L": [
        "AVOTHBO4",  # Avg other borrowed money CAVG4
        "CHBAL2",    # Cash & due from dep inst CAVG2
        "CHBALI2",   # Interest-bearing cash & due CAVG2
        "CHBALI5",   # Interest-bearing cash & due CAVG5
        "CHBALNI2",  # Nonint-bearing cash & due CAVG2
        "CHFL",      # Net operating cash flow
        "CHFLA",     # Net operating cash flow annual
        "CHFLQ",     # Net operating cash flow QTR
        "DEPFORW",   # Total deposits foreign week
        "DEPPREFD",  # Preferred deposits
        "DEPUNINS",  # Estimated uninsured deposits
        "EQCDIVA",   # Cash dividends on comm & pref annual
        "ICHBALA",   # Depository inst interest income annual
        "ICHBALQ",   # Depository inst interest income QTR
        "LNDEP2",    # Depository inst loans CAVG2
        "LNDEP5",    # Depository inst loans CAVG5
        "OBOR",      # Other borrowed funds
        "OBOR2",     # Other borrowed funds CAVG2
        "OBOR5",     # Other borrowed funds CAVG5
        "OTHBOR2",   # Other borrowed money CAVG2
        "OTHBOR5",   # Other borrowed money CAVG5
    ],
    "S": [
        "IGLSECA",   # Securities gains and losses annual
        "IGLSECJQ",  # Securities gains & losses adj QTR
        "ISCOTHA",   # All other securities income annual
        "LNCONCEN",  # Credit risk concentrated loans
        "LNREO",     # RE loans ag plus RE loans foreign
        "OBSDIR",    # Off-balance sheet derivatives
        "SC1LES",    # Debt securities <1yr
        "SCOVR1",    # Investment securities >1yr
        "TRADE2",    # Trading accounts CAVG2
        "TRADE5",    # Trading accounts CAVG5
    ],
}

# RAT — 73 variables
RAT_CAMELS = {
    "C": [
        "EQTANQTA",  # Tangible equity capital ratio
        "EQTOTY1",   # Total equity capital Y1
        "EQV",       # Bank equity capital / assets
        "LIABEQY1",  # Total liabilities & capital Y1
        "LNAGT1R",   # Ag loans / tier 1
        "LNCDT1R",   # Constr & land dev loans / tier 1
        "LNCIT1R",   # C&I loans / tier 1
        "LNCONT1R",  # Consumer loans / tier 1
        "LNHRSKR",   # High risk loans / tier 1
        "LNRERT1R",  # RE loans / tier 1
        "NCRELOCR"   # Noncurrent home equity
        "P3RELOCR"   # 30-89 past due home equity / home equity
        "RBC1RWAJ",  # Tier 1 RBC ratio PCA
        "RBCPCA",    # RBC category PCA
        "RBCT1Y1",   # Tier 1 RBC PCA Y1
        "ROE",       # Return on equity bank
        "ROEINJR",   # Retained earnings / avg equity
        "ROEQ",      # Return on equity bank QTR
    ],
    "A": [
        "ELNATQY1",  # Provision for credit losses QY1
        "LNATRESR",  # Loan loss reserve / gross loans
        "LNRESNCR",  # Loan loss reserve / noncurrent loans
        "NCLNLSY1",  # Noncurrent loans & leases Y1
        "NCOREQ1",   # Nonperforming RE loans & ORE Q1
        "NCOREY1",   # Nonperforming RE loans & ORE Y1
        "NPERFY1",   # Nonperforming assets Y1
        "NTLNLSR",   # Net charge-offs / loans & leases
        "NTLNLSY1",  # Total loans & leases net charge-offs Y1
        "NTLNLY1S",  # Total loans & leases net charge-offs S Y1
        "NTREQR",    # RE charge-off QTR / RE loans
        "P3RECONR",  # 30-89 past due constr RE / constr RE
        "P3RER",     # 30-89 past due RE / RE loans
    ],
    "M": [
        "EEFFQR",    # Efficiency ratio quarterly
        "EINTEXY1",  # Total interest expense Y1
        "EINTXQY1",  # Total interest expense QY1
        "INTEXPY",   # Interest expense / earning assets
        "NETNIXY1",  # Net noninterest expense Y1
        "NONIXAY",   # Noninterest expense / avg assets
        "NONIXQY1",  # Total noninterest expense QY1
        "NONIXY",    # Noninterest expense / earning assets
        "NONIXY1",   # Total noninterest expense Y1
        "NTNIXQY1",  # Net noninterest expense QY1
        "NUMEMPY1",  # Number of FT employees Y1
    ],
    "E": [
        "IBEFTXY1",  # Income before taxes & disc Y1
        "IBEFXTY1",  # Income before disc operations Y1
        "INTINCY",   # Interest income / earning assets
        "INTINCY1",  # Total interest income Y1
        "INTINQY1",  # Total interest income QY1
        "ITAXQY1",   # Applicable income taxes QY1
        "ITAXY1",    # Applicable income taxes Y1
        "NETINY1S",  # Net income bank S Y1
        "NETIQY1S",  # Net income bank S QY1
        "NIMQY1",    # Net interest income QY1
        "NIMY1",     # Net interest income Y1
        "NOIJQY1",   # Net operating income adj QY1
        "NOIJY",     # Net operating income adj / assets
        "NOIJY1",    # Net operating income adj Y1
        "NONIIQY1",  # Total noninterest income QY1
        "NONIIY1",   # Total noninterest income Y1
        "NTINCQY1",  # Net income bank QY1
        "NTIRTAY1",  # Retained earnings bank Y1
        "NTIRTQY1",  # Retained earnings bank QY1
    ],
    "L": [
        "CHFLQY1",   # Net operating cash flow QY1
        "CHFLY1",    # Net operating cash flow Y1
        "DEPDASTR",  # Total domestic deposit / asset
        "DEPDOMY1",  # Total deposits domestic Y1
        "EQCDIQY1",  # Cash dividends on stock QTR Y1
        "EQCDIVY1",  # Cash dividends on stock Y1
        "LNLSDEPR",  # Net loans & leases / deposits
        "OBORY1",    # Other borrowed funds Y1
    ],
    "S": [
        "LNREOY1",   # RE loans ag & RE loans foreign Y1
        "NTRENRQR",  # Nonfarm nonres RE charge-off QTR rate
        "SCMTGBY1",  # Mortgage backed securities Y1
        "SCY1",      # Securities Y1
    ],
}

# Lista plana de todas las variables temporales seleccionadas
TEMPORAL_FEATURES = (
    [v for comp in FTS_CAMELS.values() for v in comp] +
    [v for comp in CDI_CAMELS.values() for v in comp] +
    [v for comp in RAT_CAMELS.values() for v in comp]
)

# ── Panel nodos (grafo) ───────────────────────────────────────────────────

STRU_SELECTED = [
    "CERT",
    "period",
    # Identidad y tipo
    "BKCLASS",      # Institution class
    "INSTTYPE",     # Institution type  
    "CHRTAGNT",     # Charter agent
    "REGAGNT",      # Primary regulating agency
    # Geografía — para aristas por proximidad
    "STALP",        # State
    "SIMS_LAT",     # Latitude
    "SIMS_LONG",    # Longitude
    "CBSA",         # Core Based Statistical Area
    "METRO",        # Metropolitan flag
    # Estructura de holding — aristas estructurales
    "RSSDHCR",      # ID holding company regulatorio
    # Estado
    "ACTIVE",       # Active flag
    "FAILED",       # Failed flag — label auxiliar
    "DENOVO",       # Banco nuevo
    # Capacidades
    "OFFDOM",       # Domestic offices
    "OFFTOT",       # Total offices
    "OFFSTATE",     # States with offices
]

MERG_SELECTED = [
    "C_CERT",    # Banco absorbido/cerrado
    "EFFDATE",   # Fecha efectiva — clave para deduplicar
    "YEARQTR",   # Año-trimestre — para derivar period
    "CODE2XX",   # Tipo de cierre (211-350)
    "CODE8XX",   # Tipo de fusión (810-830)
    "ASSIST",    # FDIC intervino
    "L_ASSET",   # Activos del banco en el cierre
    "L_DEP",     # Depósitos del banco en el cierre
]

