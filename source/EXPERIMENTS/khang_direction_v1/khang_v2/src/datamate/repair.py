"""So sua xe (tiem sua xe may): moi dong mot luot xe / mot phieu. Rulebook muc "Sua xe".

Bang chuan P: ticket_id, date, xe (o goc), plate (bien so), dong_xe (dong xe da gop cach viet), viec (list viec chuan),
viec_raw, tho (ho ten), tien, bao_hanh, co_thu_tien, ghi_chu.

- Luot xe = phieu = mot dong so, ke ca phieu bao hanh; mot xe toi hai lan la hai luot.
- Phieu bao hanh: ghi chu / viec ghi bao hanh, tien 0; khong tinh la luot lam viec do; khong tinh vao "phieu co thu tien".
- Dong xe: cac cach ghi cung xuat hien tren mot bien so la mot dong (mot bien so = mot chiec xe), them gop hoa/thuong,
  khoang trang, 'α' = 'alpha'.
- Viec: ghi tat / ten thuong gap gop ve mot viec chuan ('vá', 'vá xe' -> 'vá ruột'; 'bd', 'bdtq' -> 'bảo dưỡng';
  'nhớt' tron -> 'thay nhớt máy', 'láp' -> 'thay nhớt láp'...). Tu khong co trong bang giu nguyen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .answers import round2, round_money
from .loader import META_COLS, Business, _split_name
from .parsing import compact_key, strip_accents

_PLATE = re.compile(r"\b(\d{2}\s?[A-Z]{1,2}\d?\s?[-.]?\s?\d{3}\.?\d{2})\b", re.I)
_PLATE_NC = re.compile(r"\b\d{2}\s?[A-Z]{1,2}\d?\s?[-.]?\s?\d{3}\.?\d{2}\b", re.I)

# viec chuan <- cac cach ghi (chu thuong, co dau). Theo rulebook: 'nhớt' tron la nhot may; 'ruột' tron la thay ruot moi;
# sua dien khong gom thay bong den, thay ac quy.
_JOBS = {
    "vá ruột": ("vá", "vá xe", "vá ruột", "vá ruột xe", "vá vỏ"),
    "thay ruột": ("ruột", "ruột xe", "thay ruột", "thay ruột xe"),
    "bảo dưỡng": ("bảo dưỡng", "bd", "bdtq", "bảo dưỡng xe", "bảo dưỡng tổng quát", "bảo trì"),
    "rửa xe": ("rửa", "rửa xe"),
    "thay nhớt máy": ("nhớt", "thay nhớt", "nhớt máy", "thay nhớt máy", "nhớt castrol", "nhớt castrol power1", "nhớt hãng", "nhớt honda",
                      "thay nhớt castrol", "motul", "nhớt motul", "thay nhớt motul", "castrol", "nhớt shell", "nhớt motul ga", "shell 1l",
                      "thay nhớt shell", "shell"),
    "thay nhớt láp": ("láp", "nhớt láp", "thay nhớt láp", "nhớt hộp số", "dầu láp", "thay dầu láp", "nhớt lap"),
    "vệ sinh nồi": ("làm nồi", "vs nồi", "vệ sinh nồi"),
    "sửa điện": ("sửa điện", "làm điện", "điện"),
    "bố thắng": ("bố thắng", "má phanh", "thay bố thắng", "thay má phanh"),  # chi ghi trong phieu bao hanh, khong ro truoc/sau
    "thay bố thắng trước": ("bố trước", "bố thắng trước", "má phanh trước", "thay bố trước"),
    "thay bố thắng sau": ("bố sau", "bố thắng sau", "má phanh sau", "thay bố sau"),
    "thay bugi": ("bu gi", "bugi", "thay bugi", "thay bu gi"),
    "thay lọc gió": ("lọc gió", "thay lọc gió"),
    "chỉnh sên": ("chỉnh sên", "cân sên", "tăng sên", "chỉnh xích"),
    "vệ sinh kim phun": ("vs kim phun", "vệ sinh kim phun", "rửa kim phun"),
    "thay bóng đèn": ("thay bóng đèn", "bóng đèn", "đèn pha", "thay đèn"),
    "thay vỏ": ("vỏ", "vỏ xe", "thay vỏ", "vỏ casumina", "thay vỏ xe"),
    "thay dây curoa": ("dây curoa", "curoa", "dây cu roa", "thay curoa", "thay dây curoa"),
    "thay nhông sên dĩa": ("bộ nhông xích", "nsd", "nhông sên dĩa", "thay nhông sên dĩa", "nhông xích"),
    "thay nước làm mát": ("nước làm mát", "thay nước mát", "nước mát", "thay nước làm mát"),
    "thay ắc quy": ("ắc quy", "bình", "bình gs", "thay bình", "thay ắc quy"),
}
_JOB_OF = {compact_key(v): k for k, vs in _JOBS.items() for v in vs + (k,)}
_JOB_OF.update({compact_key(strip_accents(v)): k for k, vs in _JOBS.items() for v in vs + (k,)})
_WARRANTY = re.compile(r"^(?:bảo hành|bao hanh|bh)\s+(.+)$", re.I)


_MODEL_ABBR = {"ab": "airblade", "ex": "exciter", "wavea": "wavealpha", "jup": "jupiter"}


def canon_job(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text).strip().lower())
    return _JOB_OF.get(compact_key(t)) or _JOB_OF.get(compact_key(strip_accents(t))) or t


def _split_jobs(cell) -> list[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    return [j.strip() for j in re.split(r"\s*(?:,|\+|;|/| và )\s*", str(cell)) if j.strip()]


@dataclass
class RepairModel:
    P: pd.DataFrame
    commission: dict = field(default_factory=dict)  # tho (ho ten) -> % hoa hong
    mapping: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _norm(c: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(c)).split())


def _display_name(variants: pd.Series) -> str:
    """Ten hien thi cua mot dong xe: cach ghi (bo hoa/thuong, khoang trang, α) pho bien nhat trong nhom
    (vision 589 luot > vision110 335 -> Vision; airblade > AB -> Air Blade), trong do uu tien dang co chu hoa
    khong viet hoa toan bo, chu ASCII, roi pho bien nhat. Loc dong_xe= theo cach ghi nao cung duoc."""
    vc = variants.astype(str).str.strip().value_counts()
    key = {v: compact_key(strip_accents(v.replace("α", "alpha"))) for v in vc.index}
    kc = pd.Series({v: vc[[u for u in vc.index if key[u] == key[v]]].sum() for v in vc.index})

    def score(v: str):
        return (int(kc[v]), any(ch.isupper() for ch in v) and not v.isupper(), v.isascii(), int(vc[v]))

    return max(vc.index, key=score)


def build_repair(biz: Business) -> RepairModel | None:
    txt = strip_accents(biz.readme.text.lower())
    if not re.search(r"sua xe|luot xe|phieu sua", txt):  # chi so sua xe (tranh so giao hang co bien so xe tai)
        return None
    for t, df in biz.tables.items():
        cols = [c for c in df.columns if c not in META_COLS]
        xe = next((c for c in cols if df[c].dtype == object and df[c].dropna().astype(str).str.contains(_PLATE_NC).mean() > 0.8), None)
        if xe is None:
            continue
        date = next((c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])), None)
        money = next((c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and any(k in _norm(c).split() for k in ("tien", "thanh", "cong"))), None)
        tho = next((c for c in cols if "tho" in _norm(c).split() and not c.endswith("__ho_ten")), None)
        job = next((c for c in cols if any(k in _norm(c) for k in ("lam gi", "viec", "noi dung", "hang muc", "sua gi"))), None)
        note = next((c for c in cols if "ghi chu" in _norm(c)), None)
        if date is None or money is None or job is None:
            continue
        P = pd.DataFrame({"ticket_id": [f"{s}#{i + 1}" for i, s in enumerate(df.get("_sheet", pd.Series([t] * len(df))).astype(str))],
                          "date": pd.to_datetime(df[date], errors="coerce").dt.normalize().values})
        P["xe"] = df[xe].astype(str).values
        P["plate"] = df[xe].astype(str).str.extract(_PLATE)[0].str.upper().str.replace(r"\s", "", regex=True).values
        raw_model = df[xe].astype(str).map(lambda x: x[: _PLATE.search(x).start()].strip() if _PLATE.search(x) else x.strip())
        P["dong_xe_raw"] = raw_model.values
        P["viec_raw"] = df[job].values
        jobs = df[job].map(_split_jobs)
        P["tien"] = pd.to_numeric(df[money], errors="coerce").fillna(0.0).values
        P["ghi_chu"] = df[note].values if note else None
        warr_job = jobs.map(lambda js: any(_WARRANTY.match(j) for j in js))
        warr_note = df[note].astype(str).str.contains(r"\bbảo hành\b|\bbao hanh\b|\bbh\b", case=False, regex=True) if note else False
        # phieu bao hanh: o lam gi ghi "bao hanh <viec>", hoac ghi chu noi bao hanh VA khong thu tien
        # (phieu co thu tien ma ghi chu "con bao hanh 6 thang" la phieu sua binh thuong)
        P["bao_hanh"] = (warr_job | (warr_note & (pd.to_numeric(df[money], errors="coerce").fillna(0.0) <= 0))).values
        P["co_thu_tien"] = (~P["bao_hanh"]) & (P["tien"] > 0)
        # viec cua phieu bao hanh la viec duoc bao hanh (khong tinh la luot lam viec do)
        P["viec"] = jobs.map(lambda js: sorted({canon_job(_WARRANTY.match(j).group(1)) if _WARRANTY.match(j) else canon_job(j) for j in js})).values
        P["tho_goi"] = df[tho].values if tho else None
        P["tho"] = df[f"{tho}__ho_ten"].fillna(df[tho]).values if tho and f"{tho}__ho_ten" in df else P["tho_goi"]
        # dong xe: cach ghi cung mot bien so la mot dong (union-find), them gop hoa/thuong / khoang trang / α
        key = P["dong_xe_raw"].map(lambda s: compact_key(strip_accents(str(s).replace("α", "alpha"))))
        key = key.map(lambda k: _MODEL_ABBR.get(k, k))  # ghi tat quen dung: AB = Air Blade, Ex = Exciter
        parent = {k: k for k in key.unique()}

        def find(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        for _, ks in pd.DataFrame({"p": P["plate"], "k": key}).dropna().groupby("p")["k"]:
            ks = list(dict.fromkeys(ks))
            for k in ks[1:]:
                a, b = find(ks[0]), find(k)
                if a != b:
                    parent[b] = a
        root = key.map(find)
        disp = P.assign(_r=root.values).groupby("_r")["dong_xe_raw"].agg(_display_name)
        P["dong_xe"] = root.map(disp).values
        P = P.dropna(subset=["date"]).reset_index(drop=True)
        commission = {}
        for t2, d2 in biz.tables.items():
            pc = next((c for c in d2.columns if "hoa hong" in _norm(c)), None)
            nm = next((c for c in d2.columns if _norm(c) in ("ho ten", "ten", "ten tho", "ho va ten")), None)
            if pc and nm:
                commission = {str(r[nm]).strip(): float(r[pc]) for _, r in d2.dropna(subset=[pc, nm]).iterrows()}
                break
        notes = [f"{int(P['bao_hanh'].sum())} phiếu bảo hành (tiền 0, không thu tiền)",
                 f"{P['dong_xe'].nunique()} dòng xe sau khi gộp {P['dong_xe_raw'].nunique()} cách ghi theo biển số"]
        mapping = {"date": f"{t}.{date}", "xe": f"{t}.{xe}", "viec": f"{t}.{job}", "tien": f"{t}.{money}", "tho": f"{t}.{tho}", "ghi_chu": f"{t}.{note}"}
        return RepairModel(P=P, commission=commission, mapping=mapping, notes=notes)
    return None


# ---------------------------------------------------------------------------- ham cho agent


class RepairMetrics:
    def __init__(self, model: RepairModel):
        self.m = model
        self.P = model.P

    def _sel(self, a, b=None, tho=None, dong_xe=None, viec=None, bao_hanh=None) -> pd.DataFrame:
        from .metrics import _ab, _filters
        from .dates import to_date

        a, b = _ab(a, b)
        P = self.P
        d = P[(P["date"] >= pd.Timestamp(to_date(a))) & (P["date"] <= pd.Timestamp(to_date(b)))]
        f = {}
        if tho is not None:
            f["tho"] = tho
        if dong_xe is not None:
            f["dong_xe"] = dong_xe
        if f:
            try:
                d = d[_filters(d, f)]
            except Exception:
                if "dong_xe" in f:  # loc theo moi cach ghi cua dong xe
                    k = compact_key(strip_accents(str(dong_xe).replace("α", "alpha")))
                    roots = set(P.loc[P["dong_xe_raw"].map(lambda s: compact_key(strip_accents(str(s).replace("α", "alpha")))) == k, "dong_xe"])
                    if not roots:
                        raise
                    d = d[d["dong_xe"].isin(roots)]
                    if tho is not None:
                        d = d[_filters(d, {"tho": tho})]
                else:
                    raise
        if viec is not None:
            v = canon_job(viec)
            d = d[d["viec"].map(lambda js: v in js) & ~d["bao_hanh"]]  # phieu bao hanh khong tinh la luot lam viec do
        if bao_hanh is not None:
            d = d[d["bao_hanh"] == bool(bao_hanh)]
        return d

    def tickets(self, a, b=None, **f) -> pd.DataFrame:
        """Cac phieu (luot xe) trong ky, loc tho=, dong_xe=, viec=, bao_hanh=True/False."""
        return self._sel(a, b, **f)

    def revenue(self, a, b=None, **f) -> int:
        """Doanh thu = tong Tien cac phieu trong ky (phieu bao hanh tien 0)."""
        return round_money(self._sel(a, b, **f)["tien"].sum())

    def n_tickets(self, a, b=None, **f) -> int:
        """So phieu = so luot xe, KE CA phieu bao hanh (tru khi loc viec=: phieu bao hanh khong tinh la luot lam viec do)."""
        return int(len(self._sel(a, b, **f)))

    def avg_ticket_value(self, a, b=None, **f) -> int | None:
        """Tien trung binh mot phieu = doanh thu / so phieu CO THU TIEN (khong tinh phieu bao hanh)."""
        d = self._sel(a, b, **f)
        n = int(d["co_thu_tien"].sum())
        return round_money(d["tien"].sum() / n) if n else None

    def warranty_rate(self, a, b=None) -> float | None:
        """Ti le bao hanh (%) = so phieu bao hanh co ngay trong ky / so phieu co thu tien trong ky."""
        d = self._sel(a, b)
        n = int(d["co_thu_tien"].sum())
        return round2(100.0 * d["bao_hanh"].sum() / n) if n else None

    def n_vehicles(self, a, b=None, **f) -> int:
        """So chiec xe khac nhau (theo bien so) mang toi trong ky, ke ca xe chi toi bao hanh."""
        return int(self._sel(a, b, **f)["plate"].nunique())

    def regular_vehicles(self, a, b=None, min_paid: int = 2) -> pd.DataFrame:
        """Xe khach quen: bien so co >= 2 phieu CO THU TIEN trong ky (phieu bao hanh khong tinh la mot lan toi)."""
        d = self._sel(a, b)
        g = d[d["co_thu_tien"]].groupby("plate").agg(dong_xe=("dong_xe", "first"), so_phieu_thu_tien=("ticket_id", "count"), tien=("tien", "sum"))
        return g[g["so_phieu_thu_tien"] >= min_paid].reset_index().sort_values(["so_phieu_thu_tien", "plate"], ascending=[False, True]).reset_index(drop=True)

    def by_tho(self, a, b=None) -> pd.DataFrame:
        """Moi tho: so_phieu (luot xe, ke ca bao hanh), so_ngay_lam (ngay co it nhat mot phieu, ke ca bao hanh), doanh_thu, so_phieu_thu_tien."""
        d = self._sel(a, b)
        g = d.groupby("tho").agg(so_phieu=("ticket_id", "count"), so_ngay_lam=("date", "nunique"), doanh_thu=("tien", "sum"),
                                 so_phieu_thu_tien=("co_thu_tien", "sum")).reset_index()
        return g.sort_values(["so_phieu", "tho"], ascending=[False, True]).reset_index(drop=True)

    def commission(self, a, b=None) -> pd.DataFrame:
        """Hoa hong moi tho = % hoa hong (danh sach tho) x tong Tien cac phieu tho lam trong ky; sap giam dan."""
        d = self._sel(a, b)
        g = d.groupby("tho")["tien"].sum().reset_index().rename(columns={"tien": "tien_phieu"})
        g["phan_tram"] = g["tho"].map(lambda n: self.m.commission.get(str(n).strip()))
        g["hoa_hong"] = (g["tien_phieu"] * g["phan_tram"] / 100.0).map(lambda x: round_money(x) if pd.notna(x) else None)
        return g.sort_values(["hoa_hong", "tho"], ascending=[False, True]).reset_index(drop=True)

    def by_dong_xe(self, a, b=None) -> pd.DataFrame:
        """Moi dong xe: doanh_thu, so_phieu (ke ca bao hanh), so_xe (bien so khac nhau); sap theo doanh_thu giam dan."""
        d = self._sel(a, b)
        g = d.groupby("dong_xe").agg(doanh_thu=("tien", "sum"), so_phieu=("ticket_id", "count"), so_xe=("plate", "nunique")).reset_index()
        return g.sort_values(["doanh_thu", "dong_xe"], ascending=[False, True]).reset_index(drop=True)

    def by_day(self, a, b=None) -> pd.DataFrame:
        """Theo ngay: doanh_thu, so_phieu (ke ca bao hanh)."""
        d = self._sel(a, b)
        return d.groupby("date").agg(doanh_thu=("tien", "sum"), so_phieu=("ticket_id", "count")).reset_index().sort_values("date").reset_index(drop=True)


def describe(model: RepairModel) -> str:
    P = model.P
    lines = [
        "## Sổ sửa xe dựng sẵn (rulebook mục Sửa xe đã áp)",
        f"P: {len(P)} phiếu (mỗi dòng sổ là một lượt xe / một phiếu), ngày {P['date'].min():%Y-%m-%d} → {P['date'].max():%Y-%m-%d}; cột: {', '.join(P.columns)}.",
        "- dong_xe: dòng xe đã gộp các cách ghi (cùng biển số là cùng xe nên cùng dòng; AB = Air Blade, Wave α = Wave Alpha…); plate: biển số; "
        "viec: list việc chuẩn của phiếu (ghi tắt đã đổi: 'vá' -> 'vá ruột', 'bd'/'bdtq' -> 'bảo dưỡng', 'nhớt' trơn -> 'thay nhớt máy', 'láp' -> 'thay nhớt láp', "
        "'làm điện' -> 'sửa điện' (không gồm bóng đèn, ắc quy)…); tho: họ tên thợ; bao_hanh: phiếu bảo hành (tiền 0); co_thu_tien.",
        f"- Việc chuẩn: {', '.join(_JOBS)}.",
        "- Hàm (kỳ là biến kỳ, lọc tho=, dong_xe=, viec=, bao_hanh=): revenue(kỳ) doanh thu = tổng Tiền; n_tickets(kỳ) số phiếu = số lượt xe (KỂ CẢ phiếu bảo hành; "
        "lọc viec= thì bỏ phiếu bảo hành); avg_ticket_value(kỳ) tiền TB một phiếu (chia số phiếu CÓ THU TIỀN); warranty_rate(kỳ) tỉ lệ bảo hành %; "
        "n_vehicles(kỳ, dong_xe=) số chiếc xe khác nhau (biển số); regular_vehicles(kỳ) xe khách quen (>= 2 phiếu có thu tiền); "
        "by_tho(kỳ) theo thợ (so_phieu, so_ngay_lam, doanh_thu); commission(kỳ) hoa hồng từng thợ; by_dong_xe(kỳ); by_day(kỳ); tickets(kỳ, ...) bảng phiếu.",
        "- Thợ làm nhiều nhất = nhiều so_phieu nhất (kể cả bảo hành); thợ chăm chỉ nhất = nhiều so_ngay_lam nhất; câu hỏi xe nào -> trả lời biển số.",
    ]
    if model.commission:
        lines.append(f"- Hoa hồng (%): {model.commission}")
    lines += [f"- {n}" for n in model.notes]
    return "\n".join(lines)
