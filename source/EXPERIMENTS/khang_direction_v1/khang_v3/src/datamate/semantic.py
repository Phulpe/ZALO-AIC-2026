"""Bang chuan ban hang dung san cho moi cua hang (khong dung model).

S: moi dong mot dong hang ban (order_id, date, item, qty, qty_base, amount, status don, revenue_ok...).
O: moi dong mot don (order_id, date, status, revenue, so dong, khach...).

Vai tro cot lay theo mo ta cot trong README (cum tu chuan nhu 'ma don', 'ngay ban', 'thanh tien sau giam'),
neu khong co mo ta thi theo ten cot. Khong nhan ra cau truc -> khong dung S/O (agent dung bang goc T).
Trang thai don (§1, §8): don co dong tra hang / so luong am / co phieu tra -> 'trả hàng' (ca don);
co dong huy -> 'huỷ'; con lai theo trang thai ghi tren don.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .helpers import status_of
from .loader import META_COLS, Business, _split_name
from .parsing import compact_key, norm_key, parse_number

# vai tro -> (tien to mo ta README, ten cot thuong gap da bo dau)
ROLES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "order_id": (("mã đơn", "số hoá đơn", "số hóa đơn", "số thứ tự đơn"), ("ma don", "ma don hang", "ma hoa don", "so hoa don", "don so", "ma bill", "bill", "so ct", "hd", "ma hd", "so don", "ma giao dich", "order id", "so bill", "ma don ban", "so hd")),
    "line_no": (("số thứ tự dòng trong đơn",), ("stt", "dong", "so thu tu", "no", "tt dong", "so dong", "sothutu")),
    "date": (("ngày bán",), ("ngay", "ngay ban", "ngay dat hang", "ngay giao dich", "ngay lap", "ngay hd", "ngay thang", "date", "thoi gian", "ngay ban hang", "ngay hoa don", "ngay chung tu", "ngay lap hd")),
    "datetime": (("thời điểm đặt đơn", "giờ đặt đơn"), ("gio", "gio tao don", "thoi gian tao", "thoi gian lap", "gio ban", "gio dat")),
    "item": (("tên hàng", "hàng bán"), ("ten hang", "ten san pham", "hang hoa", "mat hang", "hang ban", "noi dung", "ten mon", "hang", "san pham", "item", "ten hang hoa", "ten mat hang", "chi tiet hang", "mon", "ten sp", "ten hh", "san pham ban")),
    "item_code": (("mã hàng",), ("ma hang", "ma sku", "ma san pham", "ma hh", "ma hang hoa", "ma vach", "barcode", "ma h hoa", "ma sp", "id sp", "ma mon", "sku", "ma hang ban")),
    "qty": (("số lượng (trống", "số lượng ghi theo"), ("so luong", "sl", "so luong san pham", "s l", "sl ban", "so luong ban", "slg", "sl mua", "so luong mua")),
    "qty_base": (("số lượng đã đổi ra đơn vị cơ bản",), ("sl quy doi", "so luong quy doi", "sl dv co ban", "quy ra dv le", "so luong le", "sl co ban", "sl quy ra le", "so luong co ban", "slquydoi")),
    "unit": (("đơn vị tính",), ("don vi tinh", "dvt", "dv", "unit", "don vi", "d vi", "dvi")),
    "price": (("đơn giá",), ("don gia", "gia", "gia ban", "gia 1 dv", "gia le", "gia don vi", "gia san pham", "don gia d", "gia niem yet", "gia ban le")),
    "amount": (("thành tiền sau giảm", "thành tiền của dòng"), ("thanh tien", "tri gia", "amount", "cong", "thanh tien d")),
    "line_discount": (("tiền giảm của cả dòng",), ("giam d", "chiet khau", "ck", "khuyen mai", "giam gia dong")),
    "cost": (("giá vốn một đơn vị bán", "giá vốn"), ("gia von", "gia von luc ban")),
    "status": (("trạng thái đơn",), ("trang thai", "status", "tinh trang", "ket qua", "xu ly", "tien do", "trang thai don hang", "trang thai giao", "trang thai don")),
    "customer": (("tên khách",), ("khach hang", "ten khach", "khach", "doi tac", "ten khach hang", "khach hang ten", "ten kh", "nguoi mua", "ben mua", "ten khach mua")),
    "customer_id": (("mã khách",), ("ma khach", "ma kh", "ma khach hang", "ma the", "ma nguoi mua", "ma kh mua", "customer id", "ma khach mua")),
    "channel": (("kênh bán",), ("kenh", "nguon", "kenh ban", "kenh dat", "kenh ban hang")),
    "payment": (("hình thức thanh toán",), ("hinh thuc thanh toan", "pttt", "phuong thuc tt", "phuong thuc thanh toan")),
    "employee": (("mã nhân viên bán",), ("ma nv ban", "nguoi ban", "ma nhan vien", "nhan vien", "ma nv")),
    "customer_group": (("nhóm khách",), ("nhom khach", "nhom khach hang", "loai khach")),
    "returned_qty": (("số lượng khách trả lại",), ("so luong tra", "sl tra", "sl tra quy doi")),
}

LINE_ROLES = ("line_no", "item", "item_code", "qty", "qty_base", "unit", "price", "amount", "line_discount", "cost", "returned_qty")
ORDER_ROLES = ("date", "datetime", "status", "customer", "customer_id", "channel", "payment", "employee", "customer_group")

_IMPORT_HINTS = ("nhập", "nhà cung cấp", "phiếu nhập")
_DELIVERY_HINTS = ("ngày giao", "đợt giao", "tài xế", "biển số")


@dataclass
class SalesModel:
    S: pd.DataFrame
    O: pd.DataFrame
    mapping: dict[str, str]  # vai tro -> "bang.cot" nguon
    notes: list[str] = field(default_factory=list)


def _desc(biz: Business, table: str, col: str) -> str:
    ci = biz.info[table].columns.get(col)
    return (ci.description or "").strip().lower() if ci else ""


def _kind(biz: Business, table: str, col: str) -> str:
    ci = biz.info[table].columns.get(col)
    return ci.kind if ci else "text"


def find_role(biz: Business, table: str, role: str) -> str | None:
    df = biz.tables[table]
    descs, names = ROLES[role]
    cols = [c for c in df.columns if c not in META_COLS and not c.endswith("__dv")]
    for c in cols:  # 1) theo mo ta README
        d = _desc(biz, table, c)
        if d and any(d.startswith(p) for p in descs):
            if role == "qty" and ("đổi ra đơn vị cơ bản" in d or "nhập" in d):
                continue
            if role == "cost" and "nhập" in d:
                continue
            return c
    if role == "qty":
        for c in cols:
            if _desc(biz, table, c) == "số lượng":
                return c
    if role == "order_id" and biz.info[table].key:
        k = biz.info[table].key
        if not any(h in _desc(biz, table, k) for h in _IMPORT_HINTS):
            return k
    for c in cols:  # 2) theo ten cot (chi khi cot chua co mo ta)
        if _desc(biz, table, c):
            continue
        n = " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(c)).split())
        if n in names:
            return c
        toks = set(n.split())
        if role == "returned_qty" and "tra" in toks and ({"sl", "slg"} & toks or "so luong" in n):
            return c
    return None


def _kind_ok(biz, table, col, role) -> bool:
    k = _kind(biz, table, col)
    if role in ("date", "datetime"):
        return k in ("date", "datetime")
    if role in ("qty", "qty_base", "price", "amount", "line_discount", "cost", "returned_qty", "line_no"):
        return k == "number"
    return True


def _roles(biz: Business, table: str) -> dict[str, str]:
    out = {}
    for role in ROLES:
        c = find_role(biz, table, role)
        if c is not None and _kind_ok(biz, table, c, role) and c not in out.values():
            out[role] = c
    return out


def _is_import_or_delivery(biz: Business, table: str) -> bool:
    descs = " | ".join(_desc(biz, table, c) for c in biz.tables[table].columns)
    names = " | ".join(_split_name(c) for c in biz.tables[table].columns)
    tn = _split_name(table)
    if any(h in descs for h in _IMPORT_HINTS) or any(h in names for h in ("ngay nhap", "gia nhap", "sl nhap", "phieu nhap")) or "nhap" in tn.split():
        return True
    if any(h in descs for h in _DELIVERY_HINTS) or "giao hang" in tn:
        return True
    return False


def build_sales(biz: Business) -> SalesModel | None:
    roles_by_table = {t: _roles(biz, t) for t in biz.tables}
    # 1) bang dong hang
    best, best_score = None, 0
    for t, r in roles_by_table.items():
        if _is_import_or_delivery(biz, t) or not (("item" in r or "item_code" in r) and ("amount" in r or "qty" in r)):
            continue
        score = 3 * ("amount" in r) + 2 * ("qty" in r) + 2 * ("order_id" in r) + 2 * ("date" in r) + ("status" in r) + len(biz.tables[t]) / 1e7
        if score > best_score:
            best, best_score = t, score
    if best is None:
        return _orders_only(biz, roles_by_table)
    lt = best
    lr = roles_by_table[lt]
    L = biz.tables[lt]
    S = pd.DataFrame(index=L.index)
    mapping: dict[str, str] = {}
    notes: list[str] = []
    for role in ("order_id",) + LINE_ROLES + ORDER_ROLES:
        if role in lr:
            S[role] = L[lr[role]]
            mapping[role] = f"{lt}.{lr[role]}"
    oid_col = lr.get("order_id")
    if "order_id" not in S:
        oid = _guess_order_id(biz, lt, roles_by_table)
        if oid is not None:
            oid_col = oid
            S["order_id"] = L[oid]
            mapping["order_id"] = f"{lt}.{oid} (nhận theo giá trị mã)"
    if "order_id" not in S and "line_no" in lr and _is_order_number(biz, L, lr["line_no"]):
        oid_col = lr["line_no"]
        part = L[[c for c in META_COLS if c in L.columns]].astype(str).agg("/".join, axis=1) if any(c in L.columns for c in META_COLS) else ""
        num = L[oid_col].map(lambda v: str(int(v)) if isinstance(v, float) and v == v and v.is_integer() else str(v))
        S["order_id"] = (part + "#" + num) if isinstance(part, pd.Series) else num
        S = S.drop(columns=["line_no"], errors="ignore")
        mapping.pop("line_no", None)
        mapping["order_id"] = f"{lt}.{oid_col} (số thứ tự đơn trong sổ; ghép với sheet vì mỗi sheet đánh lại từ 1)"
    if "order_id" not in S:
        notes.append("bảng dòng hàng không có mã đơn: mỗi dòng tính như một bản ghi, không gom được theo đơn")
    elif biz.info[lt].key != oid_col:
        # loader chua nhan ra ma don nen chua bo dong trung trong cung don (§8): bo o day
        cmp = [c for c in L.columns if c not in META_COLS]
        dup = L.duplicated(subset=cmp, keep="first")
        if dup.any():
            notes.append(f"đã bỏ {int(dup.sum())} dòng hàng trùng hệt trong cùng đơn (§8)")
            S = S.loc[~dup]
            L = L.loc[~dup]
    # 2) bang don (noi qua ma don, tim cot noi theo do trung gia tri) cho cac vai tro cap don con thieu
    if "order_id" in S:
        ids = set(S["order_id"].dropna().astype(str).unique())
        cands = []
        for t, r in roles_by_table.items():
            if t == lt or _is_import_or_delivery(biz, t):
                continue
            df_t = biz.tables[t]
            key_cols = [r["order_id"]] if "order_id" in r else []
            key_cols += [c for c in df_t.columns if c not in key_cols and _kind(biz, t, c) == "text" and c not in META_COLS]
            for kc in key_cols:
                other = df_t[kc].dropna().astype(str)
                if other.empty or other.duplicated().any():
                    continue
                overlap = other.isin(ids).mean()
                useful = [x for x in ORDER_ROLES if x in r and x not in S and r[x] != kc]
                if overlap > 0.5 and useful:
                    cands.append((len(useful), overlap, t, kc))
                    break
        if cands:
            _, _, ot, okey = max(cands)
            orr = dict(roles_by_table[ot])
            orr["order_id"] = okey
            od = biz.tables[ot].drop_duplicates(subset=[orr["order_id"]]).set_index(orr["order_id"])
            od.index = od.index.astype(str)
            key = S["order_id"].astype(str)
            for role in ORDER_ROLES:
                if role in orr and role not in S:
                    S[role] = key.map(od[orr[role]])
                    mapping[role] = f"{ot}.{orr[role]}"
    if "status" not in S:
        note = _status_note_column(biz, lt)
        if note is not None:
            S["status"] = L[note]
            mapping["status"] = f"{lt}.{note} (ghi chú của đơn ghi huỷ/trả/đang giao; không nhắc = hoàn thành)"
    # 3) ten hang tu ma hang qua danh muc
    if "item" not in S and "item_code" in S:
        hit = _catalog_names(biz, lt, roles_by_table, S["item_code"])
        if hit is not None:
            t, code_col, name_col = hit
            cat = biz.tables[t].drop_duplicates(subset=[code_col]).set_index(code_col)[name_col]
            S["item"] = S["item_code"].map(cat)
            mapping["item"] = f"{t}.{name_col} (nối qua mã hàng {lt}.{lr['item_code']} = {t}.{code_col})"
    if "item" in S:
        abbr = _abbreviations(biz)
        if abbr:
            raw = S["item"]
            full = raw.map(lambda v: abbr.get(norm_key(v)) if isinstance(v, str) else None)
            if full.notna().any():
                S["item_raw"] = raw
                S["item"] = full.fillna(raw)
                mapping["item"] = mapping.get("item", "") + " (ghi tắt đã đổi ra tên đầy đủ theo bảng Ghi tắt)"
    if "item" in S and "item_code" in S:
        # file co ma thi theo ma (§8): moi ma hang mot ten (ten gap nhieu nhat)
        names = S.dropna(subset=["item_code", "item"]).groupby("item_code")["item"].agg(lambda x: x.value_counts().index[0])
        S["item"] = S["item_code"].map(names).fillna(S["item"])
    if "date" not in S and "datetime" in S:
        S["date"] = S["datetime"]
        mapping["date"] = mapping["datetime"]
    if "date" not in S or ("amount" not in S and "qty" not in S):
        return None
    S["date"] = pd.to_datetime(S["date"], errors="coerce").dt.normalize()
    if "datetime" in S:
        dtv = pd.to_datetime(S["datetime"], errors="coerce")
        if dtv.notna().mean() > 0.5 and (dtv.dt.hour + dtv.dt.minute).max() > 0:
            # cot gio chi co gio (ngay 1900) -> ghep voi ngay ban
            if (dtv.dt.year < 2000).mean() > 0.5:
                dtv = S["date"] + (dtv - dtv.dt.normalize())
            S["datetime"] = dtv
        else:
            S = S.drop(columns=["datetime"])
    if "datetime" in S:
        hour = S["datetime"].dt.hour
        S["hour"] = hour
        S["ca"] = np.select([(hour >= 6) & (hour <= 13), (hour >= 14) & (hour <= 21)], ["sáng", "chiều"], default=None)
    # 3b) topping ghi chung o voi mon ("Latte (M) + 2 Thach ca phe (14.000đ)"): tach thanh dong mon + dong topping
    S = _split_toppings(biz, S, notes)
    # 4a) thuoc tinh danh muc (nhom hang, hoat chat, thuoc ke don, hang, size...) gan vao S theo ma / ten hang
    _attach_catalog(biz, lt, S, mapping, notes)
    if "group" not in S:
        gcol = next((c for c in S.columns if c.startswith("dm_") and any(k in c for k in ("nhom", "loai_san_pham", "danh_muc", "phan_loai", "loai_hang"))
                     and c != "dm_loai_hang_hoa"), None)
        if gcol:
            S["group"] = S[gcol]
            mapping["group"] = mapping.get(gcol, gcol)
    # 4b) cua hang dien thoai: dong may = phan ten truoc dung luong '...GB' (khong co thi ca ten); hang = chu dau ten (iPhone -> Apple)
    if "item" in S:
        it = S["item"].astype(str)
        gb = it.str.contains(r"\b\d+\s?(?:GB|TB)\b", regex=True)
        if gb.mean() > 0.05:
            # dong may chi cho dong la may (danh muc loai 'máy…'/'điện thoại'); khong co cot loai thi dong co dung luong GB/TB.
            # Phu kien (op lung, the nho 64GB…) khong co dong may / hang.
            kind = next((c for c in S.columns if c.startswith("dm_") and S[c].dtype == object
                         and S[c].dropna().astype(str).str.lower().str.match(r"(máy|may|điện thoại|dien thoai)\b").mean() > 0.05), None)
            phone = S[kind].astype(str).str.lower().str.match(r"(máy|may|điện thoại|dien thoai)\b") if kind else gb
            S["dong_may"] = it.str.extract(r"^(.*?)\s+\d+\s?(?:GB|TB)\b")[0].fillna(it).str.strip().where(phone)
            first = it.str.split().str[0]
            S["hang"] = first.where(~first.str.lower().str.startswith("iphone"), "Apple").where(phone)
            notes.append(f"dong_may = phần tên trước dung lượng …GB (không có thì cả tên); hang = chữ đầu tên hàng (iPhone -> Apple); "
                         f"chỉ đặt cho dòng là máy{f' ({kind} bắt đầu bằng máy/điện thoại)' if kind else ' (có dung lượng GB/TB)'}, phụ kiện để trống")
    # 4) nhom hang tu danh muc (theo ma hoac ten)
    for t, r in roles_by_table.items():
        if t == lt or _is_import_or_delivery(biz, t):
            continue
        grp = next((c for c in biz.tables[t].columns if _desc(biz, t, c).startswith("nhóm hàng") or _split_name(c) in ("nhom hang", "nhom hang hoa", "nhom", "loai hang", "danh muc", "loai san pham", "nhom hang 3 cap")), None)
        if grp is None:
            continue
        cat = biz.tables[t]
        if "item_code" in r and "item_code" in S:
            m = cat.drop_duplicates(subset=[r["item_code"]]).set_index(r["item_code"])[grp]
            S["group"] = S["item_code"].map(m)
        elif "item" in r and "item" in S:
            m = cat.assign(_k=cat[r["item"]].map(compact_key)).drop_duplicates(subset=["_k"]).set_index("_k")[grp]
            S["group"] = S["item"].map(compact_key).map(m)
        else:
            continue
        mapping["group"] = f"{t}.{grp}"
        break
    # 5) so luong co ban
    if "qty_base" not in S and "qty" in S:
        S["qty_base"] = _to_base(biz, S, notes)
        mapping["qty_base"] = "tính = qty × hệ số quy đổi (README / bảng quy đổi)"
    # 6) trang thai don
    line_status = S["status"].map(status_of) if "status" in S else pd.Series(None, index=S.index, dtype=object)
    if "status" in S:
        unknown = S.loc[line_status.isna() & S["status"].notna(), "status"].astype(str).value_counts()
        if len(unknown):
            notes.append("trạng thái không nhận ra (coi như hoàn thành): " + ", ".join(f"{k} ({v})" for k, v in unknown.head(5).items()))
    S["line_status"] = line_status
    ret_line = line_status.eq("trả hàng")
    if "qty" in S:
        ret_line |= pd.to_numeric(S["qty"], errors="coerce") < 0
    if "returned_qty" in S:
        ret_line |= pd.to_numeric(S["returned_qty"], errors="coerce").fillna(0) > 0
    S["is_return_line"] = ret_line
    returned_orders = _returned_orders_from_tables(biz, lt, roles_by_table)
    if "order_id" in S:
        g = S.groupby("order_id", sort=False)
        any_ret = g["is_return_line"].transform("any")
        if returned_orders:
            any_ret |= S["order_id"].astype(str).isin(returned_orders)
            notes.append(f"đơn có phiếu trả hàng ở bảng khác: {len(returned_orders)} đơn")
        any_cancel = g["line_status"].transform(lambda s: s.eq("huỷ").any())
        any_ship = g["line_status"].transform(lambda s: s.eq("đang giao").any())
        # ngay don = ngay cua dong hang dau tien (dong tra hang co the ghi ngay tra)
        first_date = S.loc[~S["is_return_line"]].groupby("order_id")["date"].min()
        S["order_date"] = S["order_id"].map(first_date).fillna(S["date"])
    else:
        any_ret = S["is_return_line"]
        any_cancel = line_status.eq("huỷ")
        any_ship = line_status.eq("đang giao")
        S["order_date"] = S["date"]
    S["status"] = np.select([any_ret, any_cancel, any_ship], ["trả hàng", "huỷ", "đang giao"], default="hoàn thành")
    S["revenue_ok"] = ~S["status"].isin(["huỷ", "trả hàng"])
    if "amount" not in S and "qty" in S and "price" in S:
        S["amount"] = S["qty"] * S["price"]
        mapping["amount"] = "tính = qty × price"
    if "cost" in S and "qty" in S:
        # §5: gia von dong = so luong (don vi ban cua dong) x gia von 1 don vi ban ghi tren dong
        S["cogs"] = pd.to_numeric(S["qty"], errors="coerce") * pd.to_numeric(S["cost"], errors="coerce")
    if "item" in S:
        low = S["item"].astype(str).str.strip().str.lower()
        non_goods = low.str.match(r"^(voucher|dùng\s*\d+\s*điểm|dung\s*\d+\s*diem|phí ship|phi ship|phí giao|phí vận chuyển|giảm giá đơn|khuyến mãi đơn)")
        S["revenue_line"] = ~non_goods  # voucher, dung diem, phi ship khong tinh doanh thu (§1)
        S["is_item"] = ~non_goods & ~low.isin(["phí phục vụ", "phi phuc vu"])
        if non_goods.any():
            notes.append(f"{int(non_goods.sum())} dòng không phải hàng (Voucher / Dùng điểm / Phí ship): revenue_line=False, không tính doanh thu; các dòng này chỉ có trong S_gom_huy_tra (câu hỏi tiền voucher / điểm / phí ship: lọc S_gom_huy_tra theo item, đơn revenue_ok, lấy số dương)")
    else:
        S["revenue_line"] = True
    _attach_customers(biz, lt, S, mapping)
    if "customer" in S:
        S["customer_key"] = S["customer"].map(_customer_key)
    if "customer" in S or "customer_id" in S:
        walk = pd.Series(True, index=S.index)
        if "customer_id" in S:
            walk &= S["customer_id"].isna() | S["customer_id"].astype(str).str.strip().eq("")
        if "customer" in S:
            walk &= S["customer_key"].isna() | S["customer_key"].isin(_WALK_IN)
        S["is_walk_in"] = walk
    # 7) bang don
    O = _orders(S)
    front = [c for c in ("order_id", "order_date", "date", "datetime", "status", "revenue_ok", "item", "item_code", "group", "qty", "unit", "qty_base", "price", "amount", "line_discount", "cost", "customer", "customer_key", "customer_id", "is_walk_in", "customer_group", "channel", "payment", "employee", "revenue_line", "is_item", "is_return_line", "line_status", "line_no", "returned_qty", "item_raw", "hour", "ca", "cogs") if c in S]
    S = S[front + [c for c in S.columns if c not in front]].reset_index(drop=True)
    return SalesModel(S=S, O=O, mapping=mapping, notes=notes)


_ATTR_SKIP = ("tên", "mã", "đơn vị", "giá", "tồn", "định mức", "hình ảnh", "mô tả", "ghi chú", "trọng lượng", "vị trí", "bảo hành", "mẫu", "barcode", "mã vạch", "ngày", "còn kinh doanh", "quy đổi", "hệ số")


_TOPPING_PART = re.compile(r"^(?:(\d+)\s*(?:x\s*)?)?(.+?)\s*\(\s*([\d.,]+\s*(?:k|đ|d|vnd|vnđ|tr)?)\s*\)$", re.I)


def _split_toppings(biz: Business, S: pd.DataFrame, notes: list) -> pd.DataFrame:
    """Rulebook (topping ghi chung o voi mon): mon va topping la hai mat hang ban. Dong 'Mon + [n ]Topping (tien)'
    -> dong mon (tien = thanh tien - tien topping; so luong, gia, don vi giu nguyen) + moi topping mot dong
    (so luong n, tien trong ngoac), gop vao cung mat hang voi topping ghi dong rieng. Chi ap khi README noi ve topping."""
    if "item" not in S or "topping" not in biz.readme.text.lower():
        return S
    item = S["item"].astype(str)
    cand = item.str.contains(r"\s\+\s", regex=True)
    if not cand.any():
        return S
    from .parsing import strip_accents

    names = S.loc[~cand, "item"].dropna().astype(str).unique()
    known = {compact_key(x): x for x in names}
    by_ascii: dict[str, list[str]] = {}
    for x in names:  # topping viet khong dau ("Thach ca phe") -> ten co dau duy nhat khop
        by_ascii.setdefault(compact_key(strip_accents(x)), []).append(x)
    known.update({k: v[0] for k, v in by_ascii.items() if len(v) == 1 and k not in known})
    unit_of = S.loc[~cand].dropna(subset=["item"]).groupby("item")["unit"].agg(lambda u: u.mode().iloc[0] if u.notna().any() else None) if "unit" in S else {}
    new_item, extra, n_split = S["item"].copy(), [], 0
    new_amount = S["amount"].copy() if "amount" in S else None
    for idx in S.index[cand]:
        parts = [p.strip() for p in str(S.at[idx, "item"]).split(" + ")]
        tops = []
        for p in parts[1:]:
            m = _TOPPING_PART.match(p)
            money = parse_number(m.group(3))[0] if m else None
            if not m or money is None:
                tops = None
                break
            tops.append((int(m.group(1)) if m.group(1) else 1, m.group(2).strip(), float(money)))
        if not tops:
            continue
        n_split += 1
        new_item.at[idx] = parts[0]
        if new_amount is not None:
            new_amount.at[idx] = float(S.at[idx, "amount"]) - sum(t[2] for t in tops)
        for n, name, money in tops:
            keys = (compact_key("Topping " + name), compact_key(name), compact_key(strip_accents("Topping " + name)), compact_key(strip_accents(name)))
            key = next((known[k] for k in keys if k in known), "Topping " + name)
            row = S.loc[idx].copy()
            row["item"] = key
            row["qty"] = n if "qty" in S else None
            if "qty_base" in S:
                row["qty_base"] = n
            if "price" in S:
                row["price"] = money / n
            if "amount" in S:
                row["amount"] = money
            if "line_discount" in S:
                row["line_discount"] = 0.0
            if "unit" in S:
                row["unit"] = unit_of.get(key) if hasattr(unit_of, "get") else None
            for c in ("item_code", "cost"):
                if c in S:
                    row[c] = None
            extra.append(row)
    if not n_split:
        return S
    S = S.copy()
    if "item_raw" not in S:
        S["item_raw"] = S["item"]
    S["item"] = new_item
    if new_amount is not None:
        S["amount"] = new_amount
    top = pd.DataFrame(extra)
    top["item_raw"] = S.loc[top.index, "item_raw"].values
    out = pd.concat([S, top], ignore_index=True)
    notes.append(f"{n_split} dòng có topping ghi chung ô với món: đã tách thành dòng món (tiền món = thành tiền − tiền topping) "
                 f"và {len(top)} dòng topping (cùng mặt hàng với topping ghi dòng riêng)")
    return out


def _attach_catalog(biz: Business, lt: str, S: pd.DataFrame, mapping: dict, notes: list) -> None:
    """Gan cot thuoc tinh cua danh muc hang vao S (cot ten 'dm_<ten cot goc>'), noi theo ma hang hoac ten hang."""
    if "item_code" not in S and "item" in S:
        # ten -> ma qua danh muc (de noi tiep cac bang chi co ma, vd danh sach thuoc)
        want = set(S["item"].dropna().map(compact_key))
        for t, df in biz.tables.items():
            if t == lt or len(df) > 3000:
                continue
            ncol = next((c for c in df.columns if df[c].dropna().astype(str).map(compact_key).isin(want).mean() > 0.8 and not df[c].duplicated().any()), None)
            ccol = next((c for c in df.columns if c != ncol and (_desc(biz, t, c).startswith("mã hàng") or _split_name(c) in ("ma sku", "ma hang", "ma sp", "ma hang hoa", "ma mon"))
                         and not df[c].duplicated().any()), None) if ncol else None
            if ncol and ccol:
                m = dict(zip(df[ncol].astype(str).map(compact_key), df[ccol]))
                S["item_code"] = S["item"].map(lambda v: m.get(compact_key(v)) if isinstance(v, str) else None)
                mapping["item_code"] = f"{t}.{ccol} (nối qua tên hàng)"
                break
    s_codes = set(S["item_code"].dropna().astype(str)) if "item_code" in S else set()
    s_names = set(S["item"].dropna().map(compact_key)) if "item" in S else set()
    for t, df in biz.tables.items():
        if t == lt or len(df) > 3000 or _is_import_or_delivery(biz, t):
            continue
        join = None
        for c in df.columns:
            v = df[c].dropna().astype(str)
            if v.empty or v.duplicated().any():
                continue
            if s_codes and pd.Series(list(s_codes)).isin(set(v)).mean() > 0.8:
                join = ("item_code", c, lambda x: x)
                break
            if s_names and pd.Series(list(s_names)).isin(set(v.map(compact_key))).mean() > 0.8:
                join = ("item", c, compact_key)
                break
        if join is None:
            continue
        skey, ckey, f = join
        attrs = []
        for c in df.columns:
            if c == ckey or c in META_COLS:
                continue
            d = _desc(biz, t, c)
            n = _split_name(c)
            k = biz.info[t].columns.get(c).kind if biz.info[t].columns.get(c) else "text"
            if df[c].isna().all() or n in ("id cua hang",) or df[c].nunique() <= 1:
                continue
            if k == "number" and not (df[c].dropna().isin([0, 1]).all() and d.startswith(("thuốc kê đơn", "")) and any(x in n for x in ("ke don", "theo don", "rx"))):
                continue
            if k not in ("text", "bool", "number"):
                continue
            if any(d.startswith(x) for x in _ATTR_SKIP) or any(x in n for x in ("ten ", "ma ", "don vi", "gia", "ton", "hinh anh", "mo ta", "ghi chu", "barcode", "vi tri")) or n.startswith(("ten", "ma")):
                if not d.startswith(("nhóm hàng", "hoạt chất", "thuốc kê đơn", "hãng", "số của sữa", "loại")):
                    continue
            if df[c].nunique() > 300:
                continue
            attrs.append(c)
        if not attrs:
            continue
        idx = df.drop_duplicates(subset=[ckey])
        key_vals = idx[ckey].astype(str).map(f)
        src = S[skey].astype(str).map(f)
        from .loader import _safe_name

        for c in attrs:
            col = "dm_" + _safe_name(c).lower()
            if col in S:
                continue
            S[col] = src.map(dict(zip(key_vals, idx[c])))
            mapping[col] = f"{t}.{c} (nối qua {'mã hàng' if skey == 'item_code' else 'tên hàng'})"


def _attach_customers(biz: Business, lt: str, S: pd.DataFrame, mapping: dict) -> None:
    """Gan thuoc tinh cua bang khach (nhom khach, loai khach, ngay dang ky...) vao S, cot 'kh_<ten cot>'."""
    from .loader import _safe_name

    had_group = "customer_group" in S  # nhom khach da co tu so ban thi giu
    for key in ("customer_id", "customer"):
        if key not in S:
            continue
        want = set(S[key].dropna().astype(str).map(compact_key))
        if not want:
            continue
        for t, df in biz.tables.items():
            if t == lt or len(df) > 20000:
                continue
            kc = next((c for c in df.columns if not df[c].dropna().duplicated().any() and len(df[c].dropna())
                       and pd.Series(list(want)).isin(set(df[c].dropna().astype(str).map(compact_key))).mean() > 0.6), None)
            if kc is None:
                continue
            idx = df.drop_duplicates(subset=[kc]).set_index(df.drop_duplicates(subset=[kc])[kc].astype(str).map(compact_key))
            src = S[key].astype(str).map(compact_key)
            added = False
            for c in df.columns:
                if c == kc or c in META_COLS:
                    continue
                d = _desc(biz, t, c)
                n = _split_name(c)
                if not had_group and (d.startswith("nhóm khách") or n in ("nhom khach", "nhom khach hang", "loai khach", "doi tuong kh", "hang", "nhom kh", "loai khach hang", "kieu khach", "hang thanh vien")):
                    S["customer_group"] = src.map(idx[c])
                    mapping["customer_group"] = f"{t}.{c} (nối qua {key})"
                    added = True
                elif (d.startswith("ngày đăng ký") or "dang ky" in n) and _kind(biz, t, c) in ("date", "datetime"):
                    S["kh_ngay_dang_ky"] = src.map(idx[c])
                elif key == "customer_id" and "customer" not in S and df[c].dtype == object and (
                        d.startswith(("tên khách", "họ tên", "tên gọi")) or d.strip(" .") == "tên" or n in ("ten goi", "ten khach", "ten khach hang", "ho ten", "ho va ten", "ten kh", "ho ten khach")):
                    # so ban chi ghi ma khach: lay ten tu bang khach (§3 van xac dinh khach theo ma)
                    S["customer"] = src.map(idx[c]).where(S[key].notna())
                    mapping["customer"] = f"{t}.{c} (nối qua {key})"
                    added = True
                elif "customer_gender" not in S and (d.startswith("giới tính") or n in ("gioi tinh", "nam nu", "phai", "gioi")) and df[c].dtype == object:
                    S["customer_gender"] = src.map(idx[c])
                    mapping["customer_gender"] = f"{t}.{c} (nối qua {key})"
            if added:
                return


def _is_order_number(biz: Business, L: pd.DataFrame, col: str) -> bool:
    """Cot STT la so thu tu DON (cac dong cung STT la mot don) chu khong phai so dong trong don."""
    txt = biz.readme.text.lower()
    if f"cùng **{col.lower()}** là một đơn" in txt or f"cùng {col.lower()} là một đơn" in txt:
        return True
    v = pd.to_numeric(L[col], errors="coerce")
    if v.isna().mean() > 0.05:
        return False
    parts = L.groupby([c for c in META_COLS if c in L.columns], sort=False)[col] if any(c in L.columns for c in META_COLS) else [(None, v)]
    mono = all((pd.to_numeric(g, errors="coerce").diff().dropna() >= 0).mean() > 0.99 for _, g in parts)
    return mono and v.duplicated().mean() > 0.2


def _status_note_column(biz: Business, table: str) -> str | None:
    for c in biz.tables[table].columns:
        d = _desc(biz, table, c)
        if d.startswith("ghi chú của đơn") or ("ghi chú" in d and any(k in d for k in ("huỷ", "hủy", "trả lại", "đang giao"))):
            return c
    for c in biz.tables[table].columns:
        if " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(c)).split()) in ("ghi chu", "note", "ghi chu tt", "ghi chu don"):
            vals = biz.tables[table][c].dropna().astype(str)
            if len(vals) and vals.map(status_of).notna().mean() > 0.3:
                return c
    return None


def _abbreviations(biz: Business) -> dict[str, str]:
    for t, df in biz.tables.items():
        cols = {_split_name(c): c for c in df.columns}
        if "ghi tat" in cols and ("ten hang" in cols or "ten day du" in cols):
            full = cols.get("ten hang") or cols.get("ten day du")
            return {norm_key(a): f for a, f in zip(df[cols["ghi tat"]], df[full]) if isinstance(a, str) and isinstance(f, str)}
    return {}


_WALK_IN = {"khách lẻ", "khach le", "kl", "lẻ", "le", "khách vãng lai", "vãng lai", "khách", "k l", "kh lẻ", "khach"}

_PHONE_RE = re.compile(r"(?:\+?84|0)(?:[\s.\-]?\d){8,10}")


def _customer_key(name) -> str | None:
    """Khoa khach theo ten (§3): bo khac biet cach viet; so dien thoai ghi kem (cach ghi khac nhau) duoc chuan hoa."""
    if not isinstance(name, str) or not name.strip():
        return None
    phone = "".join(ch for ch in "".join(_PHONE_RE.findall(name)) if ch.isdigit())
    base = _PHONE_RE.sub(" ", name)
    base = norm_key(re.sub(r"[()\-–,:;|/]+", " ", base))
    return base + (f" | {phone}" if phone else "")


_CODE_RE = re.compile(r"^[A-Za-z]{1,8}[-_]?\d{2,}$")


def _code_like(series: pd.Series) -> bool:
    v = series.dropna().astype(str)
    return len(v) > 0 and v.head(2000).str.match(_CODE_RE).mean() > 0.9


def _guess_order_id(biz: Business, lt: str, roles_by_table) -> str | None:
    """Cot ma don nhan theo gia tri: ma dang 'ORD000123', lap lai giua cac dong cung don, trung voi
    cot ma (khong lap) cua bang khac."""
    L = biz.tables[lt]
    best = None
    for c in L.columns:
        if c in META_COLS or _kind(biz, lt, c) != "text" or not _code_like(L[c]):
            continue
        vals = L[c].dropna().astype(str)
        if not vals.duplicated().any():
            continue
        ids = set(vals.unique())
        for t in biz.tables:
            if t == lt or _is_import_or_delivery(biz, t):
                continue
            for oc in biz.tables[t].columns:
                ov = biz.tables[t][oc].dropna().astype(str)
                if ov.empty or ov.duplicated().any() or not _code_like(ov):
                    continue
                score = ov.isin(ids).mean()
                if score > 0.5 and (best is None or score > best[0]):
                    best = (score, c)
    return best[1] if best else None


def _catalog_names(biz: Business, lt: str, roles_by_table, codes: pd.Series):
    """Tim danh muc co cot ma trung voi ma hang cua dong ban, tra ve (bang, cot ma, cot ten)."""
    want = set(codes.dropna().astype(str).unique())
    best = None
    for t, r in roles_by_table.items():
        if t == lt or _is_import_or_delivery(biz, t):
            continue
        df = biz.tables[t]
        for cc in df.columns:
            v = df[cc].dropna().astype(str)
            if v.empty or v.duplicated().any():
                continue
            cover = pd.Series(list(want)).isin(set(v)).mean() if want else 0
            if cover < 0.8:
                continue
            name_col = r.get("item") if r.get("item") != cc else None
            if name_col is None:
                texts = [c for c in df.columns if c != cc and _kind(biz, t, c) == "text" and not _code_like(df[c])
                         and df[c].nunique() >= 0.8 * len(df) and df[c].astype(str).str.len().median() > 4]
                pref = [c for c in texts if any(k in _split_name(c) for k in ("ten", "mon", "hang", "san pham", "item"))]
                name_col = (pref or texts or [None])[0]
            if name_col is not None and (best is None or cover > best[0]):
                best = (cover, t, cc, name_col)
    return best[1:] if best else None


def _orders_only(biz: Business, roles_by_table) -> SalesModel | None:
    """Shop chi co danh sach hoa don (khong co dong hang): dung O tu bang don."""
    for t, r in roles_by_table.items():
        if "order_id" not in r or "date" not in r or _is_import_or_delivery(biz, t):
            continue
        df = biz.tables[t]
        rev = next((c for c in df.columns if _desc(biz, t, c).startswith(("tiền hàng của đơn", "tổng tiền hàng của đơn"))), None)
        if rev is None:
            rev = next((c for c in df.columns if not _desc(biz, t, c) and " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(c)).split()) in ("tien hang", "gia tri hang", "tong tien hang")), None)
        if rev is None or df[r["order_id"]].duplicated().mean() > 0.01:
            continue
        df = df.drop_duplicates(subset=[r["order_id"]])
        O = pd.DataFrame({"order_id": df[r["order_id"]], "order_date": pd.to_datetime(df[r["date"]], errors="coerce").dt.normalize()})
        st = df[r["status"]].map(status_of) if "status" in r else pd.Series("hoàn thành", index=df.index)
        O["status"] = st.fillna("hoàn thành")
        O["revenue_ok"] = ~O["status"].isin(["huỷ", "trả hàng"])
        O["revenue"] = pd.to_numeric(df[rev], errors="coerce")
        mapping = {"order_id": f"{t}.{r['order_id']}", "order_date": f"{t}.{r['date']}", "revenue": f"{t}.{rev}"}
        for role in ("status", "customer", "customer_id", "channel", "payment", "employee", "datetime"):
            if role in r:
                if role != "status":
                    O[role] = df[r[role]].values
                mapping[role] = f"{t}.{r[role]}"
        if "customer" in O or "customer_id" in O:  # khoa khach va khach le nhu bang S (§3)
            walk = pd.Series(True, index=O.index)
            if "customer" in O:
                O["customer_key"] = O["customer"].map(_customer_key)
                walk &= O["customer_key"].isna() | O["customer_key"].isin(_WALK_IN)
            if "customer_id" in O:
                walk &= O["customer_id"].isna() | O["customer_id"].astype(str).str.strip().eq("")
            O["is_walk_in"] = walk
        return SalesModel(S=pd.DataFrame(), O=O.reset_index(drop=True), mapping=mapping,
                          notes=["shop không có chi tiết mặt hàng: chỉ có O (mỗi dòng một đơn), revenue = tiền hàng của đơn"])
    return None


def _orders(S: pd.DataFrame) -> pd.DataFrame:
    if "order_id" not in S:
        return pd.DataFrame()
    agg = {"order_date": "first", "status": "first", "revenue_ok": "first"}
    for c in ("datetime", "customer", "customer_key", "customer_id", "is_walk_in", "customer_group", "channel", "payment", "employee"):
        if c in S:
            agg[c] = "first"
    g = S.groupby("order_id", sort=False)
    O = g.agg(agg)
    if "amount" in S:
        sel = ~S["is_return_line"] & S.get("revenue_line", True)
        O["revenue"] = S.loc[sel].groupby("order_id")["amount"].sum().reindex(O.index).fillna(0)
    O["n_lines"] = g.size()
    if "datetime" in S:
        O["datetime"] = S.loc[~S["is_return_line"]].groupby("order_id")["datetime"].min().reindex(O.index).fillna(O["datetime"])
        hour = O["datetime"].dt.hour
        O["hour"] = hour
        O["ca"] = np.select([(hour >= 6) & (hour <= 13), (hour >= 14) & (hour <= 21)], ["sáng", "chiều"], default=None)
    return O.reset_index()


def _returned_orders_from_tables(biz: Business, lt: str, roles_by_table) -> set[str]:
    """Don co tra hang ghi o bang khac: bang phieu tra (moi don trong bang) hoac bang co cot
    'so luong khach tra lai' (chi don co so luong tra > 0)."""
    out: set[str] = set()
    for t in biz.tables:
        if t == lt:
            continue
        r = roles_by_table[t]
        oid = r.get("order_id")
        if not oid:
            continue
        df = biz.tables[t]
        if "returned_qty" in r:
            q = pd.to_numeric(df[r["returned_qty"]], errors="coerce").fillna(0)
            out |= set(df.loc[q > 0, oid].dropna().astype(str))
            continue
        tn = _split_name(t)
        cols = " ".join(_split_name(c) for c in df.columns)
        if "tra hang" in tn or "phieu tra" in tn or "ma phieu tra" in cols:
            out |= set(df[oid].dropna().astype(str))
    return out


def _conversion_table(biz: Business) -> dict[tuple[str, str], float]:
    """(ten hang chuan, don vi lon) -> he so ra don vi co ban, tu README va bang quy doi."""
    conv: dict[tuple[str, str], float] = {}
    import re

    for line in biz.readme.text.splitlines():
        m = re.match(r"^\s*-\s*(.+?):\s*1\s+(.+?)\s*=\s*([\d.,]+)\s+(.+?)\s*$", line)
        if m:
            k = parse_number(m.group(3))[0]
            if k:
                conv[(norm_key(m.group(1)), norm_key(m.group(2)))] = k
    for t, df in biz.tables.items():
        cols = {c: _desc(biz, t, c) for c in df.columns}
        nm = {c: " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(c)).split()) for c in df.columns}
        fac = next((c for c, d in cols.items() if d.startswith("hệ số")), None) or next(
            (c for c, n in nm.items() if n in ("he so quy doi", "he so", "ti le quy doi", "ty le quy doi", "quy doi", "bang", "so luong quy doi")
             and _kind(biz, t, c) == "number"), None)
        big = next((c for c, d in cols.items() if d.startswith("đơn vị lớn")), None) or next(
            (c for c, n in nm.items() if n in ("don vi quy doi", "don vi tinh", "dvt", "don vi", "dv", "quy cach", "dv nhap", "loai dv") and not cols[c]), None)
        name = next((c for c, d in cols.items() if d.startswith("tên hàng") or d == "tên"), None) or next(
            (c for c, n in nm.items() if n in ("ten san pham", "ten hang", "hang", "ten mon", "san pham") and not cols[c]), None)
        code = next((c for c, d in cols.items() if d.startswith("mã hàng")), None) or next(
            (c for c, n in nm.items() if n in ("ma hang hoa", "ma hang", "ma sp", "ma sku", "ma vach", "ma h hoa", "ma hh") and not cols[c]), None)
        if not (fac and big and (name or code)):
            continue
        for _, r in df.iterrows():
            f, u = r.get(fac), r.get(big)
            if not isinstance(u, str) or f is None or f != f:
                continue
            for k in (name, code):
                if k and isinstance(r.get(k), str):
                    conv[(norm_key(r[k]), norm_key(u))] = float(f)
    return conv


def _to_base(biz: Business, S: pd.DataFrame, notes: list[str]) -> pd.Series:
    conv = _conversion_table(biz)
    qty = pd.to_numeric(S["qty"], errors="coerce")
    if not conv or "unit" not in S or "item" not in S:
        if conv:
            notes.append("có hệ số quy đổi nhưng thiếu cột đơn vị/tên hàng: qty_base = qty")
        return qty
    units = S["unit"].map(norm_key)
    names = S["item"].map(norm_key)
    codes = S["item_code"].map(norm_key) if "item_code" in S else pd.Series("", index=S.index)
    f = pd.Series([conv.get((n, u), conv.get((c, u), 1.0)) for n, c, u in zip(names, codes, units)], index=S.index)
    return qty * f


def describe(model: SalesModel) -> str:
    S, O = model.S, model.O
    lines = ["## Bảng chuẩn đã dựng sẵn: `S` (dòng hàng) và `O` (đơn) — dùng cho doanh thu, số đơn, mặt hàng, số lượng, huỷ/trả, khách, kênh, thanh toán, ca, nhân viên"]
    lines.append("CÁCH NHANH VÀ ĐÚNG NHẤT: dùng hàm chỉ số revenue / n_orders / qty_sold / top_items / cancel_rate / n_cancelled / avg_order_value / customers / price_on (xem danh sách hàm).")
    lines.append("QUAN TRỌNG: `S` và `O` ĐÃ LỌC SẴN chỉ gồm dòng/đơn TÍNH DOANH THU (đơn không huỷ, không trả; bỏ dòng Voucher/Dùng điểm/Phí ship). Số đơn (kể cả khi hỏi 'tổng cộng bao nhiêu đơn', 'mấy bill'), doanh thu, bán chạy, số lượng bán: dùng S/O hoặc hàm chỉ số. CHỈ dùng `S_gom_huy_tra` / `O_gom_huy_tra` khi câu hỏi nói rõ về đơn HUỶ, đơn TRẢ HÀNG hoặc TỈ LỆ HUỶ.")
    lines.append("Câu hỏi về KHÁCH (khách mua nhiều tiền nhất, nhiều đơn nhất, trung thành, sộp, hay đổi ý, số khách, % khách…): luôn dùng customers(a, b) — đã bỏ khách lẻ và xác định khách theo §3; không tự groupby cột tên khách.")
    lines.append("Đã áp sẵn rulebook §1, §3, §8 (dựng tự động từ README; vẫn có thể đối chiếu bảng gốc trong T):")
    if len(S):
        lines += [
            "- status: trạng thái ĐƠN ∈ {'hoàn thành','huỷ','trả hàng','đang giao'} (đơn có trả hàng dù một phần = 'trả hàng') — chỉ có ý nghĩa trong S_gom_huy_tra/O_gom_huy_tra.",
            "- order_date: ngày ĐẶT của đơn (lọc kỳ theo cột này); amount: thành tiền dòng sau giảm dòng; doanh thu = S.amount.sum() trong kỳ.",
            "- qty_base: số lượng quy về đơn vị cơ bản (bán chạy, 'bán được bao nhiêu <đơn vị>'); is_item=False ở dòng Phí phục vụ (vẫn tính doanh thu, không xếp hạng mặt hàng).",
            "- cogs = qty × cost (giá vốn dòng, §5) nếu shop ghi giá vốn trên dòng bán; lãi gộp = S.amount.sum() − S.cogs.sum() trong kỳ. Không có cột cost -> không tính lãi (unanswerable), không lấy giá vốn danh mục.",
            "- hour, ca ('sáng' 06:00–13:59 / 'chiều' 14:00–21:59 theo giờ tạo đơn, §2) nếu shop ghi giờ (có ở cả S và O).",
            "- customer_key: khoá khách theo tên (đã gộp cách viết, số điện thoại ghi kèm); is_walk_in: khách lẻ (không tính là khách); customer_group: nhóm khách (từ bảng khách); group / dm_*: thuộc tính danh mục (nhóm hàng, hoạt chất, thuốc kê đơn, hãng, size…).",
        ]
    if len(O):
        lines.append("- O: mỗi đơn một dòng (order_id, order_date, status, revenue = doanh thu của đơn, n_lines, khách, hour, ca…). Số đơn trong kỳ = len(O lọc kỳ). Tỉ lệ huỷ = (O_gom_huy_tra.status=='huỷ') / len(O_gom_huy_tra) trong kỳ.")
    lines.append("Nguồn từng cột: " + "; ".join(f"{k} ← {v}" for k, v in model.mapping.items()))
    if model.notes:
        lines.append("Lưu ý khi dựng: " + "; ".join(model.notes))
    if len(S):
        lines.append(f"S_gom_huy_tra: {len(S)} dòng; cột: {', '.join(S.columns)}")
    if len(O):
        lines.append(f"O_gom_huy_tra: {len(O)} đơn ({', '.join(f'{k} {v}' for k, v in O['status'].value_counts().items())}); cột: {', '.join(O.columns)}; order_date từ {O['order_date'].min():%Y-%m-%d} đến {O['order_date'].max():%Y-%m-%d}")
    lines.append("Câu hỏi cần thông tin S/O không có (nhập hàng, tồn kho, công nợ, giao hàng, giá vốn theo lô…) thì dùng phần dựng sẵn tương ứng hoặc bảng gốc trong T.")
    return "\n".join(lines)
