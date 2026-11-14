"""Mo ta cac bang da lam sach cua mot cua hang de dua vao prompt (gon, co vi du gia tri)."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .loader import Business

KIND_VI = {
    "date": "ngày",
    "datetime": "ngày giờ",
    "time": "giờ (chuỗi HH:MM:SS)",
    "number": "số",
    "bool": "đúng/sai",
    "text": "chuỗi",
    "empty": "trống hết",
}


def _fmt_val(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        return v.strftime("%Y-%m-%d %H:%M:%S") if (v.hour or v.minute or v.second) else v.strftime("%Y-%m-%d")
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def _column_line(df: pd.DataFrame, col: str, ci) -> str:
    s = df[col]
    kind = ci.kind if ci else ("số" if pd.api.types.is_numeric_dtype(s) else "chuỗi")
    parts = [f'  - "{col}" [{KIND_VI.get(kind, kind)}]']
    if ci and ci.description:
        parts.append(ci.description)
    nn = s.dropna()
    if kind in ("date", "datetime") and len(nn):
        parts.append(f"từ {_fmt_val(nn.min())} đến {_fmt_val(nn.max())}")
    elif kind == "number" and len(nn):
        parts.append(f"min {_fmt_val(nn.min())}, max {_fmt_val(nn.max())}")
    elif kind in ("text", "bool", "time") and len(nn):
        vc = nn.astype(str).value_counts()
        if len(vc) <= 12:
            parts.append("giá trị: " + "; ".join(f"{_fmt_val(k)} ({v})" for k, v in vc.items()))
        else:
            parts.append(f"{len(vc)} giá trị khác nhau, vd: " + "; ".join(_fmt_val(k) for k in vc.index[:4]))
    if ci and ci.examples:
        parts.append("đã đổi từ ô gốc: " + ", ".join(f"{r!r}→{_fmt_val(c) if not isinstance(c, str) else c}" for r, c in ci.examples[:2]))
    if ci and getattr(ci, "_canon", None):
        parts.append("đã gộp cách viết khác về một tên: " + ", ".join(f"{a!r}→{b!r}" for a, b in ci._canon[:2]))
    if ci and ci.unit_column:
        parts.append(f'chữ đơn vị trong ô gốc tách ra cột "{ci.unit_column}"')
    nulls = int(s.isna().sum())
    if nulls and kind != "empty":
        parts.append(f"{nulls} ô trống")
    return " | ".join(parts)


def business_profile(biz: Business) -> str:
    out = [f"# Cửa hàng {biz.id}", "", "## README của cửa hàng (nguyên văn)", biz.readme.text.strip(), ""]
    out.append("## Các bảng đã nạp sẵn (biến `T`: dict tên bảng -> pandas DataFrame đã làm sạch)")
    out.append(
        "Đã làm sạch: tiền/số ('90k', '12tr460', '84,000') đã đổi sang số; ngày ('1 thg 3 2025', '1/1', số serial Excel)"
        " đã đổi sang datetime64 (ngày không ghi năm lấy năm của sheet/file); chuỗi đã bỏ khoảng trắng thừa."
    )
    for name, df in biz.tables.items():
        inf = biz.info.get(name)
        head = f'### T["{name}"]: {len(df)} dòng'
        if inf:
            if len(inf.sources) > 1:
                srcs = sorted({f + (f"/{s}" if s else "") for f, s in inf.sources})
                shown = ", ".join(srcs[:3]) + (f", ... ({len(srcs)} nguồn)" if len(srcs) > 3 else "")
                head += f"; gộp từ {shown}; cột `_file`, `_sheet` ghi nguồn"
            else:
                f, sh = inf.sources[0]
                head += f"; nguồn {f}" + (f" / sheet {sh}" if sh and not f.endswith('.sqlite') else "")
            if inf.key:
                head += f'; khoá đơn/phiếu: "{inf.key}"'
            if inf.ffilled:
                head += f"; đã điền ô trống ở dòng sau của cùng đơn theo dòng đầu đơn (§8): {', '.join(inf.ffilled)}"
            if inf.dropped_rows:
                head += f"; đã bỏ {inf.dropped_rows} dòng tổng/ghi chú cuối sheet (không phải giao dịch)"
            if inf.duplicates_dropped:
                head += f"; đã bỏ {inf.duplicates_dropped} dòng trùng hệt trong cùng đơn (§8)"
        out.append(head)
        for col in df.columns:
            if col.endswith("__dv"):
                continue
            ci = inf.columns.get(col) if inf else None
            if ci is not None:
                ci._canon = inf.canonicalized.get(col)
            if ci is None and col in ("_file", "_sheet"):
                vc = df[col].astype(str).value_counts()
                out.append(f'  - "{col}" [nguồn] {len(vc)} giá trị, vd: {", ".join(vc.index[:3])}')
                continue
            out.append(_column_line(df, col, ci))
    sales = getattr(biz, "sales", None)
    if sales is not None:
        from .semantic import describe

        out.append("")
        out.append(describe(sales))
        S = getattr(sales, "S", None)
        if S is not None and "employee" in S:
            from .staff import employee_names

            emap = employee_names(biz.tables, S["employee"])
            if emap:
                out.append(f"- employee_name (S, O): họ tên nhân viên nối từ bảng nhân viên theo mã ({', '.join(f'{k} = {v}' for k, v in list(emap.items())[:6])}); "
                           "câu hỏi nêu tên nhân viên thì lọc employee_name=<họ tên>, trả lời bằng họ tên.")
        if S is not None and "customer_gender" in S:
            out.append(f"- customer_gender (S): giới tính khách nối từ bảng khách ({', '.join(map(str, S['customer_gender'].dropna().unique()[:4]))}); "
                       "khách lẻ không ghi mã thì trống.")
    inventory = getattr(biz, "inventory", None)
    if inventory is not None:
        from .inventory import describe as describe_inv

        out.append("")
        out.append(describe_inv(inventory))
    imports = getattr(biz, "imports", None)
    if imports is not None:
        from .imports import describe as describe_imports

        out.append("")
        out.append(describe_imports(imports))
    debt = getattr(biz, "debt", None)
    if debt is not None:
        from .debt import describe as describe_debt

        out.append("")
        out.append(describe_debt(debt))
    inst = getattr(biz, "installments", None)
    if inst is not None:
        from .installment import describe as describe_inst

        out.append("")
        out.append(describe_inst(inst))
    from .delivery import build_delivery, describe as describe_delivery

    D_giao = build_delivery(biz.tables, sales.O if sales is not None else None)
    if D_giao is not None:
        out.append("")
        out.append(describe_delivery(D_giao))
    po = getattr(biz, "phone_orders", None)
    if po is not None:
        out.append("")
        out.append("## Sổ bán (khách trả, bớt, thu máy cũ) dựng sẵn")
        out.append(f"- PO: mỗi đơn một dòng ({len(po)} đơn): order_id, order_date, status, revenue_ok, khach_tra (tiền cả đơn đã bớt và đã trừ máy cũ, gồm phần còn nợ), "
                   f"bot, thu_may_cu (tổng giá thu máy cũ trừ vào đơn; {int((po['thu_may_cu'] > 0).sum())} đơn có thu máy cũ). Đã kiểm: tiền hàng − bot − thu_may_cu = khach_tra ở mọi đơn.")
        out.append("- customer_pays(kỳ) tổng khách phải trả; trade_in_total(kỳ) tổng tiền thu máy cũ; discount_total(kỳ) tổng tiền bớt — chỉ đơn không huỷ, không trả (§1). "
                   "Tiền thu máy cũ KHÔNG trừ vào doanh thu. Tỉ lệ đơn có thu máy cũ = số đơn revenue_ok có thu_may_cu > 0 / n_orders(kỳ).")
    W = getattr(biz, "phone_warranty", None)
    if W is not None:
        out.append("")
        out.append("## Sổ bảo hành máy dựng sẵn")
        out.append(f"- W_bh: {len(W)} lượt bảo hành (mỗi lần khách mang máy tới một dòng), cột {', '.join(W.columns)}; máy xác định theo imei; dong_may ghi dòng máy (không dung lượng, màu).")
        out.append("- phone_warranty_rate(kỳ) = số lượt bảo hành có ngay_nhan trong kỳ / số máy bán trong kỳ (bỏ đơn huỷ, trả) × 100. "
                   "Máy bán tháng X đã mang tới bảo hành = số imei khác nhau trong W_bh có ngay_mua trong tháng X và ngay_nhan đến hết hôm nay. Bảo hành không phải đơn bán.")
    repair = getattr(biz, "repair", None)
    if repair is not None:
        from .repair import describe as describe_repair

        out.append("")
        out.append(describe_repair(repair))
    if biz.warnings:
        out.append("Cảnh báo khi nạp: " + "; ".join(biz.warnings))
    return "\n".join(out)
