"""DataMate: README-driven ingestion and deterministic execution. No answer lookup.

Inference accepts only shop data, its README, a question and an NLU plan.
answers.jsonl is opened exclusively by the separate evaluation script.
"""
from __future__ import annotations
import calendar, hashlib, json, math, pickle, re, sqlite3, unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import pandas as pd
import sys
# Old prepared-shop pickles record engine.Shop; the ingestion implementation
# is retained here, so accept that module name without changing the 4B code.
sys.modules.setdefault("engine",sys.modules[__name__])

VERSION = "datamate-engine-0.3.4"
PARSER_VERSION = "datamate-engine-0.4.2"

def norm(value):
    if value is None or (not isinstance(value, (list,dict)) and pd.isna(value)):
        return ""
    text = str(value).replace("đ", "d").replace("Đ", "D")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

def compact(value):
    return norm(value).replace(" ", "")

def question_norm(value):
    text = norm(value)
    aliases = {"dthu":"doanh thu", "hnay":"hom nay", "hqua":"hom qua", "sp":"san pham",
               "bnhiu":"bao nhieu", "bn":"bao nhieu", "dc":"duoc", "ko":"khong",
               "ddown":"don", "thangs":"thang", "trc":"truoc", "vs":"so voi",
               "soos":"so", "nhiu":"nhieu", "sangs":"sang", "nowj":"no",
               "nhaapj":"nhap", "tieenf":"tien", "tm":"tien mat", "nhoms":"nhom"}
    return " ".join(aliases.get(w,w) for w in text.split())

def money(value):
    return int(Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def rounded(value):
    return float(Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def number_series(series):
    def parse(x):
        if pd.isna(x) or x == "": return float("nan")
        if isinstance(x,bool):return float('nan')
        if isinstance(x,(float,int)): return float(x)
        text = re.sub(r'(?i)(?:vnd|vnđ|đồng|dong|[₫đĐ])','',str(x).strip()).replace('−','-')
        multiplier=1
        suffix=re.search(r'(?i)\s*(triệu|trieu|tr|nghìn|nghin|ngàn|ngan|k)\s*$',text)
        if suffix:
            multiplier=1_000_000 if suffix.group(1).lower() in {'triệu','trieu','tr'} else 1000
            text=text[:suffix.start()]
        text = re.sub(r'\s+','',text)
        if text.startswith('(') and text.endswith(')'): text='-'+text[1:-1]
        if ',' in text and '.' in text:
            if text.rfind(',')>text.rfind('.'):
                text=text.replace('.','').replace(',','.')
            else: text=text.replace(',','')
        if re.fullmatch(r"-?\d{1,3}(?:[.,]\d{3})+",text): text = re.sub(r"[.,]","",text)
        elif "," in text and "." not in text: text = text.replace(",", ".")
        return pd.to_numeric(text,errors="coerce")*multiplier
    return series.map(parse)

ALIASES = {
    "order_id": ["Mã đơn hàng","Mã hóa đơn","Mã chứng từ","Mã bill","Mã GD","Số CT","Số HĐ","Số phiếu","Ma don","Order ID","MaHoaDon"],
    "product_id": ["Mã hàng","Mã hàng hóa","Mã sản phẩm","Mã HH","Mã mặt hàng","Mã SKU","SKU","ID SP","MaHangHoa","ma_sp","Mã món","Mã sp","Mã hàng bán","Mã sản phẩm bán"],
    "product": ["Tên hàng","Tên hàng hóa","Tên sản phẩm","Tên SP","Hàng","Hàng hóa","Hàng bán","Mặt hàng","Nội dung","Product","Món","Tên món","TenGoi","Chi tiết hàng"],
    "customer_id": ["Mã khách hàng","Mã KH","ma_kh","Số KH","Customer ID","Mã khách mua","Mã khách lẻ/sỉ","MaKhachHang","Mã KH mua"],
    "customer": ["Khách hàng","Tên khách hàng","Tên khách","Khách","Đối tác","Bên mua","TenKH","Khách hàng (tên)","Khách lẻ/sỉ","Tên KH","ten_kh"],
    "date": ["Ngày bán","Ngày","Ngày tháng","Ngày giao dịch","ngay_ban","Ngày mua","NgayBan","Ngày đặt hàng","Ngày lập","Thời gian","NgayLap"],
    "timestamp": ["Thời gian tạo","ThoiGianLap","Giờ tạo đơn"],
    "status": ["Trạng thái","Trạng thái đơn","Trạng thái đơn hàng","Trạng thái xử lý","trang_thai","Kết quả","Status","Ghi chú TT","Xử lý","Tình trạng","Tình trạng đơn","TinhTrang"],
    "qty": ["Số lượng","so_luong","Số lượng sản phẩm","SL","SL bán","S.L","SoLuong","Slg","Sluong","Số lượng hàng","Số lượng bán"],
    "base_qty": ["Số lượng quy đổi","SLQuyDoi","SL quy đổi","Quy ra ĐV lẻ","Số lượng lẻ","SL lẻ","SL cơ bản","SL quy ra lẻ"],
    "unit": ["Đơn vị tính","ĐVT","DVT","Đơn vị (tính)","DonVi","Unit","ĐV tính","Đ.vị","DonViTinh"],
    "price": ["Đơn giá","Giá bán","Giá sản phẩm","Giá bán lẻ","Giá lẻ","don_gia","Giá đơn vị","Đơn giá bán","Giá 1 đv","Đơn giá (đ)","Price","Giá","GiaDonVi","Giá niêm yết"],
    "line_total": ["Thành tiền","Thanh tien","thanh_tien","Tổng tiền","Tổng","Trị giá","Amount","T.Tiền","Tiền bán","Tiền","Thành tiền (đ)","TriGia"],
    "revenue": ["Tổng tiền hàng","Tien hang","TongTienHang","Tiền hàng trước giảm"],
    "total": ["Tổng sau giảm giá","TongThanhToan","TongTien","Tổng bill","Tổng đơn","Phải thu","Tong tien"],
    "paid": ["Khách đã trả","Da tra","Khách trả","Thanh toán","KhachDaTra","Đã thu"],
    "debt": ["Còn nợ","Chưa trả","Còn lại","Dư nợ","ConPhaiThu"],
    "method": ["Phương thức thanh toán","Phương thức TT","Hình thức thanh toán","Trả bằng","PhuongThucTT"],
    "channel": ["Kênh bán","Bán qua","Loại bán","KenhBanHang","Nguồn đơn"],
    "seller": ["Người bán","MaNhanVien","Mã NV bán"],
    "group": ["Nhóm hàng(3 Cấp)","Nhóm hàng","Nhom hang","Nhóm hàng hóa","Loại sản phẩm","Danh mục","DanhMuc","Phân loại"],
    "customer_group": ["Nhóm khách hàng","Nhóm","LoaiKhachHang","Hạng"],
    "cost": ["Giá vốn","Giá vốn lúc bán","GiaVonLucBan","GiaVonHang","Giá vốn hiện tại"],
    "base_unit": ["Đơn vị cơ bản","Đơn vị gốc","ĐV nhỏ","ĐV gốc","DonViCoSo"],
    "factor": ["Hệ số","Tỉ lệ quy đổi","Hệ số QĐ","Quy đổi","HeSoQuyDoi","Bằng"],
    "large_unit": ["Đơn vị quy đổi","ĐV nhập","ĐV lớn"],
    "purchase_id": ["Mã nhập hàng","Mã đơn nhập","Mã phiếu nhập","MaPhieuNhap"],
    "purchase_date": ["Ngày nhập","NgayNhapKho"],
    "purchase_total": ["Cần trả NCC","TongGiaTriNhap"],
    "supplier": ["Nhà cung cấp","Mã NCC","Mã nhà cung cấp","MaNhaCungCap"],
    "stock": ["Tồn cuối kỳ","Tồn cuối","Tồn thực","Ton kho","TonCuoi"],
    "expiry": ["Hạn sử dụng","HanSuDung","HSD"],
    "lot": ["Lô","Mã lô","MaLo"],
    "return_qty": ["Số lượng trả (ĐVT cơ bản)","Số lượng trả","SL trả (ĐVT cơ bản)","SLTraQuyDoi"],
    "active": ["Hoạt chất","ThanhPhanChinh"],
    "prescription": ["Thuốc kê đơn","Kê đơn","BanTheoDon"],
    "line_no": ["STT","Thứ tự","No.","TT dòng","SoThuTu"],
    "payment_date": ["Thời gian thu","Ngày thu","Thời điểm","NgayGioThu"],
    "amount": ["Số tiền thu","SoTienThu","Giá trị","So tien"],
    "delivery_date": ["Ngày giao hàng","Ngày chở"],
    "delivery_no": ["Lần giao","Lượt giao"],
    "price_date": ["Ngày áp dụng","NgayApDung"],
    "new_price": ["Giá mới","GiaBanLe"],
    "old_price": ["Giá cũ"],
    "table": ["Phòng/Bàn"],
    "area": ["Khu vực"],
    "guests": ["Số khách"],
}
ALIASES['order_id'] += ['HĐ','HD']
ALIASES['order_id'] += ['Mã HĐ','MaHD','Số hóa đơn','SoHoaDon','Hóa đơn','HoaDon','ID hóa đơn','Mã giao dịch']
ALIASES['price'] += ['ĐG']
ALIASES['line_total'] += ['TT']
ALIASES['channel'] += ['Kênh']
ALIASES['seller'] += ['Thợ']
ALIASES['note']=['Ghi chú','GhiChu']
ALIASES['vehicle']=['Xe']
ALIASES['work']=['Làm gì']
LOOKUP = {}
for field, aliases in ALIASES.items():
    for alias in aliases: LOOKUP.setdefault(compact(alias),field)

def table_kind(filename, sheet=""):
    if 'sosuaxe' in compact(filename):return 'repairs'
    key = compact(filename+" "+sheet)
    period_sheet=bool(re.search(r'20\d{2}',sheet) and re.fullmatch(r'[\d\s/_.-]+',sheet))
    # A multi-report workbook's sheet is more specific than its filename.
    if sheet and not period_sheet and compact(sheet) not in {"sheet1","sheet","data"} and not re.search(r"(?:thang|month)\d",compact(sheet)):
        key = compact(sheet)
    if any(k in key for k in ["chitietdonlo","hoadonchitietlo","donhanglo","bantheolo"]): return "lot_sales"
    if any(k in key for k in ["thongtinthuoc","danhsachthuoc","banhangthuoc"]): return "drugs"
    if any(k in key for k in ["lohang"]): return "lots"
    if any(k in key for k in ["donvi","quydoi"]): return "units"
    if any(k in key for k in ["lichsugia","banggia"]): return "prices"
    if any(k in key for k in ["phongban"]): return "tables"
    if any(k in key for k in ["phieugiaohang","sogiaohang"]): return "deliveries"
    if any(k in key for k in ["hanno","congnotheohan"]): return "due_dates"
    if any(k in key for k in ["thanhtoan","soquy","phieuthu"]): return "payments"
    if "trahang" in key: return "returns"
    if any(k in key for k in ["tonkho","thekho"]): return "stock"
    if "chiphi" in key: return "expenses"
    if any(k in key for k in ["chitietnhap","nhaphangchitiet"]): return "purchase_lines"
    if any(k in key for k in ["nhaphang","phieunhap"]): return "purchases"
    if "khachhang" in key: return "customers"
    # Transaction roles must precede broad catalog substrings: the prefix
    # BanHang_HoaDon contains "hanghoa" across its word boundary.
    if any(k in key for k in ["chitietdon","hoadonchitiet","chitiethoadon","donhangchitiet","sochitietbanhang","sobanhang","banhangcsv"]): return "sales"
    if any(k in key for k in ["hoadon","donhang"]): return "orders"
    if any(k in key for k in ["sanpham","hanghoa","danhsachhanghoa","danhmuc"]): return "products"
    if "nhacungcap" in key: return "suppliers"
    if "nhanvien" in key or "danhsachtho" in key: return "staff"
    if "chinhanh" in key: return "branches"
    if key.startswith("banhang") or key.startswith("sotaybanhang"):return "sales"
    if sheet:return table_kind(filename,'')
    return "unknown"

def semantic_field(description,kind):
    d = norm(description)
    if "cot trong" in d: return None
    rules = [
        ("so luong khach tra", "return_qty"), ("so luong da doi", "base_qty"),
        ("so luong ban lay tu lo", "base_qty"), ("so luong nhap don vi co ban", "base_qty"),
        ("so hoa don", "order_id"), ("ngay sua", "date"),
        ("ghi chu", "note"), ("quy doi don vi lon", "conversion_rule"),
        ("so thu tu dong", "line_no"), ("ma phieu nhap", "purchase_id"),
        ("ma khach", "customer_id"), ("ma hang", "product_id"), ("ma don", "order_id"),
        ("ten khach", "customer"), ("ten hang", "product"),
        ("ngay ban", "date"), ("thoi diem dat", "timestamp"), ("ngay nhap", "purchase_date"),
        ("trang thai don", "status"), ("trang thai phieu", "status"),
        ("thanh tien sau giam", "line_total"), ("tong tien hang", "revenue"),
        ("tong tien don", "total"), ("da thanh toan", "paid"), ("con no", "debt"),
        ("hinh thuc thanh toan", "method"), ("kenh ban", "channel"),
        ("gia von", "cost"), ("don gia", "price"), ("nhom hang", "group"),
        ("nhom khach", "customer_group"), ("don vi co ban", "base_unit"),
        ("don vi lon", "large_unit"), ("he so", "factor"), ("don vi tinh", "unit"),
        ("so luong nhap", "qty"), ("so luong", "qty"),
        ("ton kho", "stock"), ("han su dung", "expiry"), ("so lo", "lot"),
        ("hoat chat", "active"), ("thuoc ke don", "prescription"),
        ("ngay giao", "delivery_date"), ("dot giao", "delivery_no"),
        ("ngay bat dau ap dung", "price_date"), ("gia ban ap dung", "new_price"),
        ("gia truoc lan doi", "old_price"), ("gia nhap mot don vi", "price"),
        ("tien nhap cua ca dong", "line_total"), ("tong tien phieu nhap", "purchase_total"),
        ("ma nha cung cap", "supplier"), ("nha cung cap", "supplier"),
        ("thoi diem thu", "payment_date"), ("so tien", "amount"),
    ]
    for prefix,field in rules:
        if d.startswith(prefix): return field
    if d == "ten": return "customer" if kind=="customers" else "product"
    return None

def readme_maps(text):
    out = {}
    for m in re.finditer(r"^## Cột trong ([^\n]+)\n(.*?)(?=^## |\Z)",text,re.M|re.S):
        refs = re.findall(r"`([^`]+)`",m.group(1))
        if not refs: continue
        key=(refs[0],refs[1] if len(refs)>1 else None)
        kind=table_kind(*key) if key[1] else table_kind(key[0])
        out[key]={compact(col):semantic_field(desc,kind) for col,desc in re.findall(r"^- \*\*(.+?)\*\*: (.+)$",m.group(2),re.M)}
    return out

def mapping_for(columns,kind,explicit):
    result={}
    for col in columns:
        field=explicit.get(compact(col)) or LOOKUP.get(compact(col))
        if compact(col)=="tengoi": field="customer" if kind=="customers" else "product"
        if kind in {"products","customers","units","stock"} and compact(col) in {"donvi","dvt","donvitinh"}:
            field="unit" if kind=="units" else "base_unit"
        if kind=='orders' and field=='line_total' and not explicit.get(compact(col)):
            field='total'
        if kind in {"purchases","purchase_lines"}:
            if field=="date": field="purchase_date"
            if compact(col) in {"sophieu"}: field="purchase_id"
            if compact(col) in {"dongianhap","gianhap"}: field="price"
            if compact(col) in {"slnhap","soluongnhap"}: field="qty"
        if kind=="payments":
            if field=="date": field="payment_date"
            if compact(col) in {"tien","giatri","sotienthu"}: field="amount"
        if kind=="lots" and compact(col) in {"slnhapquydoi","soluongdvtcoban","slnhapdvtcoban"}: field="base_qty"
        if kind=="lot_sales" and compact(col)=="soluongdvtcoban": field="base_qty"
        if kind=="stock" and compact(col)=="ngaygiophatsinh": field="date"
        if kind=="expenses" and compact(col) in {"sotien","tien","giatri","thanhtien"}: field="amount"
        if field and field not in result.values(): result[col]=field
    return result

def date_series(series,sheet=""):
    years=re.findall(r"20\d{2}",sheet)
    ym=re.search(r'(20\d{2})[-/_.](0?[1-9]|1[0-2])(?:\D|$)',sheet)
    my=re.search(r'(?:thang\s*|T)?(0?[1-9]|1[0-2])[-/_.](20\d{2})',sheet,re.I)
    word_month=re.search(r'(?:thang|thg|month|t)\s*(0?[1-9]|1[0-2])(?=\D|$)',norm(sheet))
    month=int(ym.group(2)) if ym else int(my.group(1)) if my else int(word_month.group(1)) if word_month else None
    def prepare(x):
        if pd.isna(x): return x
        if years and month and re.fullmatch(r'\d{1,2}(?:\.0+)?',str(x).strip()) and 1<=float(x)<=31:
            try:return pd.Timestamp(int(years[-1]),month,int(float(x)))
            except ValueError:return pd.NaT
        if isinstance(x,(int,float)) and 20000<=x<=80000:
            return pd.Timestamp('1899-12-30')+pd.Timedelta(days=float(x))
        if isinstance(x,(int,float)):return pd.NaT
        if isinstance(x,str):
            x=re.sub(r'(?i)^\s*(?:thứ\s*(?:hai|ba|tư|năm|sáu|bảy|[2-7])|chủ\s*nhật|cn|t[2-7])\s*[,;:-]?\s*','',x.strip())
            x=re.sub(r'(?i)^ngày\s+','',x)
            if re.fullmatch(r'\d{5}(?:\.\d+)?',x.strip()) and 20000<=float(x)<=80000:
                return pd.Timestamp('1899-12-30')+pd.Timedelta(days=float(x))
            x=re.sub(r"(?:ngày\s*)?(\d{1,2})\s*(?:thg|tháng|thang)\s*(\d{1,2})\s*(?:năm|nam)?\s*(\d{4})",r"\1/\2/\3",x,flags=re.I)
        if re.fullmatch(r"\d{1,2}[/.-]\d{1,2}",str(x).strip()):
            return str(x)+"/"+years[-1] if years else pd.NaT
        return x
    return pd.to_datetime(series.map(prepare),format="mixed",dayfirst=True,errors="coerce")

def load_frames(folder):
    """Infer semantic schema from README and headers; never branch on shop id."""
    text=(folder/"README.md").read_text(encoding="utf-8-sig")
    explicit_maps=readme_maps(text)
    frames={}
    audit=[]
    for path in sorted(folder.iterdir()):
        if path.suffix not in {".csv",".xlsx",".sqlite"}: continue
        if path.suffix==".sqlite":
            con=sqlite3.connect("file:"+str(path)+"?mode=ro",uri=True)
            names=[r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            tables={name:pd.read_sql_query('SELECT * FROM "'+name.replace('"','""')+'"',con) for name in names}
            con.close()
        elif path.suffix==".csv":
            tables={"":pd.read_csv(path,dtype=object,encoding="utf-8-sig")}
        else:
            tables=pd.read_excel(path,sheet_name=None,header=None,dtype=object)
        for sheet,raw in tables.items():
            kind=table_kind(path.name,sheet)
            explicit=explicit_maps.get((path.name,sheet),explicit_maps.get((path.name,None),{}))
            if path.suffix==".xlsx":
                scores=[]
                for i in range(min(8,len(raw))):
                    candidates=[str(c).strip() if pd.notna(c) else "" for c in raw.iloc[i].tolist()]
                    scores.append(len(mapping_for(candidates,kind,explicit)))
                header=max(range(len(scores)),key=lambda i:scores[i]) if scores else 0
                columns=[str(c).strip() if pd.notna(c) else f"_empty_{i}" for i,c in enumerate(raw.iloc[header].tolist())]
                raw=raw.iloc[header+1:].copy(); raw.columns=columns
            raw=raw.dropna(how="all")
            m=mapping_for(raw.columns,kind,explicit)
            df=raw[list(m)].rename(columns=m).copy()
            if kind=="unknown" and {"date","line_total","product"}<=set(df): kind="sales"
            if kind=="purchases" and {"product","purchase_date"}<=set(df): kind="purchase_lines"
            if kind=="units" and "factor" not in df and "group" in df: kind="products"
            for col in ["order_id","product_id","customer_id","product","customer","unit","base_unit","large_unit","status","method","channel","group","customer_group","lot","supplier","purchase_id","seller","note","vehicle","work","conversion_rule"]:
                if col in df: df[col]=df[col].map(lambda x:(str(x).strip() or None) if pd.notna(x) else None)
            for col in ["qty","base_qty","price","line_total","revenue","total","paid","debt","factor","stock","return_qty","cost","amount","purchase_total","new_price","old_price"]:
                if col in df: df[col]=number_series(df[col])
            for col in ["date","timestamp","purchase_date","expiry","payment_date","delivery_date","price_date"]:
                if col in df:
                    blank=df[col].isna()|df[col].astype(str).str.strip().eq('')
                    df[col]=date_series(df[col],path.name+' '+sheet)
                    if kind=='repairs' and col=='date':df[col]=df[col].where(~blank,df[col].ffill())
            df["_source"]=path.name+":"+sheet
            if 'date' not in df and 'timestamp' in df:df['date']=df.timestamp.dt.normalize()
            if kind=='repairs':
                if 'vehicle' in df:df=df.loc[df.vehicle.notna() & df.vehicle.astype(str).str.strip().ne('')].copy()
                df['order_id']=[f'_repair:{path.name}:{sheet}:{i}' for i in df.index]
                df['_warranty']=df.get('note',pd.Series('',index=df.index)).map(norm).str.contains('bao hanh')
            frames.setdefault(kind,[]).append(df)
            audit.append({"file":path.name,"sheet":sheet,"kind":kind,"rows":len(df),"mapping":m})
    return {k:pd.concat(v,ignore_index=True,sort=False) for k,v in frames.items()},text,audit

def status_class(x):
    t=norm(x)
    if "huy" in t or "cancel" in t or 'khong giao duoc' in t: return "cancelled"
    if re.search(r'\btra\b',t) or "return" in t or 'hoan hang' in t: return "returned"
    if any(tk in t for tk in ['dang giao','dang van chuyen','dang gui','shipping']): return "shipping"
    return "completed" if t else "unknown"

def first_known(s):
    s=s.dropna()
    s=s[s.astype(str).str.strip()!=""]
    return s.iloc[0] if len(s) else None

@dataclass
class Shop:
    tables: dict
    readme: str
    audit: list
    lines: pd.DataFrame
    orders: pd.DataFrame
    products: pd.DataFrame
    end: pd.Timestamp
    start: pd.Timestamp
    has_orders: bool

def prepare_shop(folder,cache_dir=None):
    folder=Path(folder)
    signature=hashlib.sha256((PARSER_VERSION+"|"+"|".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in sorted(folder.iterdir()))).encode()).hexdigest()
    cache=Path(cache_dir)/(signature+".pkl") if cache_dir else None
    if cache and cache.exists():
        with cache.open("rb") as f: return pickle.load(f)
    tables,text,audit=load_frames(folder)
    lines=tables.get("sales",pd.DataFrame()).copy()
    if 'repairs' in tables:lines=pd.concat([lines,tables['repairs']],ignore_index=True,sort=False)
    headers=tables.get("orders",pd.DataFrame()).copy()
    has_orders=("order_id" in lines or "order_id" in headers)
    for frame in [lines,headers]:
        if "order_id" in frame:
            if '_source' in frame:
                frame['order_id']=frame.groupby('_source',sort=False)['order_id'].ffill()
            else: frame["order_id"]=frame["order_id"].ffill()
            # Forward-fill metadata only within a real order group.
            for col in ["date","status","customer_id","customer","method","channel","seller","timestamp","note"]:
                if col in frame:
                    frame[col]=frame.groupby("order_id",sort=False)[col].ffill()
                    frame[col]=frame.groupby("order_id",sort=False)[col].bfill()
            frame.drop_duplicates(subset=[c for c in frame if c!="_source"],inplace=True)
    if len(headers) and "order_id" in headers:
        headers=headers.groupby("order_id",sort=False,dropna=False).first().reset_index()
        if len(lines) and "order_id" in lines:
            lines=lines.merge(headers,on="order_id",how="left",suffixes=("","_header"),validate="many_to_one")
            for col in list(lines):
                if col.endswith("_header"):
                    base=col[:-7]
                    if base in lines: lines[base]=lines[base].combine_first(lines[col])
                    else: lines[base]=lines[col]
                    lines.drop(columns=col,inplace=True)
    if not len(lines) and len(headers):
        lines=headers.copy()
        if "revenue" in lines: lines["line_total"]=lines["revenue"]
    if not len(lines) or "date" not in lines: raise ValueError("No recognized dated sales table")
    if 'product' in lines:
        labels=lines['product'].map(norm)
        subtotal=labels.str.match(r'^(?:tong (?:ngay|thang|cong)|cong (?:ngay|thang|cuoi)|tong ket)(?: |$)')
        lines=lines.loc[~subtotal].copy()
        labels=lines['product'].map(norm)
        adjustment=labels.str.match(r'^(?:voucher|dung \d+ diem|phi ship|phi giao hang)(?: |$)')
        if 'qty' in lines:adjustment &= lines.qty.isna()
        if adjustment.any():
            tables['adjustments']=lines.loc[adjustment].copy()
            if 'order_id' in lines and 'line_total' in lines and 'total' not in lines:
                totals=lines.groupby('order_id').line_total.sum(min_count=1)
                lines['total']=lines.order_id.map(totals)
            lines=lines.loc[~adjustment].copy()
    if 'status' not in lines:
        def from_note(x):
            t=norm(x)
            if 'huy' in t:return 'cancelled'
            if re.search(r'\b(?:tra hang|doi tra|tra lai|hoan hang|hoan tra)\b',t):return 'returned'
            if any(v in t for v in ['dang giao','dang van chuyen']):return 'shipping'
            return 'completed'
        lines['status']=lines.get('note',pd.Series('',index=lines.index)).map(from_note)
    lines['_status']=lines['status'].map(status_class).replace('unknown','completed')
    returned=set()
    for key in ["returns","lot_sales"]:
        frame=tables.get(key,pd.DataFrame())
        if "order_id" in frame:
            if key=="returns": returned.update(frame["order_id"].dropna())
            elif "return_qty" in frame: returned.update(frame.loc[frame.return_qty.gt(0),"order_id"].dropna())
    if "order_id" in lines:
        if "qty" in lines: returned.update(lines.loc[lines.qty.lt(0),"order_id"].dropna())
        bad=lines.groupby("order_id")["_status"].agg(lambda x:"cancelled" if "cancelled" in set(x) else "returned" if "returned" in set(x) else "shipping" if "shipping" in set(x) else "completed" if "completed" in set(x) else "unknown")
        lines["_status"]=lines.order_id.map(bad)
        lines.loc[lines.order_id.isin(returned),"_status"]="returned"
        positive=lines.loc[lines.qty.ge(0)] if "qty" in lines else lines
        order_dates=positive.groupby("order_id").date.min()
        lines["date"]=lines.order_id.map(order_dates).combine_first(lines.date)
    products=tables.get("products",pd.DataFrame()).copy()
    drugs=tables.get('drugs',pd.DataFrame())
    if 'product_id' in products and 'product_id' in drugs:
        extra=[c for c in ['active','prescription'] if c in drugs and c not in products]
        if extra: products=products.merge(drugs[['product_id',*extra]].drop_duplicates('product_id'),on='product_id',how='left',validate='many_to_one')
    if "product" not in products and "product" in lines:
        products=lines[[c for c in ["product_id","product","unit"] if c in lines]].drop_duplicates()
        if "unit" in products: products=products.rename(columns={"unit":"base_unit"})
    if "product" in products:
        if "product_id" not in products: products["product_id"]=products["product"].map(norm)
        products=products.drop_duplicates("product_id")
        by_name={norm(r["product"]):r["product_id"] for r in products.to_dict("records") if pd.notna(r.get("product"))}
        if "product" in lines and "product_id" not in lines: lines["product_id"]=lines["product"].map(lambda s:by_name.get(norm(s),norm(s)))
        if "product_id" in lines:
            catalog=products.set_index("product_id")
            for col in ["product","group","base_unit","active","prescription"]:
                if col in catalog:
                    mapped=lines.product_id.map(catalog[col])
                    lines[col]=lines[col].combine_first(mapped) if col in lines else mapped
    customers=tables.get('customers',pd.DataFrame())
    if 'customer_id' in lines and 'customer_id' in customers:
        lookup=customers.drop_duplicates('customer_id').set_index('customer_id')
        for col in ['customer','customer_group']:
            if col in lookup:
                mapped=lines.customer_id.map(lookup[col])
                lines[col]=lines[col].combine_first(mapped) if col in lines else mapped
    units={}
    for name,large,factor,base in re.findall(r"^\s*- (.+?): 1 (.+?) = ([\d.]+) (.+?)\s*$",text,re.M):
        units[(norm(name),norm(large))]=(float(factor),base)
    if 'conversion_rule' in products and 'product' in products:
        for r in products.to_dict('records'):
            match=re.search(r'1\s+(.+?)\s*=\s*([\d.,]+)\s+(.+)',str(r.get('conversion_rule') or ''))
            if match:
                factor=number_series(pd.Series([match.group(2)])).iloc[0]
                if pd.notna(factor):units[(norm(r['product']),norm(match.group(1)))]=(float(factor),r.get('base_unit') or match.group(3))
    unit_df=tables.get("units",pd.DataFrame())
    product_names=products.set_index("product_id")["product"].to_dict() if {"product_id","product"}<=set(products) else {}
    for r in unit_df.to_dict("records"):
        name=r.get("product",product_names.get(r.get("product_id"),""))
        large=r.get("large_unit",r.get("unit"))
        if pd.notna(r.get("factor")) and large: units[(norm(name),norm(large))]=(r["factor"],r.get("base_unit"))
    if "qty" in lines and "product" in lines:
        def convert(r):
            factor,base=units.get((norm(r.get("product")),norm(r.get("unit"))),(1,r.get("base_unit",r.get("unit"))))
            return r.get("qty",float("nan"))*factor,base
        conversions=[convert(r) for r in lines.to_dict("records")]
        converted=pd.Series([r[0] for r in conversions],index=lines.index)
        bases=pd.Series([r[1] for r in conversions],index=lines.index)
        lines["base_qty"]=lines.base_qty.combine_first(converted) if "base_qty" in lines else converted
        lines["base_unit"]=lines.base_unit.combine_first(bases) if "base_unit" in lines else bases
    if "product_id" not in lines and "product" in lines: lines["product_id"]=lines["product"].map(norm)
    if "order_id" in lines:
        fields=[c for c in ["date","_status","customer_id","customer","customer_group","method","channel","seller","total","paid","debt","timestamp","_warranty"] if c in lines]
        orders=lines.groupby("order_id",sort=False)[fields].first()
        if "line_total" in lines: orders["revenue"]=lines.groupby("order_id").line_total.sum(min_count=1)
        orders=orders.reset_index()
    else: orders=pd.DataFrame()
    dates=lines.date.dropna()
    if not len(dates): raise ValueError("All sales dates failed to parse")
    ending=re.search(r"Dữ liệu có đến hết (\d{2}/\d{2}/\d{4})",text)
    end=pd.to_datetime(ending.group(1),format="%d/%m/%Y") if ending else dates.max().normalize()
    obj=Shop(tables,text,audit,lines,orders,products,end,dates.min().normalize(),has_orders)
    if cache:
        cache.parent.mkdir(parents=True,exist_ok=True)
        with cache.open("wb") as f: pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
    return obj

class Unsupported(Exception): pass
class Unanswerable(Exception): pass

def month_bounds(year,month):
    return pd.Timestamp(year,month,1),pd.Timestamp(year,month,calendar.monthrange(year,month)[1])

def time_bounds(plan,as_of,calendar_df=None,question=""):
    now=pd.Timestamp(as_of).normalize(); t=plan.get("time") or {"kind":"none"}; kind=t.get("kind","none")
    # Recover explicit quarters even when the small planner emits a month.
    q=question_norm(question)
    quarter=re.search(r'\bquy\s+([1-4])(?:\s+(?:nam\s+)?(20\d{2}))?',q)
    if "tu ngay 1 1" in q and "nam nay" in q: return now.replace(month=1,day=1),now
    if quarter:
        quarter_no=int(quarter.group(1)); year=int(quarter.group(2)) if quarter.group(2) else now.year
        if not quarter.group(2) and (quarter_no-1)*3+1>now.month: year-=1
        begin=pd.Timestamp(year,(quarter_no-1)*3+1,1)
        finish=month_bounds(year,quarter_no*3)[1]
        return begin,finish
    if kind=="month":
        value=t.get('value')
        if not isinstance(value,str) or not re.fullmatch(r'\d{4}-(?:0[1-9]|1[0-2])',value):raise Unsupported('Month must be YYYY-MM')
        return month_bounds(*map(int,value.split('-')))
    if kind in {"day","until"}:
        d=pd.Timestamp(t.get("date",t.get("value"))).normalize()
        return (pd.Timestamp("1900-01-01"),d) if kind=="until" else (d,d)
    if kind=="range": return pd.Timestamp(t["start"]),pd.Timestamp(t["end"])
    if kind=="today": return now,now
    if kind=="yesterday": return now-pd.Timedelta(days=1),now-pd.Timedelta(days=1)
    if kind=="this_week": return now-pd.Timedelta(days=now.weekday()),now
    if kind=="last_week":
        end=now-pd.Timedelta(days=now.weekday()+1); return end-pd.Timedelta(days=6),end
    if kind in {"this_month","month_to_date"}: return now.replace(day=1),now
    if kind=="last_month":
        end=now.replace(day=1)-pd.Timedelta(days=1); return end.replace(day=1),end
    if kind in {"year_to_date","this_year"}: return now.replace(month=1,day=1),now
    if kind=="last_year": return pd.Timestamp(now.year-1,1,1),pd.Timestamp(now.year-1,12,31)
    if kind=="rolling_days": return now-pd.Timedelta(days=int(t["n"])-1),now
    if kind=="none":
        q=question_norm(question)
        if "hom nay" in q: return now,now
        if "hom qua" in q: return now-pd.Timedelta(days=1),now-pd.Timedelta(days=1)
        raise Unsupported("No verified time interval")
    raise Unsupported("Unsupported time expression: "+kind)

def period(frame,start,end,col="date"):
    if col not in frame: raise Unanswerable("Required date field absent")
    return frame.loc[frame[col].ge(start)&frame[col].lt(end+pd.Timedelta(days=1))].copy()

def valid(frame):
    return frame.loc[frame["_status"].isin(["completed","shipping"])].copy()

def product_entity(shop,question):
    q=question_norm(question)
    pairs=[]
    for row in shop.products.to_dict("records"):
        name=row.get("product")
        if name and norm(name) and norm(name) in q: pairs.append((len(norm(name)),str(row.get("product_id")),name))
    if not pairs and "product" in shop.lines:
        for row in shop.lines[["product_id","product"]].drop_duplicates().to_dict("records"):
            if row["product"] and norm(row["product"]) in q: pairs.append((len(norm(row["product"])),str(row["product_id"]),row["product"]))
    if not pairs: return None
    pairs.sort(key=lambda r:(-r[0],r[1]))
    return pairs[0][1]

def sorted_scores(frame,key,value):
    scores=frame.groupby(key,dropna=True)[value].sum(min_count=1).dropna().reset_index()
    scores["_tie"]=scores[key].astype(str)
    return scores.sort_values([value,"_tie"],ascending=[False,True],kind="stable")

def require_columns(frame,*columns):
    missing=set(columns)-set(frame)
    # A mapper may have missed a field. This alone does not prove that the
    # published data are insufficient, so abstain rather than claim unanswerable.
    if missing: raise Unsupported("Unverified required fields: "+", ".join(sorted(missing)))

def _contains(text,phrase):
    return bool(phrase) and re.search(r"\b"+re.escape(phrase)+r"\b",text) is not None

def filter_dimensions(frame,q):
    """Only apply dimensions explicitly named in the question."""
    result=frame;matched=False
    specs=[
        ('method',[('tien mat',['tien mat','cash']),('chuyen khoan',['chuyen khoan','transfer']),('vi dien tu',['vi dien tu','ewallet']),('cod',['cod','thu ho']),('ghi no',['ghi no','debt','mua chiu'])]),
        ('channel',[('dien thoai',['dien thoai','phone']),('zalo',['zalo']),('tai quay',['tai quay','instore','in store','store','cua hang']),('shopee',['shopee']),('lazada',['lazada']),('tiktok',['tiktok'])]),
        ('customer_group',[('than thiet',['than thiet','loyal']),('khach si',['si','wholesale']),('khach le',['le','retail'])]),
    ]
    for field,categories in specs:
        selected=[aliases for phrase,aliases in categories if _contains(q,phrase)]
        if len(selected)>1: raise Unsupported('Multiple dimension values need a composite executor')
        if selected:
            require_columns(result,field)
            values=normalized_series(result[field])
            mask=values.map(lambda x:any(_contains(x,a) for a in selected[0]))
            result=result.loc[mask];matched=True
    if 'nhom' in q and 'khach' not in q:
        require_columns(result,'group')
        groups=[g for g in result['group'].dropna().unique() if _contains(q,norm(g))]
        if len(groups)!=1: raise Unsupported('Unresolved product group')
        result=result.loc[result['group'].eq(groups[0])];matched=True
    if any(t in q for t in ['nhom khach','khach nhom']) and not matched:
        require_columns(result,'customer_group')
        groups=[g for g in result.customer_group.dropna().unique() if _contains(q,norm(g))]
        if len(groups)!=1: raise Unsupported('Unresolved customer group')
        result=result.loc[result.customer_group.eq(groups[0])];matched=True
    if 'hoat chat' in q:
        require_columns(result,'active')
        values={norm(v) for v in result.active.dropna().unique() if norm(v)}
        found=sorted([v for v in values if _contains(q,v)],key=lambda v:-len(v))
        if not found or len(found)>1 and len(found[0])==len(found[1]):raise Unsupported('Unresolved exact active ingredient')
        result=result.loc[normalized_series(result.active).eq(found[0])];matched=True
    if any(t in q for t in ['ke don','don bac si']):
        require_columns(result,'prescription')
        values=normalized_series(result.prescription)
        yes={'1','true','co','yes','rx','ke don'};no={'0','false','khong','no','otc','khong ke don'}
        if not set(values.unique()) <= yes|no:raise Unsupported('Unknown prescription flag')
        result=result.loc[values.isin(no if 'khong' in q else yes)];matched=True
    size=re.search(r'\bsize\s+([sml])\b',q)
    if size:
        require_columns(result,'product')
        result=result.loc[result['product'].astype(str).str.contains(r'\('+size.group(1)+r'\)\s*$',case=False,regex=True)];matched=True
    if 'ca sang' in q or 'ca chieu' in q:
        if 'ca sang' in q and 'ca chieu' in q: raise Unsupported('Shift comparison requires comparison executor')
        require_columns(result,'timestamp')
        hour=pd.to_datetime(result.timestamp,errors='coerce').dt.hour
        if hour.isna().any(): raise Unsupported('Unparsed order creation time')
        result=result.loc[hour.between(6,13) if 'ca sang' in q else hour.between(14,21)];matched=True
    return result,matched

def normalized_series(series):
    # Business categorical columns repeat values thousands of times.
    return series.map({v:norm(v) for v in series.dropna().unique()}).fillna('')

def question_months(plan,q):
    months=list((plan.get('time') or {}).get('values') or [])
    if len(months)==2 and re.search(r'\btu\s+(?:thang\s+)?\d',q) and re.search(r'\bden\s+(?:het\s+)?thang\b',q):
        months=[str(m) for m in pd.period_range(months[0],months[-1],freq='M')]
    return months

def numeric_threshold(question):
    raw=unicodedata.normalize('NFC',question).lower()
    m=re.search(r'(đạt|dat|từ|tu|trên|tren|dưới|duoi|ít nhất|it nhat|tối thiểu|toi thieu)\s+([\d.,]+)\s*(?:đồng|dong|vnd|đ\b)',raw)
    if not m:return None
    value=number_series(pd.Series([m.group(2)])).iloc[0]
    if pd.isna(value):raise Unsupported('Invalid amount threshold')
    op=norm(m.group(1))
    return value,op

def threshold_mask(values,threshold):
    value,op=threshold
    return values.lt(value) if op=='duoi' else values.gt(value) if op=='tren' else values.ge(value)

def debt_answer(shop,q,end,typ,k):
    if any(s in q for s in ['no them','thu no','tra no','tra bot no','thu hoi']): raise Unsupported('Debt flow is distinct from closing debt')
    require_columns(shop.orders,'date','order_id','_status')
    orders=shop.orders.loc[shop.orders.date.lt(end+pd.Timedelta(days=1))&shop.orders._status.ne('cancelled')].copy()
    if 'method' not in orders: raise Unsupported('Cannot exclude COD from debt without payment method')
    excluded=orders.method.map(norm).str.contains(r'\bcod\b|giao hang thu tien',regex=True)
    if 'channel' in orders:
        excluded|=orders.channel.map(norm).str.contains(r'shopee|lazada|tiki|tiktok|san tmdt|thuong mai dien tu',regex=True)
    orders=orders.loc[~excluded]
    payments=shop.tables.get('payments',pd.DataFrame())
    if {'order_id','payment_date','amount'}<=set(payments) and 'total' in orders:
        paid=payments.loc[payments.payment_date.lt(end+pd.Timedelta(days=1))].groupby('order_id').amount.sum()
        orders['_debt']=(orders.total-orders.order_id.map(paid).fillna(0)).clip(lower=0)
    elif 'debt' in orders and end.normalize()==shop.end:
        orders['_debt']=orders.debt.clip(lower=0)
    else: raise Unsupported('Historical debt needs dated payments and order totals')
    orders=orders.loc[orders._debt.gt(0)]
    if typ=='list' and 'tuoi no' in q and '0 7' in q and '8 15' in q:
        days=(end-orders.date.dt.normalize()).dt.days
        return [money(orders.loc[mask,'_debt'].sum()) for mask in [days.between(0,7),days.between(8,15),days.gt(15)]]
    age=re.search(r'\b(tren|qua|hon|duoi|tu)\s+(\d+)\s+ngay',q)
    if age:
        days=(end-orders.date.dt.normalize()).dt.days; n=int(age.group(2))
        mask=days.gt(n) if age.group(1) in {'tren','qua','hon'} else days.lt(n) if age.group(1)=='duoi' else days.ge(n)
        orders=orders.loc[mask]
    if typ=='number' and re.search(r'\b(?:so|bao nhieu|may) don\b',q): return len(orders)
    customer_col='customer_id' if 'customer_id' in orders else 'customer' if 'customer' in orders else None
    if typ=='number' and not re.search(r'\b(?:so|bao nhieu|may) khach\b',q): return money(orders._debt.sum())
    if not customer_col: raise Unsupported('No verified customer key')
    known=orders[customer_col].notna()&orders[customer_col].map(norm).ne('khach le')
    customer_debt=sorted_scores(orders.loc[known],customer_col,'_debt')
    if typ=='number': return len(customer_debt)
    if 'customer' in orders:
        names=orders.groupby(customer_col)['customer'].first().to_dict()
    else: names={}
    items=[str(names.get(r[customer_col]) or r[customer_col]) for r in customer_debt.to_dict('records')]
    if typ=='string':
        if not items: raise Unsupported('No debtor')
        return items[0]
    if typ=='list': return items[:int(k or len(items))]
    if typ=='table':
        limit=int(k or len(items))
        return {'columns':['khach_hang','so_tien_no'],'rows':[[name,money(value)] for name,value in zip(items[:limit],customer_debt['_debt'].iloc[:limit])]} 
    raise Unsupported('Debt output type')

def answer_question(shop,question,as_of,plan,calendar_df=None):
    """Return a submission record body. Abstain on unsupported semantics."""
    try:
        value=execute(shop,question,as_of,plan,calendar_df)
        typ=plan.get("answer_type")
        if typ=='unanswerable': raise Unsupported('A model label alone does not establish missing data')
        if typ=="number":
            if isinstance(value,bool) or not isinstance(value,(int,float)): raise Unsupported("numeric format")
            if not math.isfinite(float(value)): raise Unsupported("nonfinite")
        elif typ=="percent": value=rounded(value)
        elif typ=="date":
            if isinstance(value,pd.Timestamp): value=value.strftime("%Y-%m-%d")
            if not isinstance(value,str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}(?:-[0-9]{2})?',value):raise Unsupported('date format')
            pd.Timestamp(value if len(value)==10 else value+'-01')
        elif typ=="list" and not isinstance(value,list): raise Unsupported("list format")
        elif typ=="string" and not isinstance(value,str): raise Unsupported("string format")
        elif typ=="boolean" and not isinstance(value,bool): raise Unsupported("boolean format")
        elif typ=="table" and not isinstance(value,dict): raise Unsupported("table format")
        if hasattr(value,"item"): value=value.item()
        return {"type":typ,"answer":value}
    except Unanswerable as e:
        return {"type":"unanswerable","reason":str(e)}
    except (Unsupported,KeyError,ValueError,TypeError,IndexError,ZeroDivisionError) as e:
        return {"type":"abstain","reason":str(e)}

def execute(shop,question,as_of,plan,calendar_df=None):
    q=question_norm(question); intent=plan["intent"]; typ=plan["answer_type"]
    if re.search(r'\bkhach (?:hang )?o (?:duong|phuong|quan)\b',q):raise Unsupported('Location filter has no verified executor')
    if intent=='debt_ranking' and (plan.get('time') or {}).get('kind')=='none':
        plan={**plan,'time':{'kind':'until','date':str(pd.Timestamp(as_of).date())}}
    # Resolve explicit metric words against units in this shop's catalog.
    units={norm(u) for u in shop.lines.get('base_unit',pd.Series(dtype=str)).dropna().unique()}
    units|={'ly','lon','chai','vien','hop','goi','cai','bao','kg','mieng','cuon','cuc','binh','cay','dia','tuyp'}
    quantity=any(re.search(r'\b(?:bao nhieu|may|so|tong so)\s+'+re.escape(u)+r'\b',q) for u in units if u)
    if intent in {'revenue_total','other'} and quantity and 'ban' in q: intent='quantity_sold'
    if intent in {'revenue_total','order_count','other'} and re.search(r'\b(?:binh quan|trung binh)\b',q) and re.search(r'\b(?:don|bill)\b',q): intent='avg_order_value'
    if intent in {'revenue_total','other'} and re.search(r'\bnhap\b',q) and 'tien' in q and not any(t in q for t in ['gia nhap','binh quan','nhap tru','so voi']):intent='purchase_total'
    k=plan.get('k')
    km=re.search(r'\b(\d+)\s+khach\b',q)
    if intent=='debt_ranking' and k is None and km:k=int(km.group(1))
    if (plan.get('time') or {}).get('kind')=='months' and typ=='list' and intent in {'revenue_total','order_count','quantity_sold','avg_order_value','purchase_total'}:
        months=question_months(plan,q)
        return [execute(shop,question,as_of,{**plan,'time':{'kind':'month','value':m},'answer_type':'number'},calendar_df) for m in months]
    if (plan.get('time') or {}).get('kind')=='months' and intent in {'growth_vs_period','argmax_date'}:
        months=question_months(plan,q)
        if not months:raise Unsupported('Missing months')
        values={m:execute(shop,'doanh thu',as_of,{'intent':'revenue_total','answer_type':'number','time':{'kind':'month','value':m},'k':None},calendar_df) for m in months}
        if intent=='argmax_date':return min(months,key=lambda m:(-values[m],m))
        explicit=re.findall(r'\bthang (\d{1,2})(?: nam)? (20\d{2})\b',q)
        ordered=[f'{int(y):04d}-{int(m):02d}' for m,y in explicit]
        if len(ordered)!=2 or set(ordered)!=set(months):raise Unsupported('Ambiguous month comparison order')
        current,prior=[values[m] for m in ordered]
        if typ=='percent':
            if prior==0:raise Unsupported('Zero comparison denominator')
            return rounded(100*(current-prior)/prior)
        return money(current-prior)
    if typ=='list' and intent in {'channel_breakdown','revenue_total'} and (plan.get('time') or {}).get('kind')!='months':
        dimensions=[['tien mat','chuyen khoan','vi dien tu','cod','ghi no'],['dien thoai','zalo','tai quay']]
        for phrases in dimensions:
            found=[(q.find(phrase),phrase) for phrase in phrases if _contains(q,phrase)]
            if len(found)>=2:
                return [execute(shop,'doanh thu '+phrase,as_of,{**plan,'intent':'revenue_total','answer_type':'number'},calendar_df) for _,phrase in sorted(found)]
    if intent=='monotonic_check':
        months=(plan.get('time') or {}).get('values')
        if not months or len(months)<2: raise Unsupported('Missing ordered months')
        values=[]
        for month in months:
            sub={'intent':'revenue_total','answer_type':'number','time':{'kind':'month','value':month},'k':None}
            values.append(execute(shop,'doanh thu',as_of,sub,calendar_df))
        return all(b>a for a,b in zip(values,values[1:]))
    start,end=time_bounds(plan,as_of,calendar_df,question)
    # Explicit periods may extend beyond as_of; relative periods are already clipped.
    if end<shop.start or start>shop.end: raise Unanswerable("Period outside published coverage")
    if intent=='debt_ranking':
        if any(s in q for s in ['no them','thu no','tra no','tra bot no','thu hoi']):
            orders=shop.orders.copy();payments=shop.tables.get('payments',pd.DataFrame())
            require_columns(orders,'order_id','date','method','_status','total')
            require_columns(payments,'order_id','payment_date','amount')
            orders=orders.loc[orders._status.ne('cancelled') & orders.method.map(norm).str.contains(r'ghi no|debt|mua chiu',regex=True)]
            if 'channel' in orders:orders=orders.loc[~orders.channel.map(norm).str.contains(r'shopee|lazada|tiki|tiktok|san tmdt',regex=True)]
            joined=payments.merge(orders[['order_id','date']],on='order_id',how='inner',validate='many_to_one')
            if 'no them' in q:
                selected=period(orders,start,end)
                same=joined.loc[joined.payment_date.dt.normalize().eq(joined.date.dt.normalize())].groupby('order_id').amount.sum()
                return money((selected.total-selected.order_id.map(same).fillna(0)).clip(lower=0).sum())
            paid=period(joined,start,end,'payment_date')
            paid=paid.loc[paid.payment_date.dt.normalize().gt(paid.date.dt.normalize()) & paid.amount.gt(0)]
            return money(paid.amount.sum())
        return debt_answer(shop,q,end,typ,k)
    if intent=='growth_vs_period':
        if any(s in q for s in ['nhom','khach','san pham','mat hang','so luong']): raise Unsupported('Non-global growth')
        current={'intent':'revenue_total','answer_type':'number','time':{'kind':'range','start':str(start.date()),'end':str(end.date())},'k':None}
        kind=(plan.get('time') or {}).get('kind')
        if 'ca chieu' in q and 'ca sang' in q:
            frame=valid(period(shop.orders,start,end));require_columns(frame,'timestamp','revenue')
            hours=pd.to_datetime(frame.timestamp,errors='coerce').dt.hour
            if hours.isna().any():raise Unsupported('Unparsed order creation time')
            morning=frame.loc[hours.between(6,13)];afternoon=frame.loc[hours.between(14,21)]
            current_value=len(afternoon) if 'so don' in q else afternoon.revenue.sum()
            prior_value=len(morning) if 'so don' in q else morning.revenue.sum()
            if typ=='percent':
                if prior_value==0:raise Unsupported('Zero comparison denominator')
                return rounded(100*(current_value-prior_value)/prior_value)
            return money(current_value-prior_value)
        if 'nam truoc' in q or 'nam ngoai' in q:
            before_start=start-pd.DateOffset(years=1); before_end=end-pd.DateOffset(years=1)
        elif 'quy' in q:
            before_start=start-pd.DateOffset(months=3); before_end=start-pd.Timedelta(days=1)
        elif kind in {'month','this_month','month_to_date','last_month'}:
            before_start=start-pd.DateOffset(months=1)
            before_end=before_start.replace(day=min(end.day,calendar.monthrange(before_start.year,before_start.month)[1]))
            if kind in {'month','last_month'}: before_end=month_bounds(before_start.year,before_start.month)[1]
        elif kind in {'this_week','last_week','rolling_days'}:
            length=(end-start).days+1; before_end=start-pd.Timedelta(days=1); before_start=before_end-pd.Timedelta(days=length-1)
        else: raise Unsupported('Ambiguous comparison interval')
        prior={**current,'time':{'kind':'range','start':str(before_start.date()),'end':str(before_end.date())}}
        current_value=execute(shop,'doanh thu',as_of,current,calendar_df)
        prior_value=execute(shop,'doanh thu',as_of,prior,calendar_df)
        if typ=='percent':
            if prior_value==0: raise Unsupported('Zero comparison denominator')
            return rounded(100*(current_value-prior_value)/prior_value)
        return money(current_value-prior_value)
    rows=period(shop.lines,start,end); sales=valid(rows)
    orders=period(shop.orders,start,end) if len(shop.orders) else pd.DataFrame()
    good_orders=valid(orders) if len(orders) else orders
    all_sales=sales
    sales,dimension_matched=filter_dimensions(sales,q)
    rows,_=filter_dimensions(rows,q)
    if len(good_orders) and (intent=='avg_order_value' or intent=='order_count' and numeric_threshold(question) is not None):
        good_orders,_=filter_dimensions(good_orders,q)
    entity=product_entity(shop,question)
    if entity is not None:
        sales=sales.loc[sales.product_id.astype(str).eq(entity)]
        rows=rows.loc[rows.product_id.astype(str).eq(entity)]
    # Explicit unsupported business filters must never silently disappear.
    complex_terms=["lien tuc","phong lanh","nhan vien","san tmdt"]
    if any(term in q for term in complex_terms): raise Unsupported("Requires additional verified business filter")
    if intent in {"revenue_total","order_count","avg_order_value","quantity_sold"}:
        if any(t in q for t in ["gia von","lai rong","lai gop","chenh","nhap tru","so voi","nhieu hon","it hon"]):
            raise Unsupported("Composite metric requires a specific executor")
        if not dimension_matched and any(t in q for t in ["nhom ","khach hang ","khach si","khach le","tien mat","chuyen khoan","ghi no","cod","qua zalo","qua shopee","tai quay","zalo oa"]):
            raise Unsupported("Requires verified dimension filter")
        if re.search(r"(?:tren|duoi|tu) \d",q) and intent=="order_count" and numeric_threshold(question) is None: raise Unsupported("Unparsed order-value threshold")
    if intent=="revenue_total" and re.search(r"\b(?:bao nhieu ngay|so ngay)\b",q):
        threshold=numeric_threshold(question)
        if threshold is None:raise Unsupported("Missing day revenue threshold")
        return int(threshold_mask(sales.groupby(sales.date.dt.normalize()).line_total.sum(),threshold).sum())
    if intent=="revenue_total":
        require_columns(sales,"line_total")
        if sales.line_total.isna().any(): raise Unsupported('Unparsed or missing sale amounts')
        return money(sales.line_total.sum())
    if intent=="quantity_sold":
        require_columns(sales,"base_qty")
        if sales.base_qty.isna().any(): raise Unsupported('Unparsed or missing quantities')
        if entity is None:
            units={norm(u) for u in sales.get("base_unit",pd.Series(dtype=str)).dropna().unique()}
            asked=[u for u in units if re.search(r"\b"+re.escape(u)+r"\b",q)]
            if len(asked)==1: sales=sales.loc[sales.base_unit.map(norm).eq(asked[0])]
            elif len(units)>1: raise Unsupported("Ambiguous quantity unit/entity")
        return rounded(sales.base_qty.sum())
    if intent=='distinct_orders_per_item':
        if not shop.has_orders:raise Unanswerable('No order identifiers available')
        if entity is None:raise Unsupported('Unresolved product for distinct order count')
        return int(sales.order_id.nunique())
    if intent in {"order_count","cancelled_orders","avg_order_value"}:
        if not shop.has_orders: raise Unanswerable("No order identifiers available")
        if intent=="cancelled_orders": return int(orders.loc[orders._status.eq("cancelled"),"order_id"].nunique())
        if intent=="avg_order_value":
            if '_warranty' in good_orders:good_orders=good_orders.loc[~good_orders['_warranty'].fillna(False)]
            require_columns(good_orders,"revenue")
            if not len(good_orders): raise Unsupported("Empty mean")
            return money(good_orders.revenue.sum()/len(good_orders))
        threshold=numeric_threshold(question)
        if threshold is not None:
            require_columns(good_orders,"revenue")
            return int(threshold_mask(good_orders.revenue,threshold).sum())
        return int(sales.order_id.nunique())
    if intent=="share_ratio" and any(t in q for t in ["huy don","don huy","bi huy"]):
        if not shop.has_orders: raise Unanswerable("No order identifiers")
        if not len(orders): raise Unsupported("Zero denominator")
        return rounded(100*orders._status.eq("cancelled").sum()/len(orders))
    if intent=='share_ratio':
        if not any(s in q for s in ['doanh thu','tien','ti trong','ty trong','chiem']): raise Unsupported('Unverified ratio metric')
        all_sales=valid(period(shop.lines,start,end)); require_columns(all_sales,'line_total')
        denominator=all_sales.line_total.sum()
        if denominator==0: raise Unsupported('Zero share denominator')
        numerator=None
        if dimension_matched or entity is not None: numerator=sales.line_total.sum()
        elif 'group' in sales:
            groups=[g for g in sales['group'].dropna().unique() if norm(g) and norm(g) in q]
            if len(groups)==1: numerator=sales.loc[sales['group'].eq(groups[0]),'line_total'].sum()
        if numerator is None: raise Unsupported('Unresolved share dimension')
        return rounded(100*numerator/denominator)
    if intent in {"argmax_entity","topk_list"}:
        if 'khach' in q and not any(t in q for t in ['nhan vien','nha cung cap']):
            key='customer_id' if 'customer_id' in sales else 'customer'
            require_columns(sales,key,'customer','line_total','order_id')
            frame=sales.loc[sales[key].notna() & ~normalized_series(sales[key]).isin(['','khach le','khach vang lai','vang lai'])].copy()
            if frame.empty:raise Unsupported('No verified customer')
            if key=='customer':frame['_customer_key']=normalized_series(frame[key]).str.replace(' ','',regex=False);key='_customer_key'
            metric=frame.groupby(key).order_id.nunique() if any(t in q for t in ['nhieu don','nhieu lan']) else frame.groupby(key).line_total.sum()
            names=frame.groupby(key).customer.first().to_dict()
            ordered=sorted(metric.index,key=lambda x:(-metric[x],str(x)))
            if not ordered:raise Unsupported('No verified customer')
            items=[str(names[x]) for x in ordered]
            if typ=='table':return {'columns':['khach_hang','doanh_thu'],'rows':[[str(names[x]),money(metric[x])] for x in ordered[:int(k or len(ordered))]]}
            return items[:int(k or 1)] if typ=='list' else items[0]
        if any(t in q for t in ["nhan vien","nguoi ban","nha cung cap","loi","lai"]): raise Unsupported("Non-product ranking")
        require_columns(sales,"product_id","base_qty","product")
        sales=sales.loc[sales["product"].map(norm).ne("phi phuc vu")]
        scores=sorted_scores(sales,"product_id","base_qty")
        names=sales.groupby("product_id")["product"].first().to_dict()
        if not len(scores): raise Unsupported("No ranked product")
        items=[str(names[i]) for i in scores.product_id]
        return items[:int(plan.get("k") or 1)] if typ=="list" else items[0]
    if intent=="argmax_date":
        require_columns(sales,"line_total")
        frequency="M" if typ=="date" and ("thang nao" in q or "thang co" in q) else "D"
        sales=sales.copy(); sales["_day"]=sales.date.dt.to_period(frequency).astype(str)
        scores=sorted_scores(sales,"_day","line_total")
        if not len(scores): raise Unsupported("No ranked day")
        return str(scores.iloc[0]["_day"])
    if intent=="unit_price_lookup" and "nhap" not in q:
        require_columns(rows,"price","qty","base_qty")
        if entity is None: raise Unsupported("Unresolved product")
        prices=(rows.price*rows.qty/rows.base_qty.replace(0,float("nan"))).dropna().unique()
        if len(prices)!=1: raise Unsupported("Multiple/missing recorded prices")
        return money(prices[0])
    if intent=="unsold_items":
        require_columns(shop.lines,"product_id","product")
        previous=valid(shop.lines.loc[shop.lines.date.lt(start)])
        old=set(previous.loc[previous["product"].map(norm).ne("phi phuc vu"),"product_id"])
        missing=old-set(sales.product_id)
        if typ=="number": return len(missing)
        names=shop.lines.groupby("product_id")["product"].first().to_dict()
        return [str(names[k]) for k in sorted(missing,key=str)]
    if intent=="stock_level":
        # Snapshot stock is only valid at its documented snapshot date.
        if end.normalize()!=shop.end: raise Unsupported("Historical stock needs movement reconstruction")
        stock=shop.tables.get("stock",pd.DataFrame()).copy()
        require_columns(stock,"stock")
        if entity is not None:
            if "product_id" in stock: stock=stock.loc[stock.product_id.astype(str).eq(entity)]
            elif "product" in stock:
                names=shop.lines.loc[shop.lines.product_id.astype(str).eq(entity),"product"].map(norm).unique()
                stock=stock.loc[stock["product"].map(norm).isin(names)]
        else: raise Unsupported("Stock needs resolved product or valuation executor")
        if typ!="number": raise Unsupported("Stock unit decomposition")
        if not len(stock): raise Unsupported("Product absent from snapshot")
        return rounded(stock.stock.sum())
    if intent=="gross_profit":
        if any(t in q for t in ["ton","trong kho","lai rong"]): raise Unsupported("Inventory valuation/net profit")
        require_columns(sales,"line_total","cost","qty")
        if sales.cost.isna().any(): raise Unanswerable("Missing recorded sale-time cost")
        return money(sales.line_total.sum()-(sales.cost*sales.qty).sum())
    if intent=="purchase_total":
        purchases=shop.tables.get("purchases",pd.DataFrame())
        require_columns(purchases,"purchase_date","purchase_total")
        p=period(purchases,start,end,"purchase_date")
        if "status" in p: p=p.loc[p.status.map(status_class).ne("cancelled")]
        if "purchase_id" in p: p=p.drop_duplicates("purchase_id")
        return money(p.purchase_total.sum())
    raise Unsupported("Executor not implemented: "+intent)
