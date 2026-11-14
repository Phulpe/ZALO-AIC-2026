"""Check whether projection loses raw-row distinctions; fixed aggregate output."""
exec((__import__('pathlib').Path(__file__).parent/'audit_source_identity.py').read_text(encoding='utf-8').split('for business in sorted(businesses):')[0])
import pandas as pd,openpyxl,hashlib,sqlite3
from legacy_engine import compact,norm
from relearning import sha
manifest=json.loads((runtime/'DATA_MANIFEST.json').read_text(encoding='utf-8'));counts=collections.Counter();counts['businesses']=len(businesses)
for business in sorted(businesses):
 shop=shops[business];books={}
 try:
  for audit in shop.audit:
   if audit['kind']!='sales':continue
   key='data/'+business+'/'+audit['file'];path=runtime/key;meta=manifest['business_source_files'][key];assert sha(path)==meta['sha256']
   if path.suffix=='.sqlite':
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:raw=pd.read_sql_query('SELECT * FROM "'+str(audit['sheet']).replace('"','""')+'"',db)
    labels=list(raw)
   elif path.suffix=='.xlsx':
    if path not in books:books[path]=openpyxl.load_workbook(path,read_only=True,data_only=False)
    data=pd.DataFrame(books[path][audit['sheet']].values);headers=[]
    for i in range(min(8,len(data))):
     labels=[str(v).strip() if pd.notna(v) else '' for v in data.iloc[i]]
     if set(audit['mapping'])<=set(labels):headers.append((i,labels))
    if len(headers)!=1:counts['ambiguous_headers']+=1;continue
    header,labels=headers[0];raw=data.iloc[header+1:].copy();raw.columns=labels
   elif path.suffix=='.csv':raw=pd.read_csv(path,dtype=object,encoding='utf-8-sig');labels=list(raw)
   else:counts['unsupported_source_types']+=1;continue
   raw=raw.dropna(how='all');mapped=raw[list(audit['mapping'])]
   counts['raw_rows']+=len(raw);counts['raw_exact_duplicates']+=int(raw.duplicated().sum());counts['projected_exact_duplicates_before_numeric_parsing']+=int(mapped.duplicated().sum());counts['projection_loses_row_distinctions']+=int((mapped.duplicated()&~raw.duplicated()).sum())
   for name in set(labels)-set(audit['mapping']):
    token=compact(name)
    family='row_identity' if token in {'lineid','rowid','madong','id','idchitiet','chitietid','detailid'} else 'sequence' if token in {'stt','sothutu','sodong'} else 'other'
    counts['unmapped_'+family+'_columns']+=1
 finally:
  for b in books.values():b.close()
text=(runtime/'rulebook.md').read_text(encoding='utf-8');facts=[]
for line in text.splitlines():
 n=norm(line)
 if 'tinh mot lan' in n:
  facts.append({'line_sha256':hashlib.sha256(line.encode()).hexdigest(),'identical_complete_row':any(t in n for t in ['hoan toan','100','cac truong','toan bo','ca dong']),'identical_word':'giong nhau' in n,'count_once':True,'negation':'khong' in n})
atomic(exp/'RAW_PROJECTION_ALL_SOURCES_AGGREGATE.json',{'counts':dict(counts),'rule_facts':facts,'source_text_not_exposed':True});print(json.dumps({'counts':dict(counts),'rule_facts':facts},indent=2))
