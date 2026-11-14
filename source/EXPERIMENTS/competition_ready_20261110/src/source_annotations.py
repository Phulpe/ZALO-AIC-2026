"""Prove structural notes against original source bytes, never against gold."""
from pathlib import Path
import json,sqlite3,hashlib
import pandas as pd
from relearning import sha
from learning_executor import NOTE,empty,annotation_evidence

def original_notes(root,business,shop,manifest,allowed_businesses):
    root=Path(root).resolve()
    if business not in allowed_businesses:raise ValueError('Source proof outside learning split')
    approved=set();evidence=[];read={}
    for audit in shop.audit:
        if audit['kind']!='sales':continue
        name='data/'+business+'/'+audit['file'];path=(root/name).resolve()
        if not path.is_relative_to(root) or path.is_symlink():raise ValueError('Unsafe source path')
        meta=manifest['business_source_files'][name]
        if path.stat().st_size!=meta['bytes'] or sha(path)!=meta['sha256']:raise ValueError('Original source hash mismatch')
        mapping=audit['mapping'];values=list(mapping.values())
        if any(values.count(k)>1 for k in ['product','date','order_id','qty','price','line_total','unit']):continue
        if not {'product','date','qty','price','line_total'}<=set(values):continue
        suffix=path.suffix.lower();sheet=audit['sheet']
        if suffix=='.xlsx':
            if path not in read:
                import openpyxl
                book=openpyxl.load_workbook(path,read_only=True,data_only=False)
                try:
                    names={a['sheet'] for a in shop.audit if a['kind']=='sales' and a['file']==audit['file']}
                    read[path]={name:pd.DataFrame(book[name].values) for name in names}
                finally:book.close()
            raw=read[path][sheet]
            candidates=[]
            for i in range(min(8,len(raw))):
                labels=[str(c).strip() if not empty(c) else '' for c in raw.iloc[i].tolist()]
                if set(mapping)<=set(labels):candidates.append((i,labels))
            if len(candidates)!=1:continue
            header,columns=candidates[0];raw=raw.iloc[header+1:].copy();raw.columns=columns
        elif suffix=='.csv':raw=pd.read_csv(path,dtype=object,encoding='utf-8-sig')
        elif suffix=='.sqlite':
            with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:raw=pd.read_sql_query('SELECT * FROM "'+str(sheet).replace('"','""')+'"',db)
        else:continue
        if not set(mapping)<=set(raw):continue
        mapped=raw[list(mapping)].rename(columns=mapping)
        labels=mapped['product'].map(lambda value:value.strip() if isinstance(value,str) else value)
        frequencies=labels.value_counts(dropna=False)
        for index,row in mapped.iterrows():
            label=row.get('product')
            if not isinstance(label,str) or not NOTE.match(label):continue
            if frequencies.get(label.strip(),0)!=1:continue
            if not all(empty(row.get(k)) for k in ['date','order_id','qty','price','line_total','unit','base_qty','base_unit']):continue
            source=audit['file']+':'+str(sheet);approved.add((source,label.strip()))
            evidence.append({'source_sha256':meta['sha256'],'raw_row':str(index),'label_sha256':hashlib.sha256(label.strip().encode()).hexdigest(),'original_blank_identity_date_amounts':True,'independent_of_gold':True})
        if sha(path)!=meta['sha256']:raise ValueError('Source changed during proof')
    return approved,evidence

class VerifiedLearningShops:
    def __init__(self,shops,root,manifest_sha,split_sha):
        self.shops=shops;self.root=Path(root)
        if sha(self.root/'DATA_MANIFEST.json')!=manifest_sha or sha(self.root/'SPLIT_MANIFEST.json')!=split_sha:raise ValueError('Source proof manifest mismatch')
        self.manifest=json.loads((self.root/'DATA_MANIFEST.json').read_text(encoding='utf-8'));split=json.loads((self.root/'SPLIT_MANIFEST.json').read_text(encoding='utf-8'));self.allowed=set(split['groups']['learning']);self.proofs={}
    def __getitem__(self,business):
        if business not in self.allowed:raise ValueError('Shop outside learning split')
        return self.shops[business]
    def annotation_proofs(self,business,shop):
        if business not in self.allowed:raise ValueError('Shop outside learning split')
        # Detect source changes even after the original proof has been cached.
        for audit in shop.audit:
            if audit['kind']=='sales':
                name='data/'+business+'/'+audit['file'];p=(self.root/name).resolve();meta=self.manifest['business_source_files'][name]
                if not p.is_relative_to(self.root.resolve()) or p.is_symlink() or sha(p)!=meta['sha256']:raise ValueError('Original source changed after proof')
        if business not in self.proofs and not annotation_evidence(shop)[0]:
            self.proofs[business]=(set(),[])
            return self.proofs[business]
        if business not in self.proofs:self.proofs[business]=original_notes(self.root,business,shop,self.manifest,self.allowed)
        return self.proofs[business]
