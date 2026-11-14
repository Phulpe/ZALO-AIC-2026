"""Expose only closed policy enums; source text stays in a private review file."""
exec((__import__('pathlib').Path(__file__).parent/'audit_source_identity.py').read_text(encoding='utf-8').split('for business in sorted(businesses):')[0])
from legacy_engine import norm
import re,hashlib
aggregate=collections.Counter();private=[]
patterns={
 'explicit_keep_identical_transactions':r'(?:cac )?dong trung (?:nhau )?(?:van|deu) (?:duoc tinh|la giao dich rieng)',
 'explicit_ids_scoped_per_month':r'ma (?:don|hoa don) (?:chi )?duy nhat trong (?:tung|moi) thang',
 'explicit_ids_scoped_per_file':r'ma (?:don|hoa don) (?:chi )?duy nhat trong (?:tung|moi) (?:tep|file|so)',
 'explicit_id_reset_each_file':r'ma (?:don|hoa don) duoc danh lai tu 1 (?:o |trong )?moi (?:tep|file)',
 'documented_export_duplicates':r'(?:xuat lap|lap do xuat|trung do xuat|dong trung lap)',
 'documented_currency_thousands':r'(?:don vi tien|tien te|don vi tinh tien)[^\n]{0,30}(?:nghin|ngan)',
 'mentions_duplicate_handling':r'(?:trung lap|dong lap|dong trung|du lieu lap)',
 'mentions_order_identity_scope':r'ma (?:don|hoa don)[^\n]{0,80}(?:lap lai|trung|duy nhat)',
}
for business in sorted(businesses):
 shop=shops[business];facts=[];lines=[]
 for line in shop.readme.splitlines():
  text=norm(line)
  matched=[key for key,pattern in patterns.items() if re.search(pattern,text)]
  for key in matched:aggregate[key]+=1;facts.append(key)
  if matched:lines.append({'line':line,'sha256':hashlib.sha256(line.encode()).hexdigest(),'policy_candidates':matched})
 private.append({'business':business,'readme_sha256':hashlib.sha256(shop.readme.encode()).hexdigest(),'policy_candidates':facts,'matched_source_lines':lines})
from relearning import sha
out=root/'relearning_cpu_v4_rules_review_20261110';out.mkdir(exist_ok=False);atomic(out/'PRIVATE_RULE_EVIDENCE.json',private);atomic(exp/'SOURCE_RULES_AGGREGATE.json',{'closed_policy_candidates':dict(aggregate),'source_text_exposed_to_chat':False,'private_evidence_sha256':sha(out/'PRIVATE_RULE_EVIDENCE.json')});print(json.dumps(dict(aggregate),indent=2))
