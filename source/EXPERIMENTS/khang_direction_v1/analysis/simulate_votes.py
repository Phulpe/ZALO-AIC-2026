"""Mo phong offline cau hinh bieu quyet tu log tung cau (moi mau co 'answer'), cham bang gold TRAIN.
Voi moi (S mau dau, min_votes): dap an = gia tri nhieu phieu nhat trong S mau dau neu so phieu >= min_votes, nguoc lai abstain.
Mau loi/abstain khong tinh phieu. stdout chi so tong hop (dung/sai/bo/net) theo cau hinh va theo nhom ids neu co.

--traj <trajectories.jsonl> --cap N: mo phong max-tokens N tu lan chay co max-tokens lon hon: mau nao co loi goi LLM
  completion_tokens > N thi coi nhu bi cat (bo phieu). Mau k cua cau q xac dinh bang seed: k = (seed - seed_nho_nhat_cua_q) // 1000.
Dung: python simulate_votes.py <khang_src> <gold_jsonl> <logs_dir> [--traj F --cap N] [ids_file ...]
"""
import collections, glob, json, os, re, sys

argv = sys.argv[1:]
traj = cap = None
if "--traj" in argv:
    i = argv.index("--traj"); traj = argv[i + 1]; del argv[i:i + 2]
if "--cap" in argv:
    i = argv.index("--cap"); cap = int(argv[i + 1]); del argv[i:i + 2]
src, gold_f, logs = argv[:3]
sys.path.insert(0, src)
from datamate.answers import same_answer  # noqa: E402

gold = {json.loads(l)["id"]: json.loads(l) for l in open(gold_f, encoding="utf-8")}
L = {}
for f in glob.glob(os.path.join(logs, "*.json")):
    d = json.load(open(f, encoding="utf-8"))
    L[d["id"]] = [s.get("answer") for s in d.get("samples") or []]

over = set()
if traj and cap:
    QID = re.compile(r"q_[0-9a-f]{12}")
    calls = collections.defaultdict(list)
    for l in open(traj, encoding="utf-8"):
        r = json.loads(l)
        user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        m = QID.search(user if isinstance(user, str) else json.dumps(user))
        if m and r.get("seed") is not None:
            calls[m.group(0)].append((int(r["seed"]), int(r["reply"].get("completion_tokens") or 0)))
    for q, cs in calls.items():
        s0 = min(s for s, _ in cs)
        for s, t in cs:
            if t > cap:
                over.add((q, (s - s0) // 1000))
    L = {q: [None if (q, k) in over else a for k, a in enumerate(v)] for q, v in L.items()}
    print(json.dumps({"cap": cap, "samples_truncated": len(over)}))

subsets = {"all": set(L)}
for p in argv[3:]:
    subsets[os.path.basename(p)] = set(open(p).read().split()) & set(L)


def decide(ans, S, mv):
    groups = []
    for a in ans[:S]:
        if not a or a.get("type") in (None, "abstain"):
            continue
        for g in groups:
            if same_answer(a, g[0]):
                g[1] += 1
                break
        else:
            groups.append([a, 1])
    if not groups:
        return None
    top = max(groups, key=lambda g: g[1])
    return top[0] if top[1] >= mv else None


maxS = max((len(v) for v in L.values()), default=0)
for name, ids in subsets.items():
    for S in range(1, maxS + 1):
        for mv in range(1, S + 1):
            c = w = a = 0
            for i in ids:
                r = decide(L[i], S, mv)
                if r is None:
                    a += 1
                elif same_answer(r, gold[i]):
                    c += 1
                else:
                    w += 1
            print(json.dumps({"set": name, "n": len(ids), "samples": S, "min_votes": mv, "correct": c, "wrong": w, "abstain": a, "net": c - w}))
