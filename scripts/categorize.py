import json
from collections import Counter

rules = {
    "academics":     ["college", "faculty", "depart", "program", "course",
                      "curriculum", "catalog", "dar.ppu"],
    "admissions":    ["admis", "registr", "tuition", "fees", "scholarship"],
    "research":      ["research", "thesis", "publication", "journal"],
    "student_life":  ["student", "housing", "club", "scholarship"],
    "administration":["admin", "president", "council", "senate"],
    "library":       ["library"],
    "about":         ["about", "history", "vision", "mission"],
}

fixed = []
recategorized = Counter()

for line in open("data/raw/ppu_rag_data/_chunks.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("category") == "documents":
        url = r.get("url", "").lower()
        title = r.get("title", "").lower()
        haystack = url + " " + title
        matched = False
        for cat, keywords in rules.items():
            if any(kw in haystack for kw in keywords):
                r["category"] = cat
                recategorized[cat] += 1
                matched = True
                break
        if not matched:
            recategorized["documents"] += 1
    fixed.append(r)

with open("data/raw/ppu_rag_data/_chunks.jsonl", "w", encoding="utf-8") as f:
    for r in fixed:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Recategorized:")
for k, v in recategorized.most_common():
    print(f"  {v:5d}  {k}")