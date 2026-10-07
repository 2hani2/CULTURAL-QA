from encoders import MultimodalEncoder
from structured_store import load_all_raw_records

enc = MultimodalEncoder(enable_live_fetch=True)
records = [r for r in load_all_raw_records("./data/raw") if r.get("dataset") == "DRISHTIKON"]
print(f"{len(records)} DRISHTIKON records loaded")

for q in ["what is kathakali","bhartnatayam","khatak"]:
    results = enc.retrieve_relevant_images(q, records, top_k=1)
    print(q, "->", results[0][1].get("image_name") if results else "none")
if not results:
    print("No results -- check that enable_live_fetch is actually on and DRISHTIKON records have real image_url paths")
else:
    for score, rec in results:
        print(f"{score:.4f}  {rec.get('image_name')}  (record #{rec.get('id')})")