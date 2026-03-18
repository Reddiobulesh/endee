from sentence_transformers import SentenceTransformer
import endee

# init embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# connect to endee
db = endee.Endee()

# create index
try:
    index = db.create_index("resumes", dimension=384, space_type="cosine")
except Exception:
    index = db.get_index("resumes")

# read resume text
with open("data/resume.txt") as f:
    text = f.read()

# split into chunks (grouping lines for better context)
lines = [t.strip() for t in text.split("\n") if t.strip()]
chunks = ["\n".join(lines[i:i+5]) for i in range(0, len(lines), 5)] # type: ignore

vectors_to_upsert = []
for i, chunk in enumerate(chunks):
    vector = model.encode(chunk).tolist()
    vectors_to_upsert.append({
        "id": str(i),
        "vector": vector,
        "meta": {"text": chunk}
    })

# store vectors in endee
index.upsert(vectors_to_upsert)

print("Resume embedded and stored successfully!")
