from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from uuid import uuid4
from file_processor import splitter

doc_chunks = splitter()
documents = list()
metadata = list()

for chunk in doc_chunks:
    documents.append(chunk.page_content)
    metadata.append(chunk.metadata)

# 連線到資料庫
client = QdrantClient(url="http://localhost:6333")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_SIZE = embedder.get_sentence_embedding_dimension()
COLLECTION_NAME = "qdrant_ch6_test"

# 如果 collection 不存在就建立一個
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
		collection_name=COLLECTION_NAME,
		vectors_config={
			"text": models.VectorParams(
				size=VECTOR_SIZE,
				distance=models.Distance.COSINE,
			),
		},
	)

points = list()

for doc in tqdm(doc_chunks):
    vector = embedder.encode(doc.page_content).tolist()

    payload = {
        "text": doc.page_content,
        **doc.metadata,
    }

    points.append(
        models.PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload=payload,
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)
