from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
load_dotenv()

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",)

document=[
"Retired sailor Arthur spends his afternoons on the pier, telling stories of the sea to anyone who will listen.",

"Kenji stayed up all night, fueled by instant noodles, trying to fix a single bug in his video game's code.",

"Maria, a baker with flour permanently dusted on her apron, believed the secret to perfect bread was patience.",

"By day David was a meticulous accountant, but by night he filled notebooks with surprisingly tender poetry about the stars.",

"Elara, a street artist known only by her tag, brought vibrant but temporary dragons to life on city sidewalks with chalk.",
]

query="Who is elara?"

doc_embed=embeddings.embed_documents(document)
doc_query=embeddings.embed_query(query)

scores=cosine_similarity([doc_query], doc_embed)[0]

index,score=sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"'{query}':")
print(f" {document[index]}")
