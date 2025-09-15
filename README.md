# Generación de embeddings con DeepSeek-R1:1.5B y búsqueda semántica en MongoDB Atlas Vector Search

Se describe de manera detallada cómo generar embeddings con el modelo **DeepSeek-R1:1.5B** y utilizarlos para realizar búsquedas semánticas en **MongoDB Atlas Vector Search**, optimizando la eficiencia y precisión de tus consultas.

## 1. Preparación del entorno de desarrollo

```bash
pip install torch transformers pymongo
```

* `torch` y `transformers`: necesarios para cargar y ejecutar el modelo DeepSeek-R1.
* `pymongo`: para interactuar con MongoDB Atlas y gestionar la colección de documentos.

## 2. Configuración de MongoDB Atlas

1. Crear un cluster en MongoDB Atlas.
2. Habilitar **Vector Search** en la base de datos.
3. Crear una colección para almacenar documentos, por ejemplo `documents`.

```json
{
  "_id": ObjectId(),
  "title": "Ejemplo de documento",
  "text": "Este es el contenido del documento",
  "embedding": []
}
```

4. Crear un índice vectorial en el campo `embedding` para habilitar búsquedas semánticas:

```javascript
db.documents.createIndex({ embedding: "cosine" })
```

## 3. Carga y preparación del modelo DeepSeek-R1:1.5B

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "deepseek-r1:1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

## 4. Función para generar embeddings de texto

```python
import torch

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy().flatten()
```

> Nota: Ajusta `max_length` según la longitud de los fragmentos de texto que procesarás.

## 5. Preparación de documentos y generación de embeddings

```python
from pymongo import MongoClient

client = MongoClient("mongodb+srv://<usuario>:<password>@cluster.mongodb.net")
db = client['mi_base']
collection = db['documents']

documents = [
    {"title": "Doc1", "text": "Contenido del documento 1"},
    {"title": "Doc2", "text": "Contenido del documento 2"}
]

for doc in documents:
    doc['embedding'] = generate_embedding(doc['text']).tolist()
    collection.insert_one(doc)
```

## 6. Realización de consultas semánticas con Vector Search

```python
query_text = "Contenido relacionado con documento 1"
query_embedding = generate_embedding(query_text).tolist()

results = collection.aggregate([
    {
        "$search": {
            "index": "default",
            "knnBeta": {
                "vector": query_embedding,
                "path": "embedding",
                "k": 5
            }
        }
    }
])

for r in results:
    print(r['title'], r['text'])
```

> Ajusta el parámetro `k` para controlar la cantidad de resultados devueltos.

## 7. Buenas prácticas para generación de embeddings y búsquedas

1. Fragmenta textos largos en bloques de 500 a 1000 palabras antes de generar embeddings.
2. Normaliza el texto: convierte a minúsculas, elimina stopwords y realiza limpieza básica.
3. Procesa documentos en lotes (batch) para mejorar eficiencia.
4. Almacena los embeddings como listas de floats (`.tolist()`) para compatibilidad con MongoDB.
5. Selecciona el índice `cosine` para búsquedas de similitud semántica o `euclidean` para distancia euclidiana.

Con esta guía, podrás indexar documentos, generar embeddings y realizar búsquedas semánticas en MongoDB Atlas de manera efectiva utilizando **DeepSeek-R1:1.5B**.
