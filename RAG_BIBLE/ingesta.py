import logging
import sqlite3
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import NotFoundError

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "btx_collection"
BATCH_SIZE = 200
MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LOGGING_FORMATS = "%(asctime)s - %(levelname)s - %(message)s"
PATH_DB = "DB/biblia_btx.SQLite3"
DATA_BASE_QUERY = """SELECT b.long_name || ' ' || v.chapter || ':' || v.verse || ' "' || v.text || '"' as full_verse 
                     FROM books b 
                              JOIN verses v 
                                   ON b.book_number = v.book_number;"""

# ======= Logger configuration=======
logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMATS,
    handlers=[
        logging.FileHandler("bible_app.log"), # Escribe en el archivo .log
        logging.StreamHandler() # Muestra el mensaje en pantalla.
    ]
)
logger = logging.getLogger(__name__)

def models_initialization():
    model = SentenceTransformer(MODEL)
    return model

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection(name=COLLECTION_NAME) # Si existe la colección, se elimina y empieza de cero.
    except (NotFoundError, ValueError, Exception):
        logger.info(f"No se encontró la colección '{COLLECTION_NAME}', se creará una nueva ahora.") # Atrapa la exception y sigue la ejecución.

    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}) # Crea una colección de cero.
    return collection

def data_base_connection():
    logger.info("Inicializando conexión con base de datos")
    try:
        connection = sqlite3.connect(PATH_DB)
        return connection
    except ConnectionAbortedError:
        logger.critical("Error critico de conexión a base de datos", stack_info=True)

def verse_list_generator():
    connection = data_base_connection()
    cursor = connection.cursor()
    cursor.execute(DATA_BASE_QUERY)

    while True:
        # Trae un lote de la DB para ser eficiente con el disco
        rows = cursor.fetchmany(100)
        if not rows:
            break

        for row in rows:
            # Entrega uno por uno. row[0] es el string del versículo.
            yield row[0]

    connection.close()  # Cierra solo cuando termine el archivo completo

def batch_size_generator(iterable, batch_size=BATCH_SIZE):
    """
    Toma el generador de versículos y los agrupa en listas de tamaño BATCH_SIZE.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # Entrega el último lote si no llegó a completar el BATCH_SIZE
        yield batch


def embedding(batch, model):
    """Transforma el lote actual en vectores."""
    vectors_list = model.encode(batch, show_progress_bar=True)
    return vectors_list


# ======= Orchestrator =======

def main():
    model = models_initialization()
    collection = get_chroma_collection()

    logger.info("Iniciando proceso de ingesta modular...")

    # 1. Iniciar pipeline
    # verse_list_generator() devuelve los versículos uno a uno
    # batch_size_generator() los agrupa de a 100
    all_verses_flow = verse_list_generator()
    batches_flow = batch_size_generator(all_verses_flow, BATCH_SIZE)

    global_id_counter = 0

    # 2. Consumir pipeline
    for batch in batches_flow:
        vectors = embedding(batch, model)

        ids = [f"id_{i}" for i in range(global_id_counter, global_id_counter + len(batch))]

        collection.add(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=batch
        )

        global_id_counter += len(batch)
        logger.info(f"Progreso: {global_id_counter} versículos guardados.")

    logger.info("Ingesta completada.")


if __name__ == "__main__":
    main()