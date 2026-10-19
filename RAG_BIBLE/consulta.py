import chromadb
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
import sys
import unicodedata

# ======= Constants declaration =======
CHROMA_DB_PATH = "./chroma_db"
LOGGER_FORMAT = '%(asctime)s, - %(levelname)s - %(message)s'
BI_MODEL, CROSS_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2', 'mixedbread-ai/mxbai-rerank-xsmall-v1'
SIMILARITY_THRESHOLD = 0.7
NUMBER_OF_CANDIDATES = 250
STOP_WORDS = {'que', 'quién', 'quien', 'quienes', 'cual', 'cuales', 'fue', 'el', 'la', 'los', 'las', 'un', 'una', 'uno', 'de', 'del', 'es', 'serán', 'eran'}

# ======= Logger configuration=======
logging.basicConfig(
    level=logging.INFO,
    format=LOGGER_FORMAT,
    handlers=[
        logging.FileHandler("bible_app.log"), # Escribe en el archivo ".log".
        logging.StreamHandler() # Muestra el mensaje en pantalla.
    ]
)
logger = logging.getLogger(__name__)

# ======= Initial configurations =======
def models_initialization():
    logger.debug("Instancia los modelos de lenguajes")
    bi_encoder = SentenceTransformer(BI_MODEL)
    cross_encoder = CrossEncoder(CROSS_MODEL)

    return bi_encoder, cross_encoder # Siempre que se devuelve 2 cosas separadas por coma, es una tupla.

def connecting_with_local_collections():
    try:
        logger.info("Configurando cliente con dirección de chroma db local.")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        logger.info("Buscando colecciones locales")
        bible_collection = search_local_collection(client)

        return bible_collection
    except Exception as e:
        logger.critical(f"Error al inicializar las colecciones: {e}")
        sys.exit(1)

def search_local_collection(client):
    list_of_local_collections = client.list_collections()

    local_collection_names = [collection.name for collection in list_of_local_collections]
    if not local_collection_names:
        raise FileNotFoundError("No existen colecciones locales. Ejecute: Ingesta.py")

    logger.info(f"Colección local encontrada: {local_collection_names}")
    local_bible_collection = client.get_collection(name=local_collection_names[0]) # Tráeme la colección X cuyo nombre es...

    return local_bible_collection

# ======= End of initial configurations =======

# ======= Retrieval =======
def execute_search_pipeline(user_input, bible_collection, bi_encoder, number_of_candidates=NUMBER_OF_CANDIDATES):
    logger.debug("Iniciar procesamiento y limpieza de texto")
    processed_user_input = user_input_filter(user_input)

    logger.debug("Retrieval: búsqueda inicial con bi-encoder")
    embedding_user_input = bi_encoder.encode(processed_user_input).tolist()
    """
    1) bi_encoder.encode(...) → Crea el vector numérico (en formato NumPy). Nota: bi_encoder trabaja si o si con String, no con listas.
    2) .tolist() → Traduce ese vector a una Lista de Python común para que ChromaDB la pueda procesar.
    """

    logger.debug("Usando distancia coseno entre el vector numérico: (la pregunta del usuario) y cada versículo existente en la base de datos")
    candidates = bible_collection.query(query_embeddings=[embedding_user_input], n_results=number_of_candidates)
    bible_verses = candidates['documents'][0] # candidates['documents'] se ve así:
                                              # [[ "Vers 1", "Vers 2", ... ], [ "Vers 1", "Vers 2", ... ]]
                                              # El [0] indica solo la primer sublista de la lista principal.
    if not bible_verses:
        logger.info("No se encontraron versículos")
        return []
    return bible_verses

# ======= Re-ranking =======
def re_ranking (bible_verses, cross_encoder, user_input):
    logger.debug("Comparando la pregunta del usuario con los versículos en documents.")
    user_query_and_verses = [[user_input, verses] for verses in bible_verses] # Crea la lista de pares (Pregunta + Versículo)

    logger.debug("Creando lista de relevancia de versículos")
    relevance_scores = cross_encoder.predict(user_query_and_verses)

    logger.debug("Uniendo los puntajes en: 'relevance_scores' con los versículos de: 'bible_verses'.")
    scored_verses = list(zip(relevance_scores, bible_verses)) # scored_veres[(x0,y0), (x1,y1), (x2,y2)...]

    logger.debug("Invertir la lista para que los puntajes mas altos estén primeros")
    scored_verses.sort(key=lambda x: x[0], reverse=True)

    logger.debug("Retornar solo los versículos (sin los números) para el usuario.")
    final_results = [verse for score, verse in scored_verses if score > SIMILARITY_THRESHOLD]

    return final_results

def orchestrator(user_input, bible_collection, bi_model, cross_model, number_of_candidates):
    logger.debug("Traer de la base de datos los versículos que mas se parezcan a la pregunta del usuario.")
    search_pipeline_results = execute_search_pipeline(user_input, bible_collection, bi_model, number_of_candidates)

    logger.debug("Usar el otro modelo para traer los versículos ordenados.")
    final_list = re_ranking(search_pipeline_results, cross_model, user_input)

    return final_list

# ======= Preprocessing: Cleaning functions =======
def user_input_filter(user_input):
    clean_input = clean_text_pipeline(user_input)
    words_without_accents = clean_input.split()
    stop_words = STOP_WORDS

    filtered_user_input = [word for word in words_without_accents if word not in stop_words] # Retorna un String con formato: "jesus nazaret"

    if len(filtered_user_input) <= 2:
        logger.info("Búsqueda de entidad/personaje detectada.")
        return " ".join(filtered_user_input)

    logger.info("Búsqueda de tema detectada.")
    return " ".join(filtered_user_input)

def clean_text_pipeline(user_input):
    logger.debug("1: Pasando todo a minúsculas")
    user_input = user_input.lower()

    logger.debug("2: Normalizando caracteres por descomposición (NFD)")
    user_input = unicodedata.normalize('NFD', user_input) # Génesis: G + e + ´ (acento) + n + e + s + i + s

    logger.debug("3: Quedándose solo con los caracteres que no son acentos")
    user_input = ''.join(char for char in user_input if unicodedata.category(char) != 'Mn') # Un acento cae dentro de la categoria: (Mark, Nonspacing)

    logger.debug("4: Normalizando caracteres a un formato estándar (NFC)") # Cualquier letra ñ o ü en la palabra, la vuelve a componer
    return unicodedata.normalize('NFC', user_input)

# ======== End of preprocessing =======

if __name__ == "__main__":
    logger.debug("Cargar todo")
    local_bible_collection_result = connecting_with_local_collections()
    bi, cross = models_initialization()

    user_query = input("¿Qué deseas buscar en las Escrituras?: ")

    logger.debug("Cargar orquestador.")
    resultados = orchestrator(user_query, local_bible_collection_result, bi, cross, NUMBER_OF_CANDIDATES)

    logger.debug("Mostrar resultados.")
    print(f"Resultados más relevantes para: '{user_query}'\n")
    for i, v in enumerate(resultados[:5], 1):  # El [:5] limita a los mejores 5
        print(f"{i}. {v}\n")