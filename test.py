import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
import logging

# --- CONFIGURATION ---
# These MUST match the paths from your test output
DB_PATH = "./test_chroma_db_retrieval"
COLLECTION_NAME = "test_collection_new"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_test_database():
    """
    Creates a new ChromaDB collection with dummy data.
    This function fixes the "Collection does not exist" error.
    """
    logger.info(f"--- Setting up Test Database ---")
    
    # Clean up old DB directory if it exists
    if os.path.exists(DB_PATH):
        logger.warning(f"Removing old database directory: {DB_PATH}")
        shutil.rmtree(DB_PATH)

    try:
        # 1. Initialize models and client
        client = chromadb.PersistentClient(path=DB_PATH)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Failed to load embedding model. Internet connection? Error: {e}")
        return False

    # 2. Create the collection
    logger.info(f"Creating new collection: {COLLECTION_NAME}")
    collection = client.create_collection(name=COLLECTION_NAME)
    
    # 3. Define dummy documents (based on your test.pdf)
    documents = [
        "All bids must be accompanied by a bid security of at least two percent (2%) of the Approved Budget.",
        "The approved budget cost is Php 6,919,129.11.",
        "The closing date for bids is July 31, 2025 at 1:00 PM.",
        "For further information, please refer to Joel C. Macasinag."
    ]
    
    # 4. Create embeddings and metadatas
    embeddings = embedding_model.encode(documents).tolist()
    
    # Add metadata to help your hybrid scoring
    metadatas = [
        {'source': 'test.pdf', 'page': 1, 'section_tag': 'Bidding Requirements', 'text_length': len(documents[0])},
        {'source': 'test.pdf', 'page': 1, 'section_tag': 'Financial Data', 'text_length': len(documents[1])},
        {'source': 'test.pdf', 'page': 1, 'section_tag': 'Bidding Requirements', 'text_length': len(documents[2])},
        {'source': 'test.pdf', 'page': 2, 'section_tag': 'General Information', 'text_length': len(documents[3])}
    ]
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # 5. Add to collection
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info(f"✅ Successfully populated collection '{COLLECTION_NAME}' with {len(documents)} documents.")
    return True

def run_retrieval_class_test():
    """
    Runs the 'Retrieval Class' test that was failing.
    """
    logger.info(f"\n--- Running 'Retrieval Class' Test ---")
    
    try:
        # Import the class from your script
        # This assumes your file is named "procure_rag_retrieval.py"
        from procure_rag_retrieval import ProcureRAGRetrieval
    except ImportError:
        logger.error("❌ ERROR: Could not import ProcureRAGRetrieval.")
        logger.error("Please make sure your script is saved as 'procure_rag_retrieval.py'")
        return False
    except Exception as e:
        logger.error(f"❌ ERROR importing procure_rag_retrieval.py: {e}")
        return False

    try:
        # 1. Initialize the retrieval system
        #    This time, it will find the collection we just made.
        retrieval_system = ProcureRAGRetrieval(
            chroma_db_path=DB_PATH,
            collection_name=COLLECTION_NAME 
        )
        
        logger.info("✅ ProcureRAGRetrieval class initialized successfully.")
        
        # 2. Run a test query
        query = "What is the bid security requirement?"
        logger.info(f"Running test query: '{query}'")
        
        context = retrieval_system.retrieve_and_display_context(
            user_query=query,
            n_initial=4,  # Retrieve all 4 dummy docs
            k_final=2     # Select the top 2
        )
        
        if context:
            logger.info("✅ Test query successful. Context was generated.")
            return True
        else:
            logger.error("❌ Test query failed to return context.")
            return False
            
    except Exception as e:
        logger.error(f"❌ An error occurred during the retrieval test: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Step 1: Create the database that your test was missing
    if setup_test_database():
        # Step 2: Run the test using your class
        if run_retrieval_class_test():
            print("\n======================================")
            print("✅✅✅ All tests passed! ✅✅✅")
            print("======================================")
        else:
            print("\n======================================")
            print("❌ 'Retrieval Class' test failed.")
            print("======================================")
    else:
        print("\n======================================")
        print("❌ Database setup failed. Test aborted.")
        print("======================================")