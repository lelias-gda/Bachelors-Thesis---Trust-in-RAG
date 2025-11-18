# Langchain Abhängigkeiten importieren
from langchain.document_loaders.pdf import PyPDFDirectoryLoader  # Importiert den PDF Loader, der PDF-Dateien aus einem Verzeichnis lädt
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importiert den Text Splitter, der Texte rekursiv in kleinere Abschnitte teilt
from langchain.embeddings import OpenAIEmbeddings  # Importiert OpenAI Embeddings, um Texte in Vektoren umzuwandeln
from langchain.schema import Document  # Importiert das Document Schema, das zur Repräsentation der geladenen Dokumente dient
from langchain_community.vectorstores import Chroma  # Importiert den Chroma Vector Store, um Vektorindizes zu speichern
from dotenv import load_dotenv  # Importiert dotenv, um Umgebungsvariablen (z.B. API-Keys) aus der .env Datei zu laden
import os  # Importiert das os Modul, das Funktionen zur Interaktion mit dem Betriebssystem bereitstellt
import shutil  # Importiert das shutil Modul, das Funktionen für Datei- und Verzeichnisoperationen auf höherer Ebene bereitstellt

# Verzeichnis-Pfad, in dem sich die PDF-Dokumente befinden
DATA_PATH = "PDFs/"
CHROMA_PATH = "FixedSizeDB-ADA"

# Funktion: load_documents
# Zusammenfassung:
# Lädt PDF-Dokumente aus dem angegebenen Verzeichnis mithilfe von PyPDFDirectoryLoader.
# Alle PDF-Dateien im Verzeichnis werden als Langchain Document Objekte geladen.
# Rückgabewert: Eine Liste von Document Objekten, die die geladenen PDF-Dokumente repräsentieren.
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Funktion: split_text
# Zusammenfassung:
# Teilt den Text der übergebenen Dokumente in kleinere Abschnitte.
# Verwendet dazu den RecursiveCharacterTextSplitter, um eine sinnvolle und überlappende Aufteilung zu gewährleisten.
# Parameter: documents (list[Document]) – Eine Liste von Document Objekten mit dem zu teilenden Text.
# Rückgabewert: Eine Liste von Document Objekten, die jeweils einen kleineren Textabschnitt repräsentieren.
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   
        chunk_overlap=20, 
        length_function=len,
        add_start_index=True, 
    )

    chunks = text_splitter.split_documents(documents) 
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.") 

    for chunk in chunks:
        chunk.page_content= chunk.metadata.get("source") + ": " + chunk.page_content
        print(chunk.page_content)

    return chunks



# Funktion: save_to_chroma
# Zusammenfassung:
# Speichert die übergebenen Textabschnitte in einer Chroma-Datenbank.
# Falls bereits eine Datenbank existiert, wird diese gelöscht, bevor die neuen Daten gespeichert werden.
# Parameter: chunks (list[Document]) – Eine Liste von Document Objekten, die die Textabschnitte darstellen.
# Rückgabewert: Kein Rückgabewert (None).
def save_to_chroma(chunks: list[Document]):
    
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH) 

    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(
            model="text-embedding-ada-002",
            max_retries=2 
        ),
        persist_directory=CHROMA_PATH
    )

    db.persist()  # Speichert die erstellte Chroma-Datenbank dauerhaft ab
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# Funktion: generateFixedSizeDataStore
# Zusammenfassung:
# Führt den gesamten Prozess aus, indem er:
# 1. Die Dokumente lädt,
# 2. Den Text in kleinere Abschnitte teilt und
# 3. Die Abschnitte in der Chroma-Datenbank speichert.
# Am Ende wird eine Erfolgsmeldung ausgegeben.
# Rückgabewert: Kein Rückgabewert (None).
def generateFixedSizeDataStore():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    print("Fertig!")

# Lädt die Umgebungsvariablen aus der .env Datei (API-Keys)
load_dotenv()

# Startet den gesamten Prozess zur Erstellung der Fixed Size Datenbank
generateFixedSizeDataStore()
