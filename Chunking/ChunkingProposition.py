from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings 
from langchain.schema import Document  
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


from dotenv import load_dotenv # Importiere dotenv, um API-Keys aus der .env-Datei zu laden
import os 
import shutil
import json 

# Initialisiere Chat-Modelle mit verschiedenen Konfigurationen
llmGPT35 = ChatOpenAI(
    model="gpt-3.5-turbo", temperature=1, max_tokens=None, timeout=None, max_retries=2
)
llmGPT4omini = ChatOpenAI(
    model="gpt-4o-mini", temperature=1, max_tokens=None, timeout=None, max_retries=2
)
llmGPTGPT4o = ChatOpenAI(
    model="gpt-4o", temperature=1, max_tokens=None, timeout=None, max_retries=2
)
llmGPTo3mini = ChatOpenAI(
    model="o3-mini", max_completion_tokens=None, timeout=None, max_retries=2
)

# Initialisiere die Embedding-Funktion mit dem Modell "text-embedding-ada-002"
embeddingFunction = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=2)

# Verzeichnis, in dem die Markdown-Dateien liegen
DATA_PATH = "Markdown/"
# Pfad zum Verzeichnis, in dem die Chroma-Datenbank gespeichert wird
CHROMA_PATH = "PropositionsDB-ADA"

# Funktion: queryLLM
# Zusammenfassung: Diese Funktion fragt ein spezifiziertes LLM-Modell mit einem gegebenen Abfragetext an und liefert dessen Antwort zurück.
# Input-Parameter: model (str) – Modellname; queryText – Der Abfragetext oder Nachrichtenliste, die an das LLM gesendet wird.
# Output: Gibt den Antwortinhalt des LLM als String zurück.
def queryLLM(model: str, queryText):
    if model == "gpt-3.5-turbo":
        global llmGPT35
        return llmGPT35.invoke(queryText).content
    elif model == "gpt-4o-mini":
        global llmGPT4omini
        return llmGPT4omini.invoke(queryText).content
    elif model == "gpt-4o":
        global llmGPTGPT4o
        return llmGPTGPT4o.invoke(queryText).content
    elif model == "o3-mini":
        global llmGPTo3mini
        return llmGPTo3mini.invoke(queryText).content
    # Falls keines der bekannten Modelle übereinstimmt, wird ein neues LLM-Objekt erzeugt
    llm = ChatOpenAI(
        model=model, temperature=1, max_tokens=None, timeout=None, max_retries=2
    )
    return llm.invoke(queryText).content


# Funktion: load_documents
# Zusammenfassung: Lädt Markdown-Dokumente aus dem angegebenen Verzeichnis.
# Input-Parameter: Keine externen Parameter.
# Output: Gibt eine Liste von Document Objekten zurück, die die geladenen Markdown-Dokumente repräsentieren.
def load_documents():
    document_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    # Lädt die Markdown-Dokumente und gibt sie als Liste von Document Objekten zurück
    return document_loader.load()

# Funktion: split_text
# Zusammenfassung: Teilt den Text der geladenen Dokumente in kleinere Abschnitte und erzeugt daraus Vorschlags-Chunks.
# Input-Parameter: documents (list[Document]) – Liste von Document Objekten, die den zu teilenden Text enthalten.
# Output: Gibt eine Liste von Document Objekten zurück, die jeweils einen Proposition-Chunk repräsentieren.
def split_text(documents: list[Document]):
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    )

    
    propositionChunks: list[Document] = []
    

    sysPromptDocSum = SystemMessage(
        """Du bist ein Chatbot, der dabei hilft automatisch Dokumente zu verarbeiten.
            Du wirst mit den ersten Abschnitten aus einem Vertragsdokument konfrontiert und sollst in ein bis zwei Sätzen zusammenfassen worum es geht.
            In dieser Mini-Zusammenfassung sollst du folgende Fragen beantworten:
            - Was ist das für ein Dokument?
            - Wer sind die Vertragsparteien?"""
    )
    
    sysPromptProposition = SystemMessage(
        """Zerlege den "Inhalt" in klare und einfache Aussagen, die auch aus dem Kontext heraus verständlich sind.

        Zerlege zusammengesetzte Sätze in einfache Sätze. Bewahre dabei, wenn möglich, die ursprüngliche Formulierung aus der Eingabe.
        Für jede benannte Entität, die von zusätzlichen beschreibenden Informationen begleitet wird, trenne diese Informationen in eine eigene, separate Aussage.
        Dekontextualisiere die Aussage, indem du nötige Modifikatoren zu Substantiven oder ganzen Sätzen hinzufügst und Pronomen (z.B. "es", "er", "sie", "ihnen", "ihm", "das") durch den vollständigen Namen der referenzierten Entität ersetzt.
        Außerdem nutzt auch nicht allgemeine Substantive (wie z.B. "Arbeitnehmer", "Dokument", "Arbeitsvertrag", "Vertrag", "Vertragsparteien", "Arbeitgeber"). Jede Aussage muss für sich selbst stehen. 
        Beispiel: Die Aussage "Der Kunde bestellt monatlich Technikprodukte" ist nicht spezifisch genug, da nicht spezifiert wird, wer der Kunde ist und von wem er etwas bestellt.
        Gib die Ergebnisse als Liste von Strings im JSON-Format aus. WICHTIG: Füge keine weiteren Zeichen wie ``` oder "json" hinzu! Sonst kann ich die Ergebnisse von dir nicht verwenden.
        
        BEISPIEL 1:
        Input:
        Dokumentname: AV_Felix_Silbermann.
        Kontext: Dieses Dokument ist der Arbeitsvertrag zwischen Felix Silbermann und der FunkFabrik GmbH. Er ist als IT- und Technikmanager eingestellt.
        Inhalt:
        Das Arbeitsverhältnis wird auf unbestimmte Zeit geschlossen. Die ersten vier Monate gelten als
        Probezeit. Während der Probezeit kann das Arbeitsverhältnis beiderseits mit einer Frist von zwei
        Wochen gekündigt werden.

        Output:
        [
        "Das Arbeitsverhältnis zwischen Felix Silbermann und der Funkfabrik GmbH wird auf unbestimmte Zeit geschlossen.",
        "Die ersten vier Monate des Arbeitsverhältnis zwischen Felix Silbermann und der FunkFabrik GmbH gelten als Probezeit",
        "Während der Probezeit im Arbeitsverhältnis von Silbermann kann das Arbeitsverhältnis mit einer zwei-wöchigen Frist gekündigt werden"
        ]
        BEISPIEL 2:
        Input:
        Dokumentname: FunkFabrik_AGB_aktuell.md
        Kontext: Dieses Dokument sind die allgemeinen Geschäftsbedingungen der Website FunkFabrik. Es gilt für Besucher und Kunden auf der Website.
        Inhalt: ABSCHNITT 3 - GENAUIGKEIT, VOLLSTÄNDIGKEIT UND RECHTZEITIGKEIT DER INFORMATIONEN

        Wir sind nicht verantwortlich, wenn die auf dieser Seite zur Verfügung gestellten Informationen
        nicht genau, vollständig oder aktuell sind. Das Material auf dieser Website dient nur der
        allgemeinen Information und sollte nicht als alleinige Grundlage für Entscheidungen herangezogen
        werden, ohne primäre, genauere, vollständigere oder aktuellere Informationsquellen zu prüfen.
        Jegliches Vertrauen in das Material auf dieser Website geschieht auf eigene Verantwortung.
        Diese Seite enthält möglicherweise gewisse historische Informationen. Historische Informationen
        sind nicht unbedingt aktuell und werden lediglich zu Ihrer Orientierung bereitgestellt. Wir behalten
        uns das Recht vor, Inhalte auf dieser Website jederzeit zu ändern, sind aber nicht verpflichtet,
        irgendwelche Informationen auf unserer Website zu aktualisieren. Sie stimmen zu, dass Sie
        verantwortlich dafür sind, Änderungen auf unserer Website zu überwachen.
        
        Output:
        [
        "Die Funkfabrik ist laut ihren AGB nicht dafür verantwortlich, wenn Informationen auf der Seite nicht genau, vollständig oder aktuell sind.",
        "Die Inhalte auf der Website FunkFabrik dienen nur der allgemeinen Information.",
        "Die Inhalte auf der Website FunkFabrik sollten nicht die alleinige Grundlage für Entscheidungen sein, ohne primäre, genauere, vollständigere oder aktuellere Informationsquellen zu prüfen.",
        "Das Vertrauen in das Material auf der Website FunkFabrik geschieht auf eigene Verantwortung der Besucher und Kunden.",
        "Die Website FunkFabrik enthält möglicherweise gewisse historische Informationen, die nur zur Orientierung bereitgestellt werden.",
        "Die FunkFabrik behält sich das Recht auf Änderung der Inhalte auf ihrer Website ein.",
        "Die FunkFabrik ist nicht dazu verpflichtet Informationen auf ihrer Website zu aktualisieren",
        "Die Kunden und Besucher der Website FunkFabik sind dafür verantwortlich Änderungen an der Website zu überwachen."
        ]
        """
    )
    
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        messagesDocSum = [sysPromptDocSum]
        messagesDocSum.append(
            HumanMessage(
                "Das ist das Dokument, was du zusammenfassen sollst:" +
                "\n".join([chunk for chunk in chunks[:5]])
            )
        )
        docSummary = queryLLM("o3-mini", messagesDocSum)
        
        # Für jeden Chunk im Dokument, erzeuge Propositionen
        for chunk in chunks:
            messagesProposition = [
                sysPromptProposition, 
                HumanMessage(
                    "Input:\nDokumentname: " + doc.metadata.get("source") + 
                    "\nKontext: " + docSummary +
                    "\nInhalt:" + chunk
                )
            ]
            try:
                propositions = queryLLM("o3-mini", messagesProposition)
                #LLM hat bei JSON-Antworten häufiger Strings eingebaut, die das Auslesen behindert haben
                propositions = propositions.replace("```", "")
                propositions = propositions.replace("json", "")
                propositionsJSON = json.loads(propositions)
                
                for prop in propositionsJSON:
                    propositionChunks.append(Document(page_content=prop, metadata=doc.metadata))
            #Fehlerbehandlung, falls das JSON-Schema falsch genutzt wird
            except json.JSONDecodeError as e:
                print("Fehler beim Parsen des JSON: ", e)
    
    print(f"Split {len(documents)} documents into {len(propositionChunks)} Propositions")
    
    return propositionChunks

# Funktion: save_to_chroma
# Zusammenfassung: Speichert die übergebene Liste von Document Objekten in einer Chroma-Datenbank.
# Input-Parameter: chunks (list[Document]) – Liste von Document Objekten, die als Chunks gespeichert werden sollen.
# Output: Kein Rückgabewert (None).
def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=2),
        persist_directory=CHROMA_PATH,
    )
    
    # Speichert die Datenbank dauerhaft auf der Festplatte
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# Funktion: generatePropositionsDataStore
# Zusammenfassung: Führt den gesamten Prozess aus: Laden der Dokumente, Aufteilen des Textes in Proposition-Chunks, Ausgabe der Chunks und Speichern in einer Chroma-Datenbank.
# Input-Parameter: Keine externen Parameter.
# Output: Kein Rückgabewert (None).
def generatePropositionsDataStore():
    documents = load_documents() 
    propositions = split_text(documents)
    for prop in propositions:
        print("Neuer Chunk:")
        print(prop.page_content)
        print("-------------------------------------")
    save_to_chroma(propositions) 
    print("Fertig!")

load_dotenv()
generatePropositionsDataStore()
