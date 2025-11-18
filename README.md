# Bringen Quellen Vertrauen?

### Einfluss von unterschiedlichen Quellenangaben auf das Nutzervertrauen von LLM-Anwendungen

Dieses Repository beinhaltet den Quellcode, sowie Code zur Datenanalyse der Bachelorarbeit von **E. Gaida** an der Hochschule Anhalt (2025).

## Über das Projekt

Die Arbeit untersucht, ob und wie die Darstellung von Quellen in **Retrieval Augmented Generation (RAG)**-Systemen das Vertrauen der Nutzer beeinflusst. Im Kontext eines fiktiven juristischen Szenarios ("FunkFabrik") wurden vier unterschiedliche RAG-Implementierungen entwickelt und mittels Mixed-Methods-Ansatz (Usability-Test $N=10$, Online-Survey $N=139$) evaluiert

### Untersuchte RAG-Varianten

1.  **Keine Quellen:** Reine Textantwort ohne Referenzen.
2.  **Dokumentennamen:** Auflistung der verwendeten Dokumententitel.
3.  **Volltext-Chunks:** Anzeige der kompletten Text-Chunks (Fixed-Size Chunking).
4.  **Propositionen:** Anzeige atomarer Fakten-Aussagen (Proposition-based Chunking).

## Tech Stack & Implementierung

Das System wurde als Web-Anwendung realisiert:

  * **Backend:** Python, LangChain, Chroma
  * **Frontend:** Flask, HTML/JS (AJAX für asynchrone Latenz-Optimierung)
  * **LLMs:** OpenAI API (`gpt-4o`, `o3-mini`, `text-embedding-ada-002`)

### Kern-Features

  * **Offline-Processing:** PDF-Parsing und Markdown-Konvertierung.
  * **Chunking-Strategien:** Implementierung von *Recursive Character Splitter* und *Proposition Extraction* via LLM].
  * **Advanced RAG:** Nutzung von Query Rewriting (HyDE-Ansatz) und Reranking zur Optimierung der Retrieval-Qualität.
  * **LLM-as-a-Judge:** Automatisierte Evaluierung der RAG-Antwortqualität.

Dieses Repository ist nicht für Produktiv-Zwecke gedacht.

## Zusammenfassung der Ergebnisse

Die Untersuchung ergab folgende Kernpunkte:

  * **Quellen als Werkzeug:** Quellenangaben dienen primär als Instrument zur Verifikation von Inkonsistenzen und weniger als automatischer Vertrauensanker.
  * **Einflussfaktoren:** Neben der Quellenart sind die wahrgenommene Plausibilität, die Kritikalität der Aufgabe und die Übereinstimmung mit dem Vorwissen entscheidend.
  * **Rolle von Vorerfahrung:** Das Vertrauen in die Anwendung ist hoch, wenn Teilnehmende ausdrücken, dass sie über Vorerfahrung verfügen oder wenn sie angeben, dass ihre bisherigen Erfahrungen mit den juristischen Themen bestätigt werden.
  * **Empfehlung:** RAG-Systeme sollten direkten Zugriff auf Originaldokumente ermöglichen und Systemgrenzen transparent kommunizieren.

## 📂 Struktur

  * `/PDFs` & `/Markdown`: Fiktionale Dokumente der Funkfabrik
  * `/Chunking`: Quellcode der Chunking-Algorithmen Fixed Size und Proposition
  * `/RAG-App`: Quellcode der Online-Abläufe des RAG-Systems
  * `/templates`: HTML für die RAG-Webapp
  * `/Testing`: Quellcode für automatisiertes Testing und Judgen (LLM-as-a-judge).
 
-----

**Autor:** E. Gaida  
**Hochschule Anhalt** | Fachbereich Informatik und Sprachen
