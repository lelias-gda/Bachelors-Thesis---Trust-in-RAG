import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


MODEL_NAME = "gpt-5-mini"
INPUT_FILE_PATH = "Code/Testing/Testcases-filledOut.xlsx"
OUTPUT_FILE_PATH = "Code/Testing/Testcases-analyzed-WithQuestion.xlsx"
MAX_WORKERS = 5 # Anzahl der parallelen Anfragen an die API

# Initialisiere das LLM-Modell einmal global
llm = ChatOpenAI(
    model=MODEL_NAME, 
    temperature=1, 
    max_tokens=None, 
    timeout=None, 
    max_retries=2
)

# --------------------------------------------------------------------------
# 2. Definition der JSON-Ausgabeschemata
# --------------------------------------------------------------------------

class ScaleEvaluation(BaseModel):
    """Definiert das JSON-Schema für die Skalen-Bewertung."""
    score: int = Field(description="Die Bewertung auf einer Skala von 1 (falsch) bis 5 (vollständig richtig).", ge=1, le=5)
    reasoning: str = Field(description="Eine kurze Begründung für die vergebene Bewertung.")

class BooleanEvaluation(BaseModel):
    """Definiert das JSON-Schema für die Ja/Nein-Bewertung."""
    isCorrect: bool = Field(description="True, wenn die Antwort korrekt ist, andernfalls False.")
    reasoning: str = Field(description="Eine kurze Begründung für die Entscheidung.")

# Binde die Schemata an das LLM-Modell
llmWithScaleJson = llm.with_structured_output(ScaleEvaluation)
llmWithBooleanJson = llm.with_structured_output(BooleanEvaluation)

# --------------------------------------------------------------------------
# 3. Bewertungsfunktionen
# --------------------------------------------------------------------------

def evaluateScaleResponse(correctAnswer: str, generatedAnswer: str, question: str) -> ScaleEvaluation:
    """
    Bewertet eine generierte Antwort auf einer Skala von 1-5 im Vergleich zu einer korrekten Antwort.

    Args:
        correctAnswer: Der Referenztext mit der korrekten Antwort.
        generatedAnswer: Die vom System generierte Antwort, die bewertet werden soll.

    Returns:
        Ein ScaleEvaluation-Objekt mit Punktzahl und Begründung.
    """
    promptTemplate = ChatPromptTemplate.from_messages([
        ("system",
        """Du bist ein unparteiischer Richter. Deine Aufgabe ist es, zu bewerten, ob eine Frage richtig beantwortet wurde. Dazu erhälst du die Frage, sowie eine Korrekte Antwort, die alle benötigten Informationen enthält.
        Eine richtige Antwort muss nicht zwangsweise alle Informationen der korrekten Antwort, die dir gegeben wird enthalten, allerdings müssen richtige Antworten nur Informationen enthalten, die mit der korrekten Antwort begründbar sind.
        Bewerte die generierte Antwort auf einer Skala von 1 bis 5.
            1 bedeutet, die Antwort ist komplett falsch oder irrelevant.
            5 bedeutet, die Antwort ist vollständig korrekt und deckt sich perfekt mit der korrekten Antwort.
            Gib immer eine Begründung für deine Bewertung an."""),
        ("""human, 
         **Fragestellung:**\n{question}\n\n
        **Korrekte Antwort:**\n{correctAnswer}\n\n
        **Generierte Antwort:**\n{generatedAnswer}""")
    ])
    
    chain = promptTemplate | llmWithScaleJson
    return chain.invoke({"question": question, "correctAnswer": correctAnswer, "generatedAnswer": generatedAnswer})

def evaluateBooleanResponse(correctAnswer: str, generatedAnswer: str, question: str) -> BooleanEvaluation:
    """
    Bewertet, ob eine generierte Antwort im Vergleich zu einer korrekten Antwort richtig (True) oder falsch (False) ist.

    Args:
        correctAnswer: Der Referenztext mit der korrekten Antwort.
        generatedAnswer: Die vom System generierte Antwort, die bewertet werden soll.

    Returns:
        Ein BooleanEvaluation-Objekt mit dem Wahrheitswert und einer Begründung.
    """
    promptTemplate = ChatPromptTemplate.from_messages([
        ("system", 
            """Du bist ein unparteiischer Richter. Deine Aufgabe ist es, zu bewerten, ob eine Frage richtig beantwortet wurde. Dazu erhälst du die Frage, sowie eine Korrekte Antwort, die alle benötigten Informationen enthält.
            Eine richtige Antwort muss nicht zwangsweise alle Informationen der korrekten Antwort, die dir gegeben wird enthalten, allerdings müssen richtige Antworten nur Informationen enthalten, die mit der korrekten Antwort begründbar sind.
            Antworte ausschließlich mit 'True', wenn die generierte Antwort sachlich und semantisch korrekt die Frage beantwortet, andernfalls mit 'False'.
            Gib immer eine Begründung für deine Entscheidung an."""),
        ("human", 
            """**Fragestellung**:\n{question}\n\n
            **Korrekte Antwort:**\n{correctAnswer}\n\n
            **Generierte Antwort:**\n{generatedAnswer}""")
    ])
    
    chain = promptTemplate | llmWithBooleanJson
    return chain.invoke({"question": question,"correctAnswer": correctAnswer, "generatedAnswer": generatedAnswer})

# --------------------------------------------------------------------------
# 4. Verarbeitungslogik
# --------------------------------------------------------------------------

def processRowForAnalysis(idx, row):
    """
    Führt die LLM-Bewertungen für eine einzelne Zeile des DataFrames aus.

    Args:
        idx: Der Index der Zeile.
        row: Die Daten der Zeile als Pandas Series.

    Returns:
        Ein Tupel mit dem Index und den vier Bewertungsergebnissen.
    """
    question= row["TestPrompt"]
    correctAnswer = row["RichtigeAntwort"]
    
    # Antworten extrahieren (und sicherstellen, dass sie Strings sind)
    answerFixed = str(row["AntwortFixed"])
    answerProp = str(row["AntwortPropositions"])
    
    # Bewertung für "FixedSize"
    scaleFixed = evaluateScaleResponse(correctAnswer=correctAnswer, generatedAnswer=answerFixed, question= question)
    booleanFixed = evaluateBooleanResponse(correctAnswer=correctAnswer, generatedAnswer=answerFixed, question= question)
    
    # Bewertung für "Propositions"
    scaleProp = evaluateScaleResponse(correctAnswer=correctAnswer, generatedAnswer=answerProp, question= question)
    booleanProp = evaluateBooleanResponse(correctAnswer=correctAnswer, generatedAnswer=answerProp, question= question)
    
    return idx, scaleFixed, booleanFixed, scaleProp, booleanProp

def analyzeTestcases(testcasesData):
    """
    Startet die Analyse aller Testfälle im DataFrame mithilfe eines ThreadPoolExecutors.

    Args:
        testcasesData: Der Pandas DataFrame mit den Testfällen.
    
    Returns:
        Der aktualisierte DataFrame mit den neuen Bewertungsspalten.
    """
    # Füge die neuen Spalten hinzu, falls sie noch nicht existieren
    newColumns = {
        "ScaleFixed": None, "ReasoningScaleFixed": None,
        "BooleanFixed": None, "ReasoningBooleanFixed": None,
        "ScaleProp": None, "ReasoningScaleProp": None,
        "BooleanProp": None, "ReasoningBooleanProp": None
    }
    for col, default in newColumns.items():
        if col not in testcasesData.columns:
            testcasesData[col] = default
    try:
        for idx, row in testcasesData[167:].iterrows():
            try:
                idx, scaleFixed, booleanFixed, scaleProp, booleanProp= processRowForAnalysis(idx, row)
                print("+"*40)
                print(f"Index: {idx}\nScaleFixed: {scaleFixed.score}\nBooleanFixed:{booleanFixed.isCorrect}\nScaleProp: {scaleProp.score}\nbooleanProp:{booleanProp.isCorrect}")
                testcasesData.at[idx, "ScaleFixed"] = scaleFixed.score
                testcasesData.at[idx, "ReasoningScaleFixed"] = scaleFixed.reasoning
                testcasesData.at[idx, "BooleanFixed"] = booleanFixed.isCorrect
                testcasesData.at[idx, "ReasoningBooleanFixed"] = booleanFixed.reasoning
                        
                testcasesData.at[idx, "ScaleProp"] = scaleProp.score
                testcasesData.at[idx, "ReasoningScaleProp"] = scaleProp.reasoning
                testcasesData.at[idx, "BooleanProp"] = booleanProp.isCorrect
                testcasesData.at[idx, "ReasoningBooleanProp"] = booleanProp.reasoning

                print(f"Zeile {idx} erfolgreich verarbeitet.")
            except Exception as e:
                        print(f"Fehler bei der Verarbeitung von Zeile {idx}: {e}")
                        testcasesData.at[idx, "ScaleFixed"] = f"ERROR: {e}"
    except Exception as e:
        print(f"Ein schwerwiegender Fehler ist aufgetreten: {e}")
    finally:
        # Speichere den aktuellen Fortschritt, auch bei Fehlern
        saveAndFormatExcel(testcasesData, OUTPUT_FILE_PATH)
        return testcasesData

# --------------------------------------------------------------------------
# 5. Excel-Speicherfunktion
# --------------------------------------------------------------------------

def saveAndFormatExcel(testcasesData, filePath):
    """
    Speichert den DataFrame in einer formatierten Excel-Datei.

    Args:
        testcasesData: Der zu speichernde Pandas DataFrame.
        filePath: Der Pfad zur Ausgabedatei.
    """
    print(f"Speichere Ergebnisse in {filePath}...")
    with pd.ExcelWriter(filePath, engine="xlsxwriter") as writer:
        testcasesData.to_excel(writer, sheet_name="Analyse", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Analyse"]

        # Zellformat für Umbruch und Hintergrund
        wrapFormat = workbook.add_format({
            "text_wrap": True,
            "valign": "top",
        })

        # Spaltenbreiten anpassen
        minWidth = 20
        maxWidth = 70
        for idx, col in enumerate(testcasesData.columns):
            needed = max(testcasesData[col].astype(str).map(len).max(), len(col)) + 2
            width = min(maxWidth, max(minWidth, needed))
            worksheet.set_column(idx, idx, width, wrapFormat)
        
        # Header-Format
        headerFormat = workbook.add_format({
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "border": 1,
            "font_size": 12
        })
        for colNum, value in enumerate(testcasesData.columns.values):
            worksheet.write(0, colNum, value, headerFormat)
            
        # Zeilen einfrieren
        worksheet.freeze_panes(1, 1)
    print("Speichern abgeschlossen.")


# --------------------------------------------------------------------------
# 6. Hauptausführung
# --------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Fehler: Die Eingabedatei '{INPUT_FILE_PATH}' wurde nicht gefunden.")
    else:
        
        print("Lade Testfälle...")
        testcases = pd.read_excel(INPUT_FILE_PATH, engine="openpyxl")
        
        print("Starte die Analyse...")
        analyzedTestcases = analyzeTestcases(testcases)
        
        print("\nAnalyse abgeschlossen.")