import pandas as pd


from langchain_core.messages import HumanMessage

import os
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Verzeichnis eine Ebene nach oben zum Suchpfad hinzufügen
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from  RAGCallFixedSizeADA import initChat as initFS, nextMessage as nextMessageFS
from RAGCallPropositionsADA import initChat as initProp, nextMessage as  nextMessageProp


def generateFixedSizeAnswer(msg: str):
    chat= initFS()
    chat.append(HumanMessage(
        msg
    ))
    answer, source, searchString = nextMessageFS(chat)

    return answer, source, searchString


def generatePropositionAnswer(msg: str):
    chat= initProp()
    chat.append(HumanMessage(
        msg
    ))
    answer, source, searchString = nextMessageProp(chat)
    return answer, source, searchString



def process_testcase(idx, prompt):
    """
    Ruft beide Generatoren auf und baut die Strings für Antworten und Quellen zusammen.
    Gibt ein Tupel zurück: (idx, genFA, sourcesFAStr, genProp, sourcesPropStr)
    """
    # FixedSize-Antwort
    genFA, sourcesFA, searchStringFA = generateFixedSizeAnswer(prompt)
    if sourcesFA:
        sourcesFAStr = "\n".join(doc.page_content for doc in sourcesFA)
    else:
        sourcesFAStr = "Leer"

    # Proposition-Antwort
    genProp, sourcesProp, searchStringProp = generatePropositionAnswer(prompt)
    if sourcesProp:
        sourcesPropStr = "\n".join(doc.metadata.get("source") + ": " + doc.page_content for doc in sourcesProp)
    else:
        sourcesPropStr = "Leer"

    return idx, genFA, sourcesFAStr, searchStringFA, genProp, sourcesPropStr, searchStringProp

def startTesting():
    global testcases

    # Ergebnisse in-place befüllen
    # Wir sammeln Futures und tragen die Ergebnisse dann ein:
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(process_testcase, idx, row["TestPrompt"]): idx
                for idx, row in testcases[80:90].iterrows()
            }

            for future in as_completed(futures):
                idx, genFA, sourcesFAStr, searchStringFA, genProp, sourcesPropStr, searchStringProp = future.result()

                testcases.at[idx, "AntwortFixed"]        = genFA
                testcases.at[idx, "QuellenFixed"]        = sourcesFAStr
                testcases.at[idx, "FixedSearchString"]   = searchStringFA
                testcases.at[idx, "AntwortPropositions"] = genProp
                testcases.at[idx, "QuellenPropositions"] = sourcesPropStr
                testcases.at[idx, "PropSearchString"]    = searchStringProp

                print(f"[{idx}] Prompt: {testcases.at[idx, 'TestPrompt']}")
                print("  → AntwortFixed:", genFA)
                print("  → QuellenFixed: ", sourcesFAStr.replace('\n', ' | '), " Searchstring: ", searchStringFA)
                print("  → AntwortProp:", genProp)
                print("  → QuellenProp:", sourcesPropStr.replace('\n', ' | '), " Searchstring: ", searchStringProp)
                print("-" * 40)
    except Exception as e:
        print("Fehler: ", e)
        formatExcel()
    formatExcel()

def formatExcel():
    global testcases
    with pd.ExcelWriter("Code/Testing/Testcases-filledOut.xlsx", engine="xlsxwriter") as writer:
        testcases.to_excel(writer,
            sheet_name="Tabelle1",
            index=False,
            engine="openpyxl")
        workbook = writer.book
        worksheet = writer.sheets["Tabelle1"]

        wrapFormat = workbook.add_format(
            {
                "text_wrap": True,
                "valign": "top",
                "bg_color": "#1C1A1D",
                "font_color": "#E6E5E6",
            }
        )
        minwidth= 20
        maxwidth= 90
        for idx, col in enumerate(testcases.columns):
            needed = max(testcases[col].astype(str).map(len).max(),len (col))+2
            if needed > maxwidth:
                width = maxwidth
            elif needed < minwidth:
                    width= minwidth
            else:
                width= needed
            worksheet.set_column(idx, idx, width, wrapFormat)
        
        headerFormat= workbook.add_format({
        "bold":      True,
        "font_color": "#E6E5E6",
        "border_color": "#FAFAFA",
        "text_wrap": True,
        "valign":    "top",
        "border":    1,
        "font_size": 15
        })
        for col_num, name in enumerate(testcases.columns):
            worksheet.write(0, col_num, name, headerFormat)
        
        # Filter & Freeze Panes
        #worksheet.autofilter(0, 0, len(testcases), len(testcases.columns) - 1)
        worksheet.freeze_panes(1,1)

filepath= "Code/Testing/Testcases-filledOut.xlsx"

testcases = pd.read_excel(filepath, engine="openpyxl", sheet_name="Tabelle1")


startTesting()
