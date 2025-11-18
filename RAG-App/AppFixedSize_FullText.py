from flask import Flask, render_template, request, session, jsonify
from RAGCallFixedSizeADA import nextMessage, initChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


chat = []
app = Flask(__name__)
app.secret_key = "irgendein_sicherer_schluessel"  # Wichtig für Sitzungen (Sessions)


@app.route("/", methods=["GET", "POST"])
def index():
    # Falls noch keine Chat-Historie existiert, erstellen wir eine neue
    global chat
    if len(chat) == 0:
        # startChat() gibt eine initiale Liste mit SystemMessage und AIMessage zurück
        chat= initChat()
    # Rendere das HTML-Template mit der aktuellen Chat-Historie und möglichen Quellen
    if request.method == "POST":
        # Rufe die Funktion nextMessage auf, um eine Antwort zu generieren
        chat.append(
            HumanMessage(
                    request.form.get("query","")
                )
            )
        answer, sourceForAnswerObj, searchStr = nextMessage(chat)
        addNewAIMessage(answer, sourceForAnswerObj)
        # nextMessage() hängt die neue HumanMessage und die AI-Antwort direkt an session["messages"] an
        # Wir müssen nur noch sicherstellen, dass die Session aktualisiert wird
        session.modified = True
        sourceForAnswerJS = []
        if sourceForAnswerObj:
            for doc in sourceForAnswerObj:
                sourceForAnswerJS.append({
                    "metadata":doc.metadata,
                    "content": doc.page_content,
                    #"score": score
                })
        return jsonify({"message": answer, "sources": sourceForAnswerJS})
    return render_template("ChatSourcesFullText.html", chat_history=chat)


@app.route("/reset", methods= ["POST"])
def resetButton():
    global chat
    chat =initChat()
    
    return render_template("ChatSourcesFullText.html", chat_history=chat)

def addNewAIMessage(message, sources):
    global chat
    chat.append(
        AIMessage(
            content= message,
            sources= sources
        )
    )

if __name__ == "__main__":
    app.run( port= 5000)
