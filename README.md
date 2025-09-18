# IPIPE Codelmair: Lokaler Chatbot mit Gemma

Dieses Projekt ist eine Streamlit-Anwendung, die es Benutzern ermöglicht, mit Dokumenten in ganzen Ordnerstrukturen über einen lokal laufenden Chatbot zu chatten. Es verwendet das Gemma-Sprachmodell, das über Ollama ausgeführt wird, und eine RAG-Pipeline (Retrieval-Augmented Generation) mit LangChain und einer persistenten ChromaDB-Datenbank.

## Features

- **Lokale Ausführung:** Alle Komponenten (Modell, Embeddings, Anwendung) laufen lokal auf deinem Rechner. Es werden keine Daten an externe Server gesendet.
- **Verarbeitung ganzer Ordner:** Analysiere ganze Codebasen oder Dokumentensammlungen, indem du einfach einen Ordnerpfad angibst. Die Anwendung durchsucht rekursiv alle Unterordner.
- **Persistente Vektor-Datenbank:** Dank ChromaDB bleiben die verarbeiteten Daten auf der Festplatte gespeichert. Du musst die Dokumente nicht bei jedem Start der App neu verarbeiten.
- **GUI mit Streamlit:** Eine einfache und benutzerfreundliche Weboberfläche.
- **Unterstützung für mehrere Dateitypen:** Verarbeitet PDFs, Python-, Java-, JavaScript-, Text- und Markdown-Dateien.
- **Kontextbezogene Antworten:** Verwendet eine RAG-Pipeline, um Antworten basierend auf dem Inhalt der Dokumente zu generieren.
- **Gesprächsverlauf:** Der Chatbot berücksichtigt frühere Nachrichten im Gespräch.

## Voraussetzungen

1. **Python:** Du benötigst Python 3.8 oder neuer.
2. **Ollama:** Der Ollama-Dienst muss auf deinem System installiert sein und laufen.
   - **Installation:** Folge der Anleitung auf [ollama.com](https://ollama.com).
   - **Gemma-Modell:** Lade das Gemma-Modell herunter, indem du den folgenden Befehl in deinem Terminal ausführst:
     ```bash
     ollama pull gemma:2b
     ```

## Installation und Ausführung

1. **Klone das Repository oder lade die Dateien herunter.**

2. **Installiere die Python-Abhängigkeiten:**
   Navigiere in das Projektverzeichnis und führe den folgenden Befehl aus:
   ```bash
   pip install -r requirements.txt
   ```

3. **Starte die Streamlit-Anwendung:**
   Stelle sicher, dass dein Ollama-Dienst im Hintergrund läuft. Führe dann den folgenden Befehl aus:
   ```bash
   streamlit run app.py
   ```
   Beim ersten Start wird ein Ordner namens `chroma_db` im Projektverzeichnis erstellt, um die Vektor-Datenbank zu speichern.

4. **Öffne deinen Browser:**
   Streamlit öffnet automatisch einen neuen Tab in deinem Browser unter einer lokalen Adresse (normalerweise `http://localhost:8501`).

## Wie man die Anwendung benutzt

1. **Ordnerpfad eingeben:** Kopiere den vollständigen, absoluten Pfad zu dem Ordner, den du analysieren möchtest. Füge diesen Pfad in das Eingabefeld in der Seitenleiste ein.
2. **Ordner verarbeiten:** Klicke auf den Button "Ordner verarbeiten". Die Anwendung durchsucht den Ordner, liest die unterstützten Dateien, erstellt Embeddings und speichert sie in der lokalen ChromaDB-Datenbank. Dieser Vorgang kann je nach Größe und Anzahl der Dateien eine Weile dauern.
3. **Fragen stellen:** Sobald die Verarbeitung abgeschlossen ist (oder wenn du die App neu startest und bereits eine Datenbank existiert), kannst du deine Fragen im Chat-Eingabefeld stellen. Der Bot wird dir basierend auf den von dir bereitgestellten Dokumenten antworten.
