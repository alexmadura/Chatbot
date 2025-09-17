# Lokaler Chatbot mit Gemma und RAG

Dieses Projekt ist eine Streamlit-Anwendung, die es Benutzern ermöglicht, mit ihren eigenen Dokumenten (PDFs und Code-Dateien) über einen lokal laufenden Chatbot zu chatten. Es verwendet das Gemma-Sprachmodell, das über Ollama ausgeführt wird, und eine RAG-Pipeline (Retrieval-Augmented Generation) mit LangChain.

## Features

- **Lokale Ausführung:** Alle Komponenten (Modell, Embeddings, Anwendung) laufen lokal auf deinem Rechner. Es werden keine Daten an externe Server gesendet.
- **GUI mit Streamlit:** Eine einfache und benutzerfreundliche Weboberfläche zum Hochladen von Dateien und zum Chatten.
- **Unterstützung für mehrere Dateitypen:** Verarbeitet PDF-, Python-, Java-, JavaScript-, Text- und Markdown-Dateien.
- **Kontextbezogene Antworten:** Verwendet eine RAG-Pipeline, um Antworten basierend auf dem Inhalt der hochgeladenen Dokumente zu generieren.
- **Gesprächsverlauf:** Der Chatbot berücksichtigt frühere Nachrichten im Gespräch für natürlichere Interaktionen.

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

4. **Öffne deinen Browser:**
   Streamlit öffnet automatisch einen neuen Tab in deinem Browser unter einer lokalen Adresse (normalerweise `http://localhost:8501`).

## Wie man die Anwendung benutzt

1. **Dateien hochladen:** Verwende die Seitenleiste, um eine oder mehrere deiner Dokumente (PDFs, Code-Dateien etc.) hochzuladen.
2. **Dateien verarbeiten:** Klicke auf den Button "Dateien verarbeiten". Die Anwendung liest die Dateien, erstellt Embeddings und baut eine durchsuchbare Vektor-Datenbank auf. Dieser Vorgang kann je nach Größe und Anzahl der Dateien einen Moment dauern.
3. **Fragen stellen:** Sobald die Verarbeitung abgeschlossen ist, kannst du deine Fragen im Chat-Eingabefeld am unteren Rand der Seite stellen. Der Bot wird dir basierend auf den von dir bereitgestellten Dokumenten antworten.
