# AGI Therapist

An AI-powered virtual therapist that integrates perception modules (Speech-to-Text, Tone Analysis, Natural Language Understanding) with memory systems (Working Memory and Long-Term Memory) to provide empathetic and context-aware therapeutic interactions.

## Features

- **Perception Module**: Processes audio input through STT, tone analysis, and NLU for comprehensive understanding.
- **Memory Systems**: Utilizes working memory for short-term context and long-term memory for persistent user data across sessions.
- **Web Interface**: Flask-based web app for easy interaction via browser.
- **Real-time Audio Processing**: Supports live audio transcription and analysis.
- **User-Specific Memory**: Maintains separate long-term memory databases for each user.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/nomaantalib/AGI-THERAPIST.git
   cd AGI-THERAPIST
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data (required for text processing):
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
   ```

## Usage

### Running the Web Application

Start the Flask web server:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000` to access the web interface. You can upload audio files for analysis or use the live transcription feature.

### Running the Live Transcription Script

For real-time speech-to-text processing:

```bash
python main_live.py
```

This will start listening to your microphone and process speech in real-time. Press 'q' + Enter to quit.

### Testing the Integrated System

Run the integration test:

```bash
python integration.py
```

This demonstrates the full pipeline from audio/text input to memory storage and retrieval.

### Testing Memory Modules

Test working memory:

```bash
curl http://localhost:5000/test_wm
```

Test long-term memory (replace 'default' with user ID):

```bash
curl "http://localhost:5000/test_ltm?user_id=default"
```

## API Endpoints

- `GET /`: Serves the main web interface
- `POST /analyze`: Analyzes uploaded audio file and returns perception results with memory logs
  - Parameters: `audio` (file), `user_id` (optional, defaults to 'default')
- `GET /test_wm`: Tests working memory functionality
- `GET /test_ltm`: Tests long-term memory functionality
  - Query parameter: `user_id` (optional, defaults to 'default')

## Project Structure

```
AGI-THERAPIST/
├── app.py                    # Main Flask application
├── main_live.py             # Live transcription script
├── integration.py           # Integrated system test
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── TODO.md                  # Development tasks
├── templates/
│   └── index.html           # Web interface template
├── perception/
│   ├── perception.py        # Main perception module
│   ├── stt/
│   │   └── stt_live.py      # Speech-to-text functionality
│   ├── tone/
│   │   └── tone_sentiment_live.py  # Tone and sentiment analysis
│   └── nlu/
│       └── nlu_live.py      # Natural language understanding
├── memory/
│   ├── working_memory.py    # Short-term memory implementation
│   └── long_term_memory.py  # Long-term memory with ChromaDB
└── long_term_memory_db_*    # User-specific ChromaDB databases
```

## Dependencies

- Flask: Web framework
- ChromaDB: Vector database for long-term memory
- Librosa: Audio processing
- NLTK: Natural language processing
- TextBlob: Sentiment analysis
- SoundDevice: Audio recording
- WebSockets: Real-time communication
- NumPy: Numerical computations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Enhancements

- [ ] Implement more advanced NLU models
- [ ] Add emotion recognition from facial expressions
- [ ] Integrate with external therapy APIs
- [ ] Implement conversation flow management
- [ ] Add user authentication and session management
