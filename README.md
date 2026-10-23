NeuroStudy AI — minimal README

This repository processes NeuroSky EEG session data, extracts features,
trains a baseline activity classifier, and generates reports.

Quick run (after placing .xlsx session files in `data/raw/`):

```bash
cp config/activity_map.example.json config/activity_map.json
# Edit config/activity_map.json to map each input workbook filename (without .xlsx) to its activity.
python3 run_pipeline.py
```

The runner automatically uses the project's `.venv` when it exists, so you
do not need to activate it first.

`config/person_map.json` is optional and local; create it to map workbook
filenames to participant labels for person-independent evaluation. Data files,
generated reports, trained models, local virtual environments, and `.env` are
excluded from Git so participant data and credentials are not published.

Session coaching notes identify the participant using `config/person_map.json`
and name the recorded activity. Focus, distraction, fatigue, and next-session
recommendations are experimental heuristics, not validated EEG assessments.
To use Groq, copy `.env.example` to `.env` and replace the placeholder with your
`GROQ_API_KEY`. The coaching script loads `.env` from the project root; the
file is ignored by Git and should be copied securely (not committed) to each
machine where you run the project. Alternatively, set `GROQ_API_KEY` in the
environment. Without a key, the pipeline uses a local coaching template.

After running the pipeline, start the live dashboard chat:

```bash
python3 src/09_dashboard_server.py
```

Open [http://localhost:8765/](http://localhost:8765/) in your browser and keep
the server terminal running while chatting. The dashboard includes an ongoing
conversation, perspective selector (student, teacher, parent, coach, or
overall), and a new-conversation button. The live chat requires
`GROQ_API_KEY`; questions and session metrics are sent to Groq by the local
server, which keeps the key out of the browser. Stop the server with Ctrl+C.
Do not open `reports/index.html` directly for chat; that is a static file and
cannot reach the live feedback server.
