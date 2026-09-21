# baby-sing-a-song

아기 동요 앱 — a small Flask web app that lets little ones pick a Korean
nursery rhyme (동요) and sing along.

## Features

- Browse a catalog of classic Korean nursery rhymes
- Tap a song to see its lyrics on a colorful, kid-friendly card
- "따라 부르기" (sing-along) mode highlights each line in turn
- JSON API (`/api/songs`, `/api/songs/<id>`) and a `/healthz` endpoint

## Tech stack

- Python 3.12 + [Flask](https://flask.palletsprojects.com/) 3
- [pytest](https://docs.pytest.org/) for tests
- Vanilla HTML/CSS/JS frontend (no build step)

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Run the app

```bash
source .venv/bin/activate
flask --app wsgi run --host 0.0.0.0 --port 5000
```

Then open http://localhost:5000.

### Run the tests

```bash
source .venv/bin/activate
pytest -q
```

## Project layout

```
app/
  __init__.py       # create_app export
  factory.py        # Flask app factory + routes
  songs.py          # nursery rhyme catalog + lookup helpers
  templates/        # Jinja2 templates
  static/           # CSS + JS
tests/              # pytest suite
wsgi.py             # WSGI / dev-server entrypoint
```

## Cloud Agent environment

`.cursor/environment.json` provisions a virtualenv in `install` and serves the
app from the `web` terminal, so Cloud Agents can run and verify it end to end.
