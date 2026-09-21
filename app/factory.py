"""Application factory for the 아기 동요 앱."""

from __future__ import annotations

from flask import Flask, abort, jsonify, render_template

from .songs import get_song, list_songs


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html", songs=list_songs())

    @app.route("/api/songs")
    def api_songs():
        return jsonify([song.to_dict() for song in list_songs()])

    @app.route("/api/songs/<song_id>")
    def api_song(song_id: str):
        song = get_song(song_id)
        if song is None:
            abort(404, description=f"'{song_id}' 동요를 찾을 수 없어요")
        return jsonify(song.to_dict())

    @app.route("/healthz")
    def healthz():
        return jsonify({"status": "ok", "song_count": len(list_songs())})

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": getattr(error, "description", "not found")}), 404

    return app
