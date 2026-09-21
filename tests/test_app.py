import pytest

from app import create_app
from app.songs import get_song, list_songs


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client


def test_catalog_not_empty():
    songs = list_songs()
    assert len(songs) >= 1
    assert all(song.lyrics for song in songs)


def test_get_song_found():
    song = get_song("gom-se-mari")
    assert song is not None
    assert song.title == "곰 세 마리"


def test_get_song_missing():
    assert get_song("does-not-exist") is None


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "아기 동요 앱".encode("utf-8") in response.data


def test_api_songs(client):
    response = client.get("/api/songs")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == len(list_songs())
    assert {"id", "title", "emoji", "lyrics"} <= set(data[0].keys())


def test_api_song_detail(client):
    response = client.get("/api/songs/nabiya")
    assert response.status_code == 200
    data = response.get_json()
    assert data["title"] == "나비야"


def test_api_song_not_found(client):
    response = client.get("/api/songs/nope")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["song_count"] == len(list_songs())
