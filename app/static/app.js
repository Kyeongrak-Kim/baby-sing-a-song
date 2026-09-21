(function () {
  "use strict";

  const placeholder = document.getElementById("placeholder");
  const card = document.getElementById("lyrics-card");
  const titleEl = document.getElementById("lyrics-title");
  const linesEl = document.getElementById("lyrics-lines");
  const singalongBtn = document.getElementById("singalong-btn");

  let singalongTimer = null;

  function clearSingalong() {
    if (singalongTimer) {
      clearInterval(singalongTimer);
      singalongTimer = null;
    }
    [...linesEl.children].forEach((li) => li.classList.remove("highlight"));
    singalongBtn.disabled = false;
  }

  function startSingalong() {
    clearSingalong();
    const lines = [...linesEl.children];
    if (lines.length === 0) return;
    singalongBtn.disabled = true;
    let index = 0;
    lines[0].classList.add("highlight");
    singalongTimer = setInterval(() => {
      lines.forEach((li) => li.classList.remove("highlight"));
      index += 1;
      if (index >= lines.length) {
        clearSingalong();
        return;
      }
      lines[index].classList.add("highlight");
    }, 1500);
  }

  async function loadSong(songId, button) {
    const response = await fetch(`/api/songs/${songId}`);
    if (!response.ok) {
      placeholder.textContent = "동요를 불러오지 못했어요 😢";
      return;
    }
    const song = await response.json();

    document
      .querySelectorAll(".song-button")
      .forEach((b) => b.classList.remove("active"));
    if (button) button.classList.add("active");

    titleEl.textContent = `${song.emoji} ${song.title}`;
    linesEl.innerHTML = "";
    song.lyrics.forEach((line) => {
      const li = document.createElement("li");
      li.textContent = line;
      linesEl.appendChild(li);
    });

    placeholder.hidden = true;
    card.hidden = false;
    clearSingalong();
  }

  document.querySelectorAll(".song-button").forEach((button) => {
    button.addEventListener("click", () => {
      loadSong(button.dataset.songId, button);
    });
  });

  singalongBtn.addEventListener("click", startSingalong);
})();
