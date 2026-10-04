# Assets

Every image, GIF or other asset in `assets/` must be listed here with its source and
licence. Assets with an unclear licence are not used.

| File | Sign | Source URL | Author | Licence | Checked on |
|---|---|---|---|---|---|

## Fonts

Bundled in `src/signtrainer/web/fonts/` so the app works offline. All are under the
SIL Open Font License 1.1 (licence text next to each font). Downloaded 2026-10-01 from
https://github.com/google/fonts.

| File | Font | Used for | Licence |
|---|---|---|---|
| `Fredoka.ttf` | Fredoka (variable) | title, big letters | OFL 1.1, `OFL-Fredoka.txt` |
| `Nunito.ttf` | Nunito (variable) | subtitles, body text | OFL 1.1, `OFL-Nunito.txt` |
| `Quicksand.ttf` | Quicksand (variable) | small text, labels | OFL 1.1, `OFL-Quicksand.txt` |

Fredoka has no č, ć, đ; the CSS falls back to Nunito for those letters.
