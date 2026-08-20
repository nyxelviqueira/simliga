"""Descarga los escudos mapeados para que el panel no dependa de URLs externas."""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simliga.output.team_identity import (
    ASSET_DIR,
    ESPN_LOGO_IDS,
    logo_filename,
    logo_source_url,
)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for name in sorted(ESPN_LOGO_IDS):
        target = ASSET_DIR / logo_filename(name)
        if target.exists() and target.stat().st_size > 0:
            print(f"ok     {target}")
            continue

        source = logo_source_url(name)
        if source is None:
            continue
        request = Request(source, headers={"User-Agent": "SimLiga/1.0"})
        try:
            with urlopen(request, timeout=30) as response:
                target.write_bytes(response.read())
        except (HTTPError, URLError, TimeoutError) as exc:
            raise SystemExit(f"No se pudo descargar {name} desde {source}: {exc}") from exc
        print(f"nuevo  {target}")


if __name__ == "__main__":
    main()
