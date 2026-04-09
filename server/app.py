from __future__ import annotations

from typing import Any

from inference import app as api
from inference import main as inference_main


class ScriptApp:
    def __init__(self, fastapi_app: Any) -> None:
        self._fastapi_app = fastapi_app

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return self._fastapi_app(*args, **kwargs)
        return main()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fastapi_app, name)


app = ScriptApp(api)


def main() -> None:
    inference_main()


if __name__ == "__main__":
    main()
