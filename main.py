import time
from typing import List, TypedDict

import requests
from ulauncher.api.client.EventListener import EventListener
from ulauncher.api.client.Extension import Extension
from ulauncher.api.shared.action.DoNothingAction import DoNothingAction
from ulauncher.api.shared.action.ExtensionCustomAction import ExtensionCustomAction
from ulauncher.api.shared.action.RenderResultListAction import RenderResultListAction
from ulauncher.api.shared.event import ItemEnterEvent, KeywordQueryEvent
from ulauncher.api.shared.item.ExtensionResultItem import ExtensionResultItem

from fuzzyfinder import fuzzyfinder


class RunningModel(TypedDict):
    model: str
    name: str
    state: str


class Model(TypedDict):
    id: str
    name: str
    state: str


class LlamaSwapExtension(Extension):
    def __init__(self):
        super().__init__()
        self.subscribe(KeywordQueryEvent, KeywordQueryEventListener(self))
        self.subscribe(ItemEnterEvent, ItemEnterEventListener(self))
        self._cache_models: List[Model] = []
        self._cache_timestamp: float = 0
        self._cache_duration: int = 5
        self._last_error: str | None = None

    def get_base_url(self) -> str:
        return self.preferences.get("base_url", "http://localhost:8080")

    def get_api_token(self) -> str:
        return self.preferences.get("api_token", "")

    def get_headers(self) -> dict:
        headers = {}
        token = self.get_api_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def get_limit(self) -> int:
        return int(self.preferences.get("limit", "9"))

    def fetch_models(self) -> List[Model]:
        try:
            all_models_response = requests.get(
                f"{self.get_base_url()}/v1/models",
                headers=self.get_headers(),
                timeout=5,
            )
            all_models_response.raise_for_status()
            all_models = all_models_response.json().get("data", [])

            running_response = requests.get(
                f"{self.get_base_url()}/running",
                headers=self.get_headers(),
                timeout=5,
            )
            running_response.raise_for_status()
            running_data = running_response.json().get("running", [])
            running_map: dict[str, RunningModel] = {r["model"]: r for r in running_data}

            models: List[Model] = []
            for m in all_models:
                model_id = m.get("id", "")
                running_model = running_map.get(model_id)
                state = (
                    running_model.get("state", "stopped")
                    if running_model
                    else "stopped"
                )
                models.append(
                    {
                        "id": model_id,
                        "name": m.get("name", model_id),
                        "state": state,
                    }
                )

            self._last_error = None
            return models
        except requests.RequestException as e:
            self._last_error = str(e)
            return self._cache_models if self._cache_models else []

    def list_models(self) -> List[Model]:
        current_time = time.time()

        if current_time - self._cache_timestamp < self._cache_duration:
            return self._cache_models

        self._cache_models = self.fetch_models()
        self._cache_timestamp = current_time

        return self._cache_models

    def load_model(self, model_id: str, query: str | None):
        try:
            response = requests.post(
                f"{self.get_base_url()}/upstream/{model_id}",
                headers=self.get_headers(),
                json={"model": model_id},
                timeout=30,
            )
            response.raise_for_status()
            self._last_error = None
        except requests.RequestException as e:
            self._last_error = str(e)

        time.sleep(1)
        self._cache_timestamp = 0
        return self.render(query)

    def unload_model(self, model_id: str, query: str | None):
        try:
            response = requests.post(
                f"{self.get_base_url()}/api/models/unload/{model_id}",
                headers=self.get_headers(),
                timeout=30,
            )
            response.raise_for_status()
            self._last_error = None
        except requests.RequestException as e:
            self._last_error = str(e)

        time.sleep(1)
        self._cache_timestamp = 0
        return self.render(query)

    def unload_all_models(self, query: str | None):
        try:
            response = requests.post(
                f"{self.get_base_url()}/api/models/unload",
                headers=self.get_headers(),
                timeout=30,
            )
            response.raise_for_status()
            self._last_error = None
        except requests.RequestException as e:
            self._last_error = str(e)

        time.sleep(1)
        self._cache_timestamp = 0
        return self.render(query)

    def render(self, query: str | None):
        limit = self.get_limit()
        models = self.list_models()

        items: List[ExtensionResultItem] = []

        items.append(
            ExtensionResultItem(
                icon="images/llama-swap.png",
                name="Unload All Models",
                description="Unload all currently loaded models",
                on_enter=ExtensionCustomAction(
                    {"action": "unload_all", "query": query}, True
                ),
                keyword="unload-all",
            )
        )

        state_icons = {
            "ready": "images/online.png",
            "starting": "images/loading.png",
            "stopping": "images/loading.png",
            "stopped": "images/offline.png",
            "shutdown": "images/offline.png",
        }

        for model in models:
            state = model.get("state", "unknown")
            icon = state_icons.get(state, "images/llama-swap.png")
            name = f"{model.get('name', model.get('id', 'unknown'))} [{state}]"

            items.append(
                ExtensionResultItem(
                    icon=icon,
                    name=name,
                    description=f"ID: {model.get('id', 'unknown')}",
                    on_enter=ExtensionCustomAction(
                        {
                            "action": "toggle",
                            "model_id": model.get("id"),
                            "query": query,
                        },
                        True,
                    ),
                    keyword=model.get("name", model.get("id", "")),
                )
            )

        if self._last_error:
            items.append(
                ExtensionResultItem(
                    icon="images/error.png",
                    name="Error",
                    description=self._last_error,
                    on_enter=DoNothingAction(),
                    keyword="error",
                )
            )

        if not query:
            return RenderResultListAction(items[:limit])

        filtered: list[ExtensionResultItem] = list(
            fuzzyfinder(
                query,
                items,
                accessor=lambda item: item.get_keyword(),
            )
        )[:limit]

        return RenderResultListAction(filtered)


class KeywordQueryEventListener(EventListener):
    extension: LlamaSwapExtension

    def __init__(self, extension):
        super().__init__()
        self.extension = extension

    def on_event(self, event: KeywordQueryEvent, _):
        return self.extension.render(event.get_argument())


class ItemEnterEventListener(EventListener):
    extension: LlamaSwapExtension

    def __init__(self, extension):
        super().__init__()
        self.extension = extension

    def on_event(self, event: ItemEnterEvent, _):
        data = event.get_data()
        if not data:
            return DoNothingAction()

        action = data.get("action")
        if action == "toggle":
            model_id = data.get("model_id")
            query = data.get("query")
            models = self.extension.list_models()
            model = next((m for m in models if m.get("id") == model_id), None)
            if model and model.get("state") == "ready":
                return self.extension.unload_model(model_id, query)
            else:
                return self.extension.load_model(model_id, query)
        elif action == "unload_all":
            query = data.get("query")
            return self.extension.unload_all_models(query)

        return DoNothingAction()


if __name__ == "__main__":
    LlamaSwapExtension().run()
