from typing import Any, Generator

import requests

from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin import Tool

from .utils import get_base_url, get_common_params, handle_api_error, get_image_messages


class TextToImageTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        try:
            params = get_common_params(tool_parameters)
            base_url = get_base_url(self.runtime.credentials["base_url"])
            response = requests.post(
                f"{base_url}/v1-openai/images/generations",
                headers={"Authorization": f"Bearer {self.runtime.credentials['api_key']}"},
                json=params,
                verify=self.runtime.credentials.get("tls_verify", True),
            )

            if not response.ok:
                raise Exception(handle_api_error(response))

            for message in get_image_messages(response, self):
                yield message

        except ValueError as e:
            yield self.create_text_message(str(e))
        except Exception as e:
            yield self.create_text_message(f"An error occurred: {str(e)}")
