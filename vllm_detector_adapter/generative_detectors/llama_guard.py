# Standard
from typing import Optional
import asyncio

# Third Party
from fastapi import Request
from vllm.entrypoints.openai.protocol import ErrorResponse

# Local
from vllm_detector_adapter.generative_detectors.base import ChatCompletionDetectionBase
from vllm_detector_adapter.logging import init_logger
from vllm_detector_adapter.protocol import (
    ContentsDetectionRequest,
    ContentsDetectionResponse,
    ContentsDetectionResponseObject,
)
from vllm_detector_adapter.utils import DetectorType

logger = init_logger(__name__)


class LlamaGuard(ChatCompletionDetectionBase):

    DETECTION_TYPE = "risk"

    # Model specific tokens
    SAFE_TOKEN = "safe"
    UNSAFE_TOKEN = "unsafe"

    # Risk Bank name defined in the chat template
    RISK_BANK_VAR_NAME = "categories"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Process risk_bank_objs
        if self.risk_bank_objs:
            self.risk_bank = {
                risk.key.value: risk.value.value for risk in self.risk_bank_objs
            }
        else:
            logger.warning(
                f"{self.__class__.__name__} is missing the RISK_BANK_VAR_NAME variable"
            )
            self.risk_bank = {}

    def __post_process_result(self, response, scores, detection_type, req_content):
        """Function to process chat completion results for content type detection.

        Args:
            response: ChatCompletionResponse,
            scores: List[float],
            detection_type: str,
            req_content: str
        Returns:
            ContentsDetectionResponseObject
        """
        # NOTE: Llama-guard returns specific safety categories in the last line and in a csv format
        # this is guided by the prompt definition of the model, so we expect llama_guard to adhere to it
        # atleast for Llama-Guard-3 (latest at the time of writing)

        # In this function, we will basically remove those "safety" category from output and later on
        # move them to evidences.

        new_choices = []
        new_scores = []
        metadata = {}

        # NOTE: we are flattening out choices here as different categories
        for i, choice in enumerate(response.choices):
            content = choice.message.content
            if self.UNSAFE_TOKEN in content:
                choice.message.content = self.UNSAFE_TOKEN
                new_choices.append(choice)
                new_scores.append(scores[i])
                # Process categories
                metadata[self.RISK_BANK_VAR_NAME] = []
                for category in content.splitlines()[-1].split(","):
                    if category in self.risk_bank:
                        category_name = self.risk_bank.get(category)
                        metadata[self.RISK_BANK_VAR_NAME].append(category_name)
                    else:
                        logger.warning(
                            f"Category {category} not found in risk bank for model {self.__class__.__name__}"
                        )
            else:
                # "safe" case
                new_choices.append(choice)
                new_scores.append(scores[i])

        response.choices = new_choices
        return ContentsDetectionResponseObject.from_chat_completion_response(
            response, new_scores, detection_type, req_content, metadata
        )

    async def content_analysis(
        self,
        request: ContentsDetectionRequest,
        raw_request: Optional[Request] = None,
    ):
        """Function used to call chat detection and provide a /text/contents response"""

        # Apply task template if it exists
        if self.task_template:
            request = self.apply_task_template(
                request, fn_type=DetectorType.TEXT_CONTENT
            )
            if isinstance(request, ErrorResponse):
                # Propagate any request problems that will not allow
                # task template to be applied
                return request

        # Since separate batch processing function doesn't exist at the time of writing,
        # we are just going to collect all the text from content request and fire up
        # separate requests and wait asynchronously.
        # This mirrors how batching is handled in run_batch function in entrypoints/openai/
        # in vLLM codebase.
        completion_requests = self.preprocess_request(
            request, fn_type=DetectorType.TEXT_CONTENT
        )

        # Send all the completion requests asynchronously.
        tasks = [
            asyncio.create_task(
                self.process_chat_completion_with_scores(
                    completion_request, raw_request
                )
            )
            for completion_request in completion_requests
        ]

        # Gather all the results
        # NOTE: The results are guaranteed to be in order of requests
        results = await asyncio.gather(*tasks)

        # If there is any error, return that otherwise, return the whole response
        # properly formatted.
        processed_result = []
        for result_idx, result in enumerate(results):
            # NOTE: we are only sending 1 of the error results
            # and not every one (not cumulative)
            if isinstance(result, ErrorResponse):
                return result
            else:
                # Process results to split out safety categories into separate objects
                processed_result.append(
                    self.__post_process_result(*result, request.contents[result_idx])
                )

        return ContentsDetectionResponse(root=processed_result)
