import logging
from typing import Any

from multimedia_search.service.multimedia_searching_service import MultiMediaSearchHandler
from multimedia_search.service.sentence_searching_service import SentenceBasedSearchHandler

logger = logging.getLogger(__name__)
logger.info("GameGator Multi Service Inference")

_service_mm = MultiMediaSearchHandler(logger=logger)
_service_ss = SentenceBasedSearchHandler(logger=logger)

REINITIALIZE_CACHED = False


def handle(data: Any, context: Any) -> Any:
    """
    Handle request
    :param data:
    :param context:
    :return:
    """
    try:
        if not _service_ss.initialized:
            _service_ss.initialize(context)

        if not _service_mm.initialized:
            _service_mm.initialize(context)

        if data is None:
            return None

        search_type = "text"
        for row in data:
            query = row.get("data") or row.get("body")
            search_type = query.get("search_type")
            break

        if search_type == "text":
            # Sentence transformer is responsible for text search
            logger.info("Searching with custom text")
            preprocessing_result = _service_ss.preprocess(data)
            inference_result = _service_ss.inference(preprocessing_result)
            result = _service_ss.postprocess(inference_result)
        elif search_type in ["image", "game", "new_game"]:
            # Multimedia search is responsible for image and game_id search, also new_game registration
            logger.info("Searching | Registry with multimedia")
            preprocessing_result = _service_mm.preprocess(data)
            inference_result = _service_mm.inference(preprocessing_result)
            result = _service_mm.postprocess(inference_result)

            if search_type == "new_game" and REINITIALIZE_CACHED:
                # Reinitialize cached data if new game is registered
                _service_mm.reinitialize()
                _service_ss.reinitialize()

        else:
            logger.error(f"Invalid search type: {search_type}")
            return None

        return result
    except Exception as e:
        raise e
