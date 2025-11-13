import logging
from typing import Any

from multimedia_search.service.sentence_searching_service import SentenceBasedSearchHandler


logger = logging.getLogger(__name__)
logger.info("GameGator Inference")

_service = SentenceBasedSearchHandler(logger=logger)


def handle(data: Any, context: Any) -> Any:
    """
    Handle request
    :param data:
    :param context:
    :return:
    """
    try:
        if not _service.initialized:
            _service.initialize(context)

        logger.info(f"Received Data : {data}")
        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
