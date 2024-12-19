# setup_log.py
from tqdm import tqdm
import os
import logging
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # tqdm.write()를 사용하여 진행 바를 방해하지 않고 출력
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 전체 로깅 레벨을 DEBUG로 설정

    # 파일 핸들러 설정
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)  # 파일에 모든 로그 레벨 기록
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # 콘솔 핸들러 설정
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)  # 콘솔에는 WARNING 이상 로그만 출력
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def setup_logging_for_model(model_name, log_dir):
    import logging
    import os
    import sys

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"{model_name}.log")

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거
    logger.handlers = []

    # 파일 핸들러 추가
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # 콘솔 핸들러를 표준 에러로 설정
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)  # WARNING 이상만 출력
    ch_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # 로그 전파 방지
    logger.propagate = False

    return logger


