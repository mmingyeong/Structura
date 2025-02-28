# structura/logger.py
import logging

# Logger 설정
logger = logging.getLogger("Structura")
logger.setLevel(logging.INFO)

# 로그 포맷 정의
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 콘솔 핸들러 추가
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 핸들러 중복 추가 방지
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# 로거 사용 예시
if __name__ == "__main__":
    logger.info("Logger is successfully set up!")
