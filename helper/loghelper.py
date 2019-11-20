import logging

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)


def get_logger(name: str):
    return logging.getLogger(name)


if __name__ == "__main__":
    logging.debug("Test debug")
    logging.info("Test info.")
    logging.warning("Test warning")
    logging.error("Test error")
