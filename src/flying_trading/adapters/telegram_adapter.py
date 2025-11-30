import requests

from flying_trading.logger import get_logger

logger = get_logger(__name__)


class TelegramAdapter:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.logger = get_logger(f"{__name__}.TelegramAdapter")

    def send(self, message: str) -> None:
        """Alias for send_message for compatibility"""
        self.send_message(message)

    def send_message(self, message: str) -> dict:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        params = {
            "chat_id": self.chat_id,
            "text": message,
        }
        try:
            self.logger.debug(f"Sending Telegram message: {message[:50]}...")
            response = requests.post(url, params=params)
            if response.status_code != 200:
                error_msg = f"Failed to send message: {response.status_code}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
            self.logger.debug("Telegram message sent successfully")
            return response.json()
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}", exc_info=True)
            raise
