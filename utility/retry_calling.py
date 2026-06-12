import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def with_retry(fun,retries:int=3,backOff:float=2.0):
    
    last_error=None
    for attempt in range(retries):
            try:
          
                return fun()
            except Exception as e:
                 last_error=e
                 wait= attempt**backOff
                 logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait:.0f}s...")
                 time.sleep(wait)
            logger.error(f"All {retries} attempts failed. Last error: {last_error}")
            raise last_error     

        