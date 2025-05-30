# Language Model Utils
# Some functions copied from: https://github.com/milesaturpin/cot-unfaithfulness/blob/main/utils.py

import random
from time import sleep

from pyrate_limiter import Duration, RequestRate, Limiter


def add_retries(f):

    def wrap(*args, **kwargs):
        max_retries = 10
        num_retries = 0
        max_pause = 90
        while True:
            try:
                result = f(*args, **kwargs)
                return result
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except KeyError:
                raise KeyError
            except Exception as e:
                wait = 2 ** num_retries + random.choice([0, 1, 2])
                seconds_to_pause = min(wait, max_pause)
                print("Error: ", e, "\nRetrying in ", seconds_to_pause, "seconds")
                if num_retries == max_retries:
                    print("Max retries reached. Exiting OAI request.")
                    raise e
                num_retries += 1
                sleep(seconds_to_pause)
            
    return wrap

OAI_rate = RequestRate(18, Duration.MINUTE)
limiter = Limiter(OAI_rate)
