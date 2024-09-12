import time
import openai
from deepgram import DeepgramClient
from loguru import logger

from src.constants import DEEPGRAM_API_KEY, OPENAI_API_KEY, OUTPUT_FILE_NAME
from src.prompts import SYSTEM_PROMPT

openai.api_key = OPENAI_API_KEY



# SHORTER_INSTRACT = "Concisely respond, limiting your answer to 70 words."

LONGER_INSTRUCT = (
    "Before answering, take a deep breath and think one step at a time. Believe the answer in no more than 150 words."
)


def log_time(func):
    """
    Decorator to measure and log the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Executing {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper


@log_time
def transcribe_audio(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """
    Transcribes an audio file into text.

    Args:
        path_to_file (str, optional): The path to the audio file to be transcribed.

    Returns:
        str: The transcribed text.

    Raises:
        Exception: If the audio file fails to transcribe.
    """
    with open(path_to_file, "rb") as audio_file:
        try:
            transcript = openai.Audio.translate("whisper-1", audio_file)
        except Exception as error:
            logger.error(f"Can't transcribe audio: {error}")
            raise error
    return transcript["text"]


@log_time
def transcribe_audio_deepgram(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """
    Transcribes an audio file into text using Deepgram.

    Args:
        path_to_file (str): The path to the audio file to be transcribed.

    Returns:
        str: The transcribed text.

    Raises:
        Exception: If the audio file fails to transcribe.
    """
    with open(path_to_file, "rb") as audio_file:
        audio_data = audio_file.read()

    try:
        dp = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        response = dp.listen.rest.v("1").transcribe_file({
            'buffer': audio_data,
            'mimetype': 'audio/wav'
        }, {
            'punctuate': True,
            'diarize': True
        })
        transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
    except Exception as error:
        logger.error(f"Can't transcribe audio: {error}")
        raise error

    return transcript


def generate_answer(transcript: str, history: str, short_answer: bool = True, temperature: float = 0.7) -> str:
    """
    Generates an answer based on the given transcript using the OpenAI GPT-3.5-turbo model.

    Args:
        transcript (str): The transcript to generate an answer from.
        short_answer (bool): Whether to generate a short answer or not. Defaults to True.
        temperature (float): The temperature parameter for controlling the randomness of the generated answer.
        history (str): conversation history
    Returns:
        str: The generated answer.

    Example:
        ```python
        transcript = "Can you tell me about the weather?"
        answer = generate_answer(transcript, short_answer=False, temperature=0.8)
        print(answer)
        ```

    Raises:
        Exception: If the LLM fails to generate an answer.
    """
    # if short_answer:
    #     system_prompt = SYSTEM_PROMPT + SHORTER_INSTRACT
    # else:

    system_prompt = SYSTEM_PROMPT + LONGER_INSTRUCT

    if history:
        system_prompt += f"\nconversation history: \n {history}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
        )
    except Exception as error:
        logger.error(f"Can't generate answer: {error}")
        raise error
    return response["choices"][0]["message"]["content"]
