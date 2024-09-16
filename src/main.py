import PySimpleGUI as sg
import pyaudio
import threading
import websockets
import json
import logging
import asyncio
import time
import queue
import openai
from src.constants import DEEPGRAM_API_KEY, OPENAI_API_KEY
from src.prompts import SYSTEM_PROMPT

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Audio configuration
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Deepgram WebSocket configuration
WS_URL = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1"
WS_HEADER = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

# Keep-alive configuration
KEEP_ALIVE_INTERVAL = 5  # seconds
openai.api_key = OPENAI_API_KEY


async def websocket_handler(audio_queue, window):
    """
    Handles the WebSocket connection to Deepgram, sending audio data and receiving transcriptions.
    """
    reconnect_delay = 1
    while True:
        try:
            async with websockets.connect(WS_URL, extra_headers=WS_HEADER) as ws:
                logger.info("Connected to Deepgram WebSocket.")
                reconnect_delay = 1  # Reset reconnect delay on successful connection
                state = {'last_audio_time': time.time()}

                # Create tasks for sending audio, receiving messages, and keep-alive
                send_task = asyncio.create_task(send_audio(ws, audio_queue, state))
                receive_task = asyncio.create_task(receive_messages(ws, window))
                keep_alive_task = asyncio.create_task(send_keep_alive(ws, state))

                # Wait for any task to complete (e.g., due to an exception)
                done, pending = await asyncio.wait(
                    [send_task, receive_task, keep_alive_task],
                    return_when=asyncio.FIRST_EXCEPTION
                )

                # Cancel all pending tasks
                for task in pending:
                    task.cancel()

        except websockets.exceptions.InvalidStatusCode as e:
            logger.error(f"Invalid status code: {e.status_code}")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket exception: {e}")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)
        except Exception as e:
            logger.error(f"Unexpected exception: {e}")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)
        else:
            # If the connection closes normally, reset the reconnect delay
            reconnect_delay = 1

    logger.debug("WebSocket handler terminated.")


async def send_audio(ws, audio_queue, state):
    """
    Sends audio data from the queue to the WebSocket.
    """
    while True:
        audio_data = await asyncio.to_thread(audio_queue.get)
        if audio_data is None:
            logger.debug("Received stop signal for sending audio.")
            break
        try:
            await ws.send(audio_data)
            state['last_audio_time'] = time.time()
            logger.debug("Sent audio data to Deepgram.")
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket exception while sending audio: {e}")
            break


async def receive_messages(ws, window):
    """
    Receives transcription messages from the WebSocket and updates the GUI.
    """
    while True:
        try:
            response = await ws.recv()
            response_json = json.loads(response)
            if 'channel' in response_json and 'alternatives' in response_json['channel']:
                transcription = response_json['channel']['alternatives'][0]['transcript']
                if transcription:
                    window.write_event_value("-TRANSCRIPT-", transcription)
                    logger.debug(f"Received transcription: {transcription}")
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket exception while receiving messages: {e}")
            break
        except Exception as e:
            logger.error(f"Exception while receiving messages: {e}")
            break


async def send_keep_alive(ws, state, keep_alive_interval=KEEP_ALIVE_INTERVAL):
    """
    Sends keep-alive messages to maintain the WebSocket connection.
    """
    while True:
        await asyncio.sleep(keep_alive_interval)
        if time.time() - state['last_audio_time'] > keep_alive_interval:
            try:
                await ws.send(json.dumps({"type": "KeepAlive"}))
                logger.debug("Sent keep-alive message.")
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"WebSocket exception while sending keep-alive: {e}")
                break


class AudioRecorder:
    """
    Handles audio recording using PyAudio and sends audio data to a queue.
    """
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.p = None
        self.stream = None
        self.is_recording = False

    def start(self):
        if self.is_recording:
            logger.warning("Audio recording is already in progress.")
            return
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      frames_per_buffer=CHUNK,
                                      stream_callback=self.callback)
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
            self.p.terminate()
            self.p = None
            return
        self.stream.start_stream()
        self.is_recording = True
        logger.debug("Audio recording started.")

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def stop(self):
        if not self.is_recording:
            logger.warning("Audio recording is not active.")
            return
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.p:
            self.p.terminate()
            self.p = None
        # Signal the WebSocket handler to stop by sending None
        self.audio_queue.put(None)
        logger.debug("Audio recording stopped.")


def start_event_loop(loop, audio_queue, window):
    """
    Starts the asyncio event loop.
    """
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_handler(audio_queue, window))


def gen_llm_answer(transcript: str, window, history: str, temperature: float = 0.7) -> str:
    system_prompt = SYSTEM_PROMPT

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

    rv = response["choices"][0]["message"]["content"]
    window.write_event_value("-LLM_ANSWER-", rv)

    return rv


def main():
    # Define the GUI layout
    layout = [
        [sg.Text("Real-time Audio Transcription", font=("Helvetica", 20))],
        [sg.Button("Start Recording", key="-RECORD-")],
        [sg.Text("Transcript:", font=("Helvetica", 16))],
        [sg.Multiline(size=(60, 10), key="-TRANSCRIPT-", disabled=True, autoscroll=True)],
        [sg.Multiline(size=(60, 10), key="-LLM_ANSWER-", disabled=True, autoscroll=True)],
        [sg.Button("Exit")]
    ]

    # Create the window
    window = sg.Window("Audio Transcription App", layout, finalize=True)

    # Create the audio queue
    audio_queue = queue.Queue()

    # Create the AudioRecorder
    recorder = AudioRecorder(audio_queue)

    # Create the asyncio event loop
    loop = asyncio.new_event_loop()

    # Start the event loop in a separate daemon thread
    loop_thread = threading.Thread(target=start_event_loop, args=(loop, audio_queue, window), daemon=True)
    loop_thread.start()

    # Initialize recording state and transcript
    recording = False
    transcript = ""
    llm_answer = ""

    history = ""

    while True:
        event, values = window.read(timeout=100)
        if event in (sg.WIN_CLOSED, "Exit"):
            logger.debug("Exit event triggered. Closing application.")
            if recording:
                recorder.stop()
            # Stop the asyncio event loop
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join()
            break

        if event == "-RECORD-":
            if not recording:
                # Start recording
                recorder.start()
                recording = True
                window["-RECORD-"].update("Stop Recording")
                logger.debug("Recording started via GUI.")
            else:
                # Stop recording
                recorder.stop()
                recording = False
                window["-RECORD-"].update("Start Recording")
                logger.debug("Recording stopped via GUI.")

        if event == "-TRANSCRIPT-":
            # Append new transcription to the existing transcript
            transcription = values["-TRANSCRIPT-"]

            llm_thread = threading.Thread(target=gen_llm_answer, args=(transcription, window, history))
            llm_thread.start()

            transcript += transcription + "\n"
            window["-TRANSCRIPT-"].update(transcript)
            history += f'USER: {transcription}'

        if event == "-LLM_ANSWER-":
            # Append new transcription to the existing transcript
            generated_answer = values["-LLM_ANSWER-"]
            history += f'AI: {generated_answer}'
            llm_answer += generated_answer + "\n"
            window["-LLM_ANSWER-"].update(llm_answer)

    window.close()


if __name__ == "__main__":
    main()
