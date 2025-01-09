import re
import numpy as np
import torch
import whisper_timestamped as whisper

from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play

import gradio as gr
import spaces

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUFFER = 0.2


def new_audio_location(audio_path):
    extension = audio_path.split(".")[-1]
    output_path = audio_path.replace("." + extension, ".wav")
    audio = AudioSegment.from_file(audio_path)
    audio.export(output_path, format="wav")
    return output_path


def replace_segments_with_note(
    audio_file, start_times, end_times, censoring_type, censoring_opts
):
    current_time = 0
    audio = AudioSegment.from_file(audio_file)

    if type(censoring_type) == str:
        censoring_type = AudioSegment.from_file(censoring_type)
        censoring_type = censoring_type * int(
            max(len(audio), len(start_times) * censoring_opts["duration"])
            / len(censoring_type)
            + 1
        )

    for start, end in zip(start_times, end_times):
        note = None

        if type(censoring_type) == int:
            note = Sine(censoring_type).to_audio_segment(duration=(end - start))
        else:
            note = censoring_type[
                current_time : current_time
                + max((end - start - 2 * BUFFER), censoring_opts["duration"])
            ]
            current_time += end - start

        audio = audio[: start + BUFFER] + note + audio[end - BUFFER :]

    return audio


def get_transcript(audio_path):
    audio = whisper.load_audio(audio_path)
    model = whisper.load_model("small", device=device)
    transcript = whisper.transcribe_timestamped(
        model, audio, language="en", refine_whisper_precision=BUFFER, vad="silero"
    )

    transcript_text = transcript["text"]
    transcript_timestamps = transcript["segments"]

    return transcript_text, transcript_timestamps


def get_start_and_end_times(transcript_timestamps, censoring_words):
    start_times = []
    end_times = []

    words_done = 0
    for i, segment in enumerate(transcript_timestamps):
        for j, word in enumerate(segment["words"]):
            for censoring_word in censoring_words:
                if re.search(censoring_word, word["text"].lower()):
                    start_times.append(int(word["start"] * 1000))
                    end_times.append(int(word["end"] * 1000))

            words_done += 1

    return start_times, end_times


@spaces.GPU()
def censor_words(
    audio_path,
    censoring_words,
    censoring_type,
    censoring_freq,
    censoring_audio,
    censoring_duration,
):
    censoring_words = [word.strip().lower() for word in censoring_words.split(",")]

    audio_path = new_audio_location(audio_path)

    transcript, transcript_timestamps = get_transcript(audio_path)

    transcript_text = transcript.split(" ")
    transcript_text = [word for word in transcript_text if word]

    assert len(transcript_text) == sum(
        [len(segment["words"]) for segment in transcript_timestamps]
    )

    start_times, end_times = get_start_and_end_times(
        transcript_timestamps, censoring_words
    )

    censored_audio = replace_segments_with_note(
        audio_path,
        start_times,
        end_times,
        censoring_freq if censoring_type == "Frequency" else censoring_audio,
        {"duration": censoring_duration},
    )

    transcript = transcript.strip()

    sample_rate = censored_audio.frame_rate
    audio_data = np.array(censored_audio.get_array_of_samples())
    if censored_audio.channels == 2:
        audio_data = audio_data.reshape((-1, 2))

    return [transcript, (sample_rate, audio_data)]


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Word Censoring
    - Built on Whisper and Python
    - Censors words or phrases in audio.
    - Supports frequency and custom audio censoring (this will continue where the last censorship stopped in the audio).
"""
    )

    with gr.Row():
        with gr.Column():
            audio = gr.Audio(type="filepath")
            censoring_words = gr.Textbox(
                label="Censoring Words (comma separated; supports regex)"
            )

            with gr.Row():
                with gr.Column():
                    censoring_type = gr.Radio(
                        ["Frequency", "Audio"],
                        label="Censoring Type",
                        value="Frequency",
                    )

                    with gr.Accordion("Censoring Frequency Options"):
                        censoring_freq = gr.Slider(
                            20, 20000, label="Frequency", step=1, value=440
                        )

                    with gr.Accordion("Censoring Audio Options", open=False):
                        censoring_audio = gr.Audio(type="filepath")

                with gr.Group():
                    censoring_duration = gr.Slider(
                        4, 3000, label="Min Censorship Duration (ms)", step=1, value=4
                    )

        with gr.Column():
            output = gr.Audio()
            transcript = gr.Textbox(
                label="Transcript",
                placeholder="Transcript will appear here",
            )

    button = gr.Button("Censor Words")

    gr.Examples(
        examples=[
            [
                "tests/example.m4a",
                "audio",
                "Frequency",
                440,
                None,
                4,
            ],
            [
                "tests/example.m4a",
                "audio",
                "Audio",
                440,
                "tests/Rick-Roll-Sound-Effect.mp3",
                1000,
            ],
        ],
        inputs=[
            audio,
            censoring_words,
            censoring_type,
            censoring_freq,
            censoring_audio,
            censoring_duration,
        ],
    )

    button.click(
        censor_words,
        inputs=[
            audio,
            censoring_words,
            censoring_type,
            censoring_freq,
            censoring_audio,
            censoring_duration,
        ],
        outputs=[transcript, output],
    )


if __name__ == "__main__":
    demo.launch()
