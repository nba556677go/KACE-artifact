from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset



transcriber = pipeline(model="openai/whisper-large-v2", device_map="auto")
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:50]")
for out in transcriber(KeyDataset(dataset, "audio")):
    print(out)