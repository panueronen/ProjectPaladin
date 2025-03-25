import os
import time
import whisper
import torch
from scipy.io import wavfile
import numpy as np
import asyncio
import telnetlib3
import re
import unicodedata
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
from dotenv import load_dotenv


# === CONFIGURATION ===
load_dotenv()
API_KEY = os.getenv("API_KEY")
RECORDING_DIR = r"D:\PMABotRecordings"
WHISPER_MODEL_SIZE = "small"
TOXICITY_THRESHOLD = 0.96
RULE_BASED_BADWORDS = {
    "idiootti", "haista", "pässi", "täysi pelle", "tapa ittes", "skibidi", "retardi", "ignore my previous", "midget", "kääpiö", "kalju", "bald"
}
KICK_WHITELIST = {"pyr00z"}
VIOLATION_LOG_PATH = "violations.log"
VIOLATION_CSV_PATH = "violations.csv"
CONCURRENT_TASKS = 4


# === LOAD MODELS ===
print("Loading Whisper model...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE).to(DEVICE)

print("Loading AI toxicity detection model...")
toxicity_model_name = "christinacdl/XLM_RoBERTa-Offensive-Language-Detection-8-langs-new" #This was pretty much the only option that somewhat supports finnish. I also requested and was granted access to google perspective api, but for now I dont want to use it.
tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
tox_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name).to(DEVICE)

# === MEMORY FOR SEEN FILES ===
PROCESSED_FILES = set()

def is_rule_violation(text):
    text = text.lower()
    for word in RULE_BASED_BADWORDS:
        if word in text:
            return True, word
    return False, None

def is_offensive(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True). to(DEVICE)
    with torch.no_grad():
        outputs = tox_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        offensive_score = probs[0][1].item()
    return offensive_score >= TOXICITY_THRESHOLD, offensive_score

def is_audio_loud(filepath, threshold=500): #500 is pretty solid for normal speaking volume in ts3
    try:
        samplerate, samples = wavfile.read(filepath)
        if len(samples.shape) > 1:
            samples = samples.mean(axis=1)
        volume = np.abs(samples).mean()
        return volume > threshold
    except:
        return False

def extract_nickname_from_filename(speaker):
    try:
        parts = speaker.split("_")
        if len(parts) >= 2:
            raw_nick = parts[1]
            return raw_nick.encode("latin1").decode("utf-8")
    except Exception:
        pass
    return speaker  # fallback if decoding fails or format unexpected

def try_delete_file(filepath, retries=5, delay=2):
    for i in range(retries):
        try:
            os.remove(filepath)
            print(f"Deleted file: {filepath}")
            return True
        except Exception as e:
            time.sleep(delay)
    print(f"Failed to delete file after retries: {filepath}")
    return False

async def process_file(filepath, speaker, session):
    file_id = f"{session}/{os.path.basename(filepath)}"
    if file_id in PROCESSED_FILES:
        return

    try:
        result = whisper_model.transcribe(filepath, language="fi", fp16=(DEVICE == "cuda"))
        text = result["text"].strip()

        if not text:
            PROCESSED_FILES.add(file_id)
            return

        print(f"{speaker}: {text}")

        is_bad_ai, score = is_offensive(text)
        is_bad_rule, matched_word = is_rule_violation(text)

        if is_bad_ai or is_bad_rule:
            if is_bad_rule:
                violation_reason = f"Rule Violation: {matched_word}"
                print(f"!!! PMA Violation (word: '{matched_word}')")
            elif is_bad_ai:
                violation_reason = f"Toxicity Score: {score:.2f}, {text}"
                print(f"!!! PMA Violation (AI score: {score:.2f})")

            log_violation(speaker, text, violation_reason)
            log_violation_csv(speaker, text, violation_reason, session)

            username = extract_nickname_from_filename(speaker)
            if username not in KICK_WHITELIST:
                await clientquery_kick(username, violation_reason)

        # Mark as processed
        PROCESSED_FILES.add(file_id)

        # Try to delete the file
        try:
            if try_delete_file(filepath):
                parent = os.path.dirname(filepath)
                if not os.listdir(parent):
                    os.rmdir(parent)
                    print(f"Deleted empty session folder: {parent}")
            print(f"Deleted file: {filepath}")
            parent = os.path.dirname(filepath)
            if not os.listdir(parent):
                os.rmdir(parent)
                print(f"Deleted empty session folder: {parent}")
        except Exception as delete_err:
            print(f"Failed to delete file or folder: {delete_err}")

    except Exception as e:
        print(f"Failed to process file: {e}")


def normalize(s):
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.lower()

def ts3_unescape(s):
    return s.replace("\\s", " ").replace("\\p", "|").replace("\\/", "/").replace("\\\\", "\\")

def extract_nickname(entry):
    match = re.search(r"client_nickname=([^\s]+)", entry)
    if match:
        return ts3_unescape(match.group(1))
    return None

async def clientquery_kick(nickname, reason="PMA Violation"):
    try:
        reader, writer = await telnetlib3.open_connection('127.0.0.1', 25639)
        await reader.readuntil(b"TS3 Client")

        # Authenticate
        writer.write(f"auth apikey={API_KEY}\n")
        await reader.readuntil(b"error id=0")

        # Request client list
        writer.write("clientlist\n")
        data = ""
        while "error id=0" not in data:
            data += await reader.read(4096)

        print(f"Trying to match: '{nickname}'")
        found_match = False

        entries = data.split("|")
        for entry in entries:
            actual_nick = extract_nickname(entry)
            if actual_nick:
                print(f" → {actual_nick}")
            if actual_nick and normalize(nickname) in normalize(actual_nick):
                match = re.search(r"clid=(\d+)", entry)
                if match:
                    clid = match.group(1)
                    sanitized_reason = reason.replace(" ", "_").replace(":", "").replace("'", "")
                    writer.write(f"clientkick clid={clid} reasonid=5 reasonmsg={sanitized_reason}\n")
                    await reader.read(4096)
                    print(f"Kicked {actual_nick} for {reason}")
                    found_match = True
                else:
                    print(f"Could not extract clid for {actual_nick}")
                break

        if not found_match:
            print(f"{nickname} not found in clientlist.")

        writer.close()

    except Exception as e:
        print(f"ClientQuery kick failed: {e}")

def extract_nickname_from_filename(filename):
    try:
        base = os.path.basename(filename)
        name_part = base.split("_")[1]
        # Try to recover the original name
        return name_part.encode("latin1").decode("utf-8")
    except Exception:
        return name_part  # fallback if decoding fails

def log_violation(speaker, text, reason):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(VIOLATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {speaker} - {reason}\n")
        f.write(f"  \"{text}\"\n\n")

def log_violation_csv(speaker, text, reason, session):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, speaker, reason, text, session]

    # Write header if file doesn't exist yet
    write_header = not os.path.exists(VIOLATION_CSV_PATH)

    with open(VIOLATION_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "speaker", "reason", "text", "session"])
        writer.writerow(row)

async def main_loop():
    print("PMA Paladin is standing by...\n")
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)

    while True:
        try:
            tasks = []

            for session_folder in os.listdir(RECORDING_DIR):
                session_path = os.path.join(RECORDING_DIR, session_folder)
                if not os.path.isdir(session_path):
                    continue

                for filename in os.listdir(session_path):
                    if filename.lower().endswith(".wav"):
                        filepath = os.path.join(session_path, filename)
                        speaker = os.path.splitext(filename)[0]

                        async def limited_task(fp=filepath, sp=speaker, sess=session_folder):
                            async with semaphore:
                                await process_file(fp, sp, sess)

                        tasks.append(limited_task())

            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Loop error: {e}")

        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main_loop())