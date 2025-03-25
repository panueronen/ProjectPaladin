# PMA Paladin

PMA Paladin is an AI-powered voice moderation bot built for TeamSpeak 3.  
It transcribes what users say in multitrack TS3 recordings, detects toxic speech, and kicks users from the server during "PMA Week" — a self-imposed event where only friendly and respectful communication is allowed.

This project is not intended for public use or distribution. It exists to demonstrate a complete and unconventional system integration: speech recognition, toxicity detection, automation, and live client interaction in a real voice platform. 
It was mainly created as a joke but it got taken so far that I am putting it here if someone wants to build a similar system.

---

## What It Does

- Transcribes audio recordings (multitrack `.wav` files per user)
- Detects toxic or banned phrases using both:
  - A simple rule-based word filter
  - An XLM-RoBERTa-based multilingual offensive language classifier
- Logs violations to a structured CSV file
- Automatically kicks violators via TeamSpeak ClientQuery interface
- Deletes processed files to avoid redundant work
- Operates continuously, scanning new session folders

---

## Why These Choices

**Whisper (OpenAI):**  
Used for automatic speech recognition because of its multilingual support and ability to run entirely offline. Finnish is (somewhat) supported out of the box, which is essential for this use case.

**XLM-RoBERTa Offensive Detection Model:**  
Chosen for its support of multiple languages, including Finnish, and its reasonable performance on short spoken phrases. It's compact enough to run locally without relying on cloud services.
To expand on this. I requested and was granted access to googles Perspective API, but ended up not using because I did not want to distribute everything spoken on our server to google. Open Ais API was also considered because of its support for multiple languages and even "online slang", however it was quite pricy just for a joke project

**Autohotkey (AHK) for Recording Control:**  
TeamSpeak does not offer an API to control when recordings start and stop. AHK is used to simulate the hotkey press that toggles recording. This allows automated 30-second recording intervals, ensuring short audio files that are quick to transcribe.
This works quite well. Just need to find 2 buttons that are generally not used for anything smart and the delay can be adjusted from the AHK script. A delay between the stopping and starting new recording is needed or ts3 does no begin a new recording. 
So it is a weakness of this system that theoretically someone could say something between the recordings. TS3 also plays a sound to notify people that someone has started recording and it cannot be disabled on server so everyone has to personally turn it off from notification settings

**ClientQuery API:**  
I am renting my server from an outside host that does not allow direct access to the serverquery from outside so the only way to interact with the running client is through the built-in local telnet interface (ClientQuery). This allows sending commands such as `clientkick` to remove users live.

**Multitrack Recording:**  
Using TeamSpeak’s multitrack mode allows each user’s audio to be transcribed independently, which greatly improves transcription accuracy and helps associate speech with the correct speaker. 
It would maybe be possible to train an AI to detect everyones speech individually from a single file, but that would be a lot of work and it would open a door to issues like someone recording other persons speech and playing that from a soundboard

**Volume Filtering:**  
The bot checks for low-volume audio and skips those files. This avoids wasting time on silence or background noise that Whisper would otherwise attempt to transcribe.
This was a lot less necessary after I moved the AI models to run on GPU but before that the processing was very slow and whatever performance gains I could make were absolutely necessary.
This could theoretically be used to prevent loud noises in the future

**Parallel Processing:**  
Implemented using asyncio and semaphores. Since Whisper can run on GPU, multiple transcriptions can be processed in parallel for better throughput.
This was maybe the second biggest improvement after switching to using cuda.

---

## Dependencies

- Python 3.10+
- openai-whisper
- torch with CUDA support for GPU use
- transformers
- scipy, numpy
- telnetlib3
- python-dotenv
- Autohotkey (for Windows recording automation)

---

## Running the Bot

1. Set up TeamSpeak 3 with multitrack recording enabled and setup the start and stop recording buttons. (make sure you are using the multitrack recording)
2. Enable and configure the ClientQuery plugin
3. Run the AHK script to toggle recording every 30 seconds
4. Start the PMA bot via python pmaBot.py
5. Violations will be printed and logged; users will be kicked if necessary

---

## Notes

- The bot assumes the user running it has Admin privileges and ClientQuery enabled locally.
- The script is designed to monitor local recording folders. It was not designed to be run on a remote server.
- While this could be adapted to other platforms, it is tightly integrated with TeamSpeak’s specific quirks and tools.

---


This project is provided as-is
It is published for demonstration and portfolio purposes only.  
Usage in real communities should be done with caution and consent.

