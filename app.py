import os
import tempfile
import queue
import threading
import speechmatics
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, jsonify, Response, request, render_template
import scipy.io.wavfile as wav
import os, tempfile
from flask_cors import CORS


CHANNELS = 1
RATE = 44100



audio_queue = queue.Queue()
result_queue = queue.Queue()
complete_transcript = ""
single_line_transcript = ""
stop_transcription = False




load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


SPEECHMATIC_API_KEY = "FkveCrAfHrMHWzAKUNYaGexvBfdgdNWk"
CONNECTION_URL = "wss://eu2.rt.speechmatics.com/v2"

ws = speechmatics.client.WebsocketClient(
    speechmatics.models.ConnectionSettings(
        url=CONNECTION_URL,
        auth_token=SPEECHMATIC_API_KEY,
    )
)

def print_transcript_of_speechmatic(msg):
    global complete_transcript
    global single_line_transcript
    transcript_piece = msg['metadata']['transcript']
    complete_transcript += transcript_piece
    single_line_transcript += transcript_piece
    # print(f"Transcript: {transcript_piece}")

ws.add_event_handler(
    event_name=speechmatics.models.ServerMessageType.AddTranscript,
    event_handler=print_transcript_of_speechmatic,
)

settings = speechmatics.models.AudioSettings()
conf = speechmatics.models.TranscriptionConfig(
    language="en",
    enable_partials=True,
    max_delay=1,
)




def transcribe_audio_for_whisper(MODEL, LANGUAGE,TEMPERATURE):
    global complete_transcript, stop_transcription, result_queue
    while not stop_transcription:
        audio_file = None
        try:
            audio_file = audio_queue.get(timeout=1)
            file_name = os.path.basename(audio_file)
            # print(f"*****File name: {file_name} ********")
            
            if audio_file is None:
                print("************ No audio file received for transcribing ******************.") 
            with open(audio_file, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model=MODEL,
                    file=f,
                    language=LANGUAGE,
                    temperature=TEMPERATURE,
                )
                text = transcript.text.strip()
                print("Transcript:", text)
                complete_transcript += text + " "
                result_queue.put(text)
                if stop_transcription:  
                    break
        except queue.Empty:
            continue  
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
        finally:
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
            if audio_file is not None:
                audio_queue.task_done()


   
def transcribe_audio_for_speechmatic():
    global complete_transcript, stop_transcription,single_line_transcript
    while not stop_transcription:
        audio_file = None
        try:
            audio_file = audio_queue.get(timeout=1) 
            with open(audio_file, 'rb') as aud_file:
                ws.run_synchronously(aud_file, conf, settings)
                print(single_line_transcript)
                result_queue.put(single_line_transcript)
                single_line_transcript = ""
                if stop_transcription:  
                    break
        except queue.Empty:
            continue  
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
        finally:
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
            if audio_file is not None:
                audio_queue.task_done()


def system_prompts(transcript):
    
    prompt = f"""
    here is a transcript= {transcript}:
    You are a highly intelligent AI assistant. Your task is to convert the given 'transcript' into a clear and properly formatted dialogue between a Patient and a Doctor. Carefully analyze the 'transcript' to correctly identify which parts of the text belong to the Patient and which belong to the Doctor.

    Important Rules:
    Ensure Patient statements are assigned to the Patient, and Doctor statements to the Doctor — no mix-ups.
    The output should reflect the original transcription precisely, only changing the format to a labeled dialogue.

    Output Example:
    Doctor: [Doctor's line]
    Patient: [Patient's line]
    """
    return prompt

def user_system_prompts(complete_transcript , user_prompt):
    
    prompt = f"""
        here is a transcript= {complete_transcript}:
        here is a user_prompt ={user_prompt}:
        according to user_prompt, you are a highly intelligent AI assistant. Your task is to convert the given 'transcript' into a clear and properly formatted dialogue between a Patient and a Doctor. 
        
        Output Example:
        Doctor: [Doctor's line]
        Patient: [Patient's line]
        """
    return prompt


def dialogue_transcript(prompt):
    try:
        response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
            ],
            temperature=0,
        )
        print("\nResponse:\n", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during dialogue transcription: {str(e)}")
        return ""


app = Flask(__name__)
CORS(app)



@app.route('/')
def index():
    return render_template('test.html')



@app.route('/upload_audio', methods=['POST'])
def upload_audio():

    global audio_queue
    audio = request.files.get('audio_data')

    if not audio:
        return jsonify({'error': 'No audio_data provided'}), 400

    # Save WAV blob to disk
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio.save(tmp)
        audio_queue.put(tmp.name)

    return jsonify({'status': 'received'}), 200




@app.route('/transcribe', methods=['POST'])
def start_transcription():
  
    global stop_transcription, complete_transcript
    stop_transcription = False
    complete_transcript = ""

    transcriber = request.json['transcriber']
    if transcriber == 'speechmatics':
        threading.Thread(
            target=transcribe_audio_for_speechmatic,
            daemon=True
        ).start()
    else:
        MODEL = request.json['model']
        LANGUAGE = request.json['language']
        TEMP = request.json['temperature']
        threading.Thread(
            target=transcribe_audio_for_whisper,
            args=(MODEL, LANGUAGE, TEMP),
            daemon=True
        ).start()

    return jsonify({'status':'started'})


@app.route('/transcribe_stream', methods=['GET'])
def stream_transcription():
    def event_stream():
        while not stop_transcription:
            try:
                text = result_queue.get(timeout=2)
                yield f"data: {text}\n\n"
                result_queue.task_done()
            except queue.Empty:
                yield f"data: ...\n\n"
    return Response(event_stream(), content_type='text/event-stream')



@app.route('/transcription_result', methods=['GET'])
def get_transcription_result():
    global stop_transcription, complete_transcript, audio_queue
    stop_transcription = True  
    return jsonify({"transcription": complete_transcript.strip()}), 200


@app.route('/dialogue_transcript', methods=['GET'])
def get_dialuge_transcript():
    user_system_prompt= request.args.get('user_system_prompt')
    global complete_transcript
    if not complete_transcript:
        return jsonify({"dialogue": "..."}), 400
    if user_system_prompt:
        prompt = user_system_prompts(complete_transcript, user_system_prompt)
        dialogue = dialogue_transcript(prompt)
        return jsonify({"dialogue": dialogue.strip()}), 200

    
    prompt = system_prompts(complete_transcript)
    dialogue = dialogue_transcript(prompt)
    return jsonify({"dialogue": dialogue.strip()}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, ssl_context=('cert.pem', 'key.pem'))
    # app.run( debug=True, threaded=True)





