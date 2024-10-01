import speech_recognition as sr
from gtts import gTTS
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
import cv2
from ultralytics import YOLO
import serial
import threading
import pygame
import tempfile

load_dotenv()

API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=API_KEY)

# 음성 인식기 객체 생성
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# pygame 초기화 (음성 재생용)
pygame.mixer.init()

# TTS 함수 정의
def speak(text):
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts = gTTS(text=text, lang='ko')
        tts.save(temp_file.name)
        pygame.mixer.music.load(temp_file.name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"음성 생성 오류: {e}")
    finally:
        pygame.mixer.music.unload()
        if temp_file:
            temp_file.close()
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"임시 파일 삭제 오류: {e}")

# YOLO 모델 및 카메라 설정
model = YOLO('/home/user/Desktop/fin/best.onnx')
print("모델 로딩 완료")

print("카메라 초기화")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다")
    exit()
print("카메라 초기화 완료")



# 아두이노 설정
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)
    print("아두이노 연결 성공")
except:
    print("아두이노 연결 실패")
    arduino = None



# 아두이노 모터 제어 함수
def down():
    if arduino:
        arduino.write(b'0')
    print('down')
    time.sleep(1)
    mid()

def mid():
    if arduino:
        arduino.write(b'1')

def up():
    if arduino:
        arduino.write(b'2')
    print('up')
    time.sleep(1)
    mid()



# 전역 변수 및 동기화 객체
detection_pause = threading.Event()
detection_pause.set()  # 객체 감지 활성화



# 객체 감지 함수
def detect_objects():
    prev_time = time.time()
    FPS = 0.65
    print("객체 감지 시작")

    while True:
        if detection_pause.is_set():
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다")
                continue

            if time.time() - prev_time > 1/FPS:
                frame = cv2.resize(frame, (640, 480))
                results = model(frame, conf=0.3)

                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        conf = box.conf.item()
                      # 정확도가 0.5 이상일때 
                        if conf >= 0.5:
                          # 모델 config에서 0 : 캔, 1 : 종이, 2 : 플라스틱
                            if cls == 0:
                                print("캔 감지")
                                speak('캔 입니다')
                                down()
                            elif cls == 1:
                                print("종이 감지")
                                speak('종이 입니다')
                                up()
                            elif cls == 2:
                                print("플라스틱 감지")
                                speak('플라스틱 감지')
                                up()


                prev_time = time.time()

        else:
            print("객체 감지 일시 중지")
            time.sleep(1)



def respond(source):
    detection_pause.clear()  # 객체 감지 일시 중지
    print("객체 감지가 일시 중지되었습니다.")

    recognizer.adjust_for_ambient_noise(source)

    try:
        audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)  # Reduced times for faster recognition
        text2 = recognizer.recognize_google(audio, language="ko-KR")
        print(f"인식된 음성 : {text2}")

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 쓰레기 분리수거 최고 전문가야. 한국어로 설명하고 최대한 한 줄 이내로 설명해줘"},
                {"role": "user", "content": text2}
            ]
        )
        response_content = response.choices[0].message.content
        if response_content:
            speak(response_content)
            print(response_content)
    except sr.WaitTimeoutError:
        print("음성 입력시간 초과")
    except sr.RequestError as e:
        print(f"Google TTS 오류 {e}")
    finally:
        detection_pause.set()
        print("객체 검자 재개")



def listen_for_keyword():
    with sr.Microphone() as source:
        print("'쓰레기통' 키워드 기다리는중...")
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)  # Reduced times for faster recognition
                text1 = recognizer.recognize_google(audio, language="ko-KR")
                print(f"인식된 텍스트 : {text1}")

                if "쓰레기통" in text1:
                    speak("네 말씀해 주세요")
                    respond(source)

            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Google TTS 오류 {e}")

if __name__ == "__main__":
    object_detection_thread = threading.Thread(target=detect_objects)
    voice_recognition_thread = threading.Thread(target=listen_for_keyword)

    object_detection_thread.start()
    voice_recognition_thread.start()

    try:
        object_detection_thread.join()
        voice_recognition_thread.join()
    except KeyboardInterrupt:
        print("프로그램 종료")
    finally:
        detection_pause.clear() 
        object_detection_thread.join()
        voice_recognition_thread.join()

    print("모든 쓰레드가 종료되었습니다.")
