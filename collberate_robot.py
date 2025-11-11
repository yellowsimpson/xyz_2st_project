import os
import openai
import rospy
from std_msgs.msg import String
import speech_recognition as sr

# 음성 인식
r = sr.Recognizer()
with sr.Microphone() as source:
    print("명령을 말씀해주세요:")
    audio = r.listen(source)
    command_text = r.recognize_google(audio, language="ko-KR")

# 자연어 명령 해석 (LLM)
openai.api_key = os.getenv("OPENAI_API_KEY")  # 환경변수 사용 권장

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": f"로봇 팔에게 수행할 명령으로 변환: {command_text}"
        }
    ]
)
robot_command = response.choices[0].message["content"]

# 로봇 제어 명령 발행 (ROS)
rospy.init_node('cobot_controller')
pub = rospy.Publisher('/robot_command', String, queue_size=10)

msg = String()
msg.data = robot_command
pub.publish(msg)

print("로봇 명령:", robot_command)
