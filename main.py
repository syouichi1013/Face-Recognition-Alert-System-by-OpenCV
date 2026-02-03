import os
import cv2 as cv
import numpy as np
import requests
import json

# WeChat Work robot Webhook (for sending alert messages)
WECHAT_WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2fa3737b-6e42-44f2-a8b2-fea253394f"#modify this URL to yourself's webhook

# Face recognition configuration parameters
model_path='trainer/trainer.yml'
face_detector_path='haarcascade_frontalface_default.xml'
confidence_threshold=80# Threshold for face recognition confidence (lower = more confident)

# Define function to send alert to WeChat Work group via webhook
def send_wechat_alert():
    try:
        # Construct alert message
        alert_msg = {
            "msgtype": "text",
            "text": {
                "content": "[Alert] Unknown person detected"
            }
        }
        # Send POST request to WeChat Work webhook
        response = requests.post(
            WECHAT_WEBHOOK,
            data=json.dumps(alert_msg),
            headers={"Content-Type": "application/json"}
        )
        result = response.json()
        if result.get("errcode") == 0:
            print("Alert message sent to WeChat Work successfully")
        else:
            print(f"Failed to send alert：{result.get('errmsg')}")
    except Exception as e:
        print(f"Error occurred while sending alert：{str(e)}")

# Core function: Real-time face recognition via camera + unknown person alert
def camera_recognize():
    recognizer=cv.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    face_detector=cv.CascadeClassifier(face_detector_path)
    cap=cv.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    alert_sent = False  # Flag to prevent repeated alert sending

    while cap.isOpened():
        ret,img=cap.read()
        if not ret:
            break
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        faces=face_detector.detectMultiScale(gray,1.1,6)
        has_unknown_face = False

        for (x,y,w,h) in faces:
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            id,conf=recognizer.predict(gray[y:y+h,x:x+w])

            if conf>confidence_threshold:
                cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv.putText(img,f"no{conf:.0f}", (x+5, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                has_unknown_face = True

            else:
                cv.putText(img, f"yes{conf:.0f}", (x + 5, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                has_unknown_face =False

        if has_unknown_face == True and alert_sent == False:
            send_wechat_alert()
            alert_sent = True

        cv.imshow("Camera Recognition", img)
        k = cv.waitKey(1) & 0xFF
        if k==ord('s'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    camera_recognize()