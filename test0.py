from picamera2 import Picamera2
from datetime import datetime
from PIL import Image
import time
import numpy as np
import tensorflow as tf
from gpiozero import Button, Buzzer

IMG_SIZE = (160, 160)
FACE_LABELS = ["known", "unknown"]
FACE_UNKNOWN_THRESHOLD = 0.75

BUTTON_PIN = 17
BUZZER_PIN = 18

button = Button(BUTTON_PIN, pull_up=False)
buzzer = Buzzer(BUZZER_PIN)

face_interpreter = tf.lite.Interpreter(model_path="doorbell_model.tflite")
face_interpreter.allocate_tensors()
face_input = face_interpreter.get_input_details()[0]
face_output = face_interpreter.get_output_details()[0]

def preprocess_for_mobilenet(img_pil):
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = x / 127.5 - 1.0
    x = np.expand_dims(x, axis=0)
    return x

def run_face_model(image_path):
    img = Image.open(image_path)
    x = preprocess_for_mobilenet(img)
    face_interpreter.set_tensor(face_input["index"], x)
    face_interpreter.invoke()
    preds = face_interpreter.get_tensor(face_output["index"])[0]
    idx = int(np.argmax(preds))
    label = FACE_LABELS[idx]
    conf = float(preds[idx])
    return label, conf, preds

def beep_known():
    buzzer.on()
    time.sleep(0.1)
    buzzer.off()

def beep_unknown():
    for _ in range(2):
        buzzer.on()
        time.sleep(0.2)
        buzzer.off()
        time.sleep(0.2)

def capture_and_infer(picam2):
    filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    picam2.capture_file(filename)

    img = Image.open(filename)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(filename)

    print(f"Captured: {filename}")

    label, conf, probs = run_face_model(filename)

    print(f"[FACE] probs={probs}")
    print(f"[FACE] predicted={label}  conf={conf:.2f}")

    if label == "known" and conf >= FACE_UNKNOWN_THRESHOLD:
        print("Final decision: RESIDENT_KNOWN")
        beep_known()
    else:
        print("Final decision: UNKNOWN_VISITOR")
        beep_unknown()

if __name__ == "__main__":
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (1920, 1080)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    print("System ready. Press the button to capture.")

    try:
        while True:
            button.wait_for_press()
            print("\nButton pressed. Running face check...")
            capture_and_infer(picam2)
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        buzzer.off()
