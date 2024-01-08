import pyautogui
import time
n = 0
time.sleep(0)
while(n < 2):
    screenshot = pyautogui.screenshot()
    sc = "liveImage" + str(n) +(".rgba")
    screenshot.save("img/"+sc)
    time.sleep(10)
    n = n+1
