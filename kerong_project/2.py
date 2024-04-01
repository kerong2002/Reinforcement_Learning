# 2024_03_14 kerong
# hw2
from tkinter import *
import threading
import time

w = 400
h = 400
r = 80
window = Tk() #建置容器

canva = Canvas(window, bd=10, bg="Darkgray", height=400, width=400) 
canva.pack()

up_down = 0
start_num = 30
extent_num = 300
running = 1

def draw():
    global up_down, start_num, extent_num, running
    while running:
        canva.delete('all')  # 清除畫布上的所有元素
        canva.create_arc(200+r, 200+r, 200-r, 200-r, start=start_num, extent=extent_num, width=10, fill='yellow')
        change()
        window.update()  # 更新畫布
        time.sleep(0.03)

def change():
    global up_down, start_num, extent_num
    if up_down == 0:
        extent_num -= 4
        start_num += 2
        if start_num == 40:
            up_down = 1
    else:
        extent_num += 4
        start_num -= 2
        if start_num == 2:
            up_down = 0

def on_closing():
    global running, t1
    running = 0
    window.destroy()
    t1.join()  # 等待線程結束
    

window.protocol("WM_DELETE_WINDOW", on_closing) 

t1 = threading.Thread(target=draw)
t1.start()

window.mainloop()
