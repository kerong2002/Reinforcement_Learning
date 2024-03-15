# 2024_03_04 kerong
# keyboard
import tkinter as tk
from pynput.keyboard import Listener, KeyCode

KEYBOARD_WIDTH = 900
KEYBOARD_HEIGHT = 280


def on_press(key):
    try:
        key_char = key.char
    except AttributeError:
        key_char = key.name

    for button_row in keyboard_buttons:
        for button in button_row:
            if button["text"] == key_char:
                button.config(bg="red")


def on_release(key):
    try:
        key_char = key.char
    except AttributeError:
        key_char = key.name

    for button_row in keyboard_buttons:
        for button in button_row:
            if button["text"] == key_char:
                button.config(bg="black")


def create_keyboard_button(parent, text, width, height):
    button = tk.Button(parent, text=text, width=width, height=height, bg="black", fg="white")
    return button


window = tk.Tk()
window.title("Virtual Keyboard")
window.geometry(f"{KEYBOARD_WIDTH}x{KEYBOARD_HEIGHT}")
window.configure(bg="black")

keyboard_keys = [
    ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'Backspace'],
    ['Tab', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
    ['Caps Lock', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\'', 'Enter'],
    ['Shift', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 'Shift'],
    ['Ctrl', 'Win', 'Alt', 'space', 'Alt', 'Win', 'Menu', 'Ctrl']
]

special_keys = {'Tab': 1.5, 'Caps Lock': 1.8, 'Shift': 2.497, 'Backspace': 2, 'space': 5, "\\": 1.49, "Enter": 2.2,
                "Ctrl": 1.5, "Win": 1.4, "Alt": 1.4, "Menu": 1.4}

default_button_width = KEYBOARD_WIDTH // max(len(row) for row in keyboard_keys) - 10
default_button_height = KEYBOARD_HEIGHT // len(keyboard_keys)

keyboard_buttons = []

for row_index, row in enumerate(keyboard_keys):
    button_row = []
    prev_x = 0
    prev_y = row_index * default_button_height
    for col_index, key_text in enumerate(row):
        button_width = default_button_width * special_keys.get(key_text, 1)
        button = create_keyboard_button(window, key_text, int(button_width / 10), 2)
        button.place(x=prev_x, y=prev_y, width=button_width, height=default_button_height)
        button_row.append(button)
        prev_x += button_width
    keyboard_buttons.append(button_row)

# Start listening for keyboard events
with Listener(on_press=on_press, on_release=on_release) as listener:
    window.mainloop()
