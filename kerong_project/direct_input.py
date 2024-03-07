import pyautogui
  
ACTION = {  # map each action to its corresponding key to press
    Move.HOLD_LEFT: 'a',
    Move.HOLD_RIGHT: 'd',
    # Move.LOOK_LEFT: 'a',
    # Move.LOOK_RIGHT: 'd',
    Displacement.TIMED_SHORT_JUMP: 'space',
    Displacement.TIMED_LONG_JUMP: 'space',
    # Displacement.DASH: 'k',
    Attack.ATTACK: 'j',
    # Attack.UP_ATTACK: ('w', 'j'),
    # Attack.SPELL: 'q'
}