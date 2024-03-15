import pyautogui
import time
from PIL import Image
class FindMenu:
    def __init__(self):
        self._find_window()
        self.loc = None

    def show_image(self):
        loc = self.loc
        # 擷取指定位置的螢幕影像
        screenshot = pyautogui.screenshot(region=(int(loc["left"]), int(loc["top"]), int(loc["width"]), int(loc["height"])))

        # 顯示影像
        screenshot.show()

    def _find_window(self):
        window = pyautogui.getWindowsWithTitle('Hollow Knight')                                 # 找到Hollow Knight名字的視窗
        assert len(window) == 1, f'found {len(window)} windows called Hollow Knight {window}'   # 確認是否找到
        window = window[0]
        try:
            window.activate()           # 激活他
        except Exception:
            window.minimize()
            window.maximize()
            window.restore()
        window.moveTo(0, 0)

        geo = None
        conf = 0.9995
        while geo is None:
            geo = pyautogui.locateOnScreen('./locator/geo.png', confidence=conf)
            print(geo)
            conf = max(0.92, conf * 0.999)
            time.sleep(0.1)
        loc = {
            'left': geo.left - 36,
            'top': geo.top - 97,
            'width': 1020,
            'height': 692
        }
        self.loc = loc
        # self.show_image()

        return loc

        # def _find_menu():
        #     """
        #     locate the menu badge,
        #     when the badge is found, the correct game is ready to be started
        #
        #     :return: the location of menu badge
        #     """
        #     monitor = self.monitor
        #     # print(monitor)
        #     monitor = (monitor['left'] + monitor['width'] // 2,
        #                monitor['top'] + monitor['height'] // 4,
        #                monitor['width'] // 2,
        #                monitor['height'] // 2)
        #     return pyautogui.locateOnScreen(f'locator/attuned.png',
        #                                     region=monitor,
        #                                     confidence=0.925)
def main():
   test = FindMenu()
if __name__ == '__main__':
    main()