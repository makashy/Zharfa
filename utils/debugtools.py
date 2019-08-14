from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen 



#Prints mouse location######################↓  
class MainScreen(Screen):
    def on_mouse_pos(self, pos):
        print(pos)

    Window.bind(mouse_pos = on_mouse_pos)

class MyScreenManager(Screen):
    pass
#############################################↑
