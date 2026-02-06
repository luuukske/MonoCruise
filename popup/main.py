"""
Entry point for the popup notification system.
"""
import sys
import threading
import time
from PyQt6.QtWidgets import QApplication
from popup_window import PopupWindow
from message_types import MessageStyle


def main():
    app = QApplication(sys.argv)
    
    popup = PopupWindow()
    
    def wait_for_trigger():
        while True:
            cmd = input("\nCommand (e=error, w=warning, c=check, n=notice, p=priority test, q=quit): ").strip().lower()
            
            if cmd == 'q':
                app.quit()
                break
            elif cmd == 'e':
                popup.emit_message(
                    text="Error: Something went wrong!\nPlease check the logs.",
                    style=MessageStyle.ERROR,
                    duration_ms=5000,
                    priority=10
                )
            elif cmd == 'w':
                popup.emit_message(
                    text="Warning: Low disk space detected.",
                    style=MessageStyle.WARNING,
                    duration_ms=4000,
                    priority=5
                )
            elif cmd == 'c':
                popup.emit_message(
                    text="Success: Operation completed!",
                    style=MessageStyle.CHECK,
                    duration_ms=3000,
                    priority=1
                )
            elif cmd == 'n':
                popup.emit_message(
                    text="Notice: New update available.",
                    style=MessageStyle.NOTICE,
                    duration_ms=3000,
                    priority=0
                )
            elif cmd == 'p':
                popup.emit_message(
                    text="1. Low priority (0)",
                    style=MessageStyle.NOTICE,
                    duration_ms=3000,
                    priority=0
                )
                time.sleep(0.1)
                popup.emit_message(
                    text="2. Medium priority (5)",
                    style=MessageStyle.WARNING,
                    duration_ms=3000,
                    priority=5
                )
                time.sleep(0.1)
                popup.emit_message(
                    text="3. High priority (10)",
                    style=MessageStyle.ERROR,
                    duration_ms=3000,
                    priority=10
                )
                time.sleep(0.1)
                popup.emit_message(
                    text="4. CRITICAL (100) - Should interrupt!",
                    style=MessageStyle.ERROR,
                    duration_ms=5000,
                    priority=100
                )
    
    trigger_thread = threading.Thread(target=wait_for_trigger, daemon=True)
    trigger_thread.start()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()