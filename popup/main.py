"""
Entry point for the popup notification system.
"""
import sys
import threading
import time
from PySide6.QtWidgets import QApplication
from popup_window import PopupWindow


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
                    title="Error",
                    message="Something went wrong!\nPlease check the logs.",
                    message_type="e"
                )
            elif cmd == 'w':
                popup.emit_message(
                    title="Warning",
                    message="Low disk space detected.",
                    message_type="w"
                )
            elif cmd == 'c':
                popup.emit_message(
                    title="Success",
                    message="Operation completed!",
                    message_type="c"
                )
            elif cmd == 'n':
                popup.emit_message(
                    title="Notice",
                    message="New update available.",
                    message_type="n"
                )
            elif cmd == 'p':
                popup.emit_message(
                    title="Info",
                    message="Low priority (0)",
                    message_type="n"
                )
                time.sleep(0.1)
                popup.emit_message(
                    title="Warning",
                    message="Medium priority (5)",
                    message_type="w"
                )
                time.sleep(0.1)
                popup.emit_message(
                    title="Error",
                    message="High priority (10)",
                    message_type="e"
                )
                time.sleep(0.1)
                popup.emit_message(
                    title="CRITICAL",
                    message="Should interrupt!",
                    message_type="e"
                )
    
    trigger_thread = threading.Thread(target=wait_for_trigger, daemon=True)
    trigger_thread.start()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()