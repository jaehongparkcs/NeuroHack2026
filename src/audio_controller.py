import subprocess

class AudioController:
    """Handles all Apple Music interactions via macOS AppleScript."""
    
    @staticmethod
    def get_current_song():
        try:
            is_running = subprocess.run(['osascript', '-e', 'application "Music" is running'], capture_output=True, text=True).stdout.strip()
            if is_running == 'true':
                track = subprocess.run(['osascript', '-e', 'tell application "Music" to name of current track'], capture_output=True, text=True).stdout.strip()
                artist = subprocess.run(['osascript', '-e', 'tell application "Music" to artist of current track'], capture_output=True, text=True).stdout.strip()
                if track and artist:
                    return f"{track} - {artist}"
        except Exception:
            pass
        return "None / Silence"

    @staticmethod
    def pause_music():
        try:
            subprocess.run(['osascript', '-e', 'tell application "Music" to pause'])
        except Exception:
            pass

    @staticmethod
    def play_music():
        try:
            subprocess.run(['osascript', '-e', 'tell application "Music" to play'])
        except Exception:
            pass

    @staticmethod
    def set_music_volume(target_level):
        applescript = f"""
        tell application "Music"
            set targetVol to {target_level}
            set currentVol to sound volume
            if currentVol < targetVol then
                repeat while currentVol < targetVol
                    set currentVol to currentVol + 1
                    if currentVol > targetVol then set currentVol to targetVol
                    set sound volume to currentVol
                    delay 0.1
                end repeat
            else if currentVol > targetVol then
                repeat while currentVol > targetVol
                    set currentVol to currentVol - 1
                    if currentVol < targetVol then set currentVol to targetVol
                    set sound volume to currentVol
                    delay 0.1
                end repeat
            end if
        end tell
        """
        try:
            subprocess.Popen(['osascript', '-e', applescript])
        except Exception:
            pass