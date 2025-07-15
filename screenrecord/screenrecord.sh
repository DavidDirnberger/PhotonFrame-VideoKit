#!/bin/bash

set -e

# 📁 Zielverzeichnis
output_dir="$HOME/data/Videos/Screenrecords"

# 📥 Mikrofonquelle bestimmen (smart)
get_mic_source() {
  echo "🎙️ Microphone sources available (PulseAudio):"
  IFS=$'\n' read -d '' -r -a mic_sources < <(pactl list short sources | grep input && printf '\0')

  mic_count=${#mic_sources[@]}

  if (( mic_count == 0 )); then
    echo "❌ No microphone source found"
    exit 1
  elif (( mic_count == 1 )); then
    mic_source=$(echo "${mic_sources[0]}" | awk '{print $2}')
    echo "✅ Automatically selected: $mic_source"
  else
    for i in "${!mic_sources[@]}"; do
      name=$(echo "${mic_sources[$i]}" | awk '{print $2}')
      echo "  [$((i+1))] $name"
    done

    read -rp "👉 Enter the number of your desired source: " selection
    if ! [[ "$selection" =~ ^[0-9]+$ ]] || (( selection < 1 || selection > mic_count )); then
      echo "❌ Invalid selection."
      exit 1
    fi
    mic_source=$(echo "${mic_sources[$((selection-1))]}" | awk '{print $2}')
    echo "✅ Elected: $mic_source"
  fi
}




fpath="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
infofile="$fpath/screenrecord.info"


# Infodatei anzeigen
if [[ "$1" == "--help" ]]; then
  cat $infofile
  exit 0
fi

# ❗ Benötigte Programme
REQUIRED_TOOLS=(xdotool xrandr ffmpeg)

# 📦 Prüfen und ggf. installieren
for tool in "${REQUIRED_TOOLS[@]}"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "❌ '$tool' is not installed."
    if command -v apt >/dev/null 2>&1; then
      echo "🔧 Try to install '$tool' ..."
      sudo apt update
      sudo apt install -y "$tool"
    else
      echo "❗ Automatic installation not possible - please install ‘$tool’ manually."
      exit 1
    fi
  fi
done

echo " "

# 🎯 Fensterposition des aufrufenden Terminals
win_id=$(xdotool getactivewindow)
eval "$(xdotool getwindowgeometry --shell "$win_id")"
term_x=$X
term_y=$Y

# 🖥 Monitorposition und Auflösung ermitteln
monitor_info=$(xrandr | grep ' connected')
monitor_found=false

while read -r line; do
  name=$(echo "$line" | awk '{print $1}')
  res_pos=$(echo "$line" | grep -o '[0-9]\+x[0-9]\++[0-9]\++[0-9]\+')
  [[ -z "$res_pos" ]] && continue

  res=$(echo "$res_pos" | cut -d+ -f1)
  pos_x=$(echo "$res_pos" | cut -d+ -f2)
  pos_y=$(echo "$res_pos" | cut -d+ -f3)

  width=$(echo "$res" | cut -dx -f1)
  height=$(echo "$res" | cut -dx -f2)

  if (( term_x >= pos_x && term_x < pos_x + width && term_y >= pos_y && term_y < pos_y + height )); then
    echo "✅ Terminal is on monitor: $name ($width x $height @ ${pos_x},${pos_y})"
    monitor_found=true
    break
  fi
done <<< "$monitor_info"

if ! $monitor_found; then
  echo "❌ Error: Could not determine the monitor position of the terminal."
  exit 1
fi


# Videoformat abfragen
echo
read -rp "📼 Which format to save in? [mp4/mkv] (Default: mp4): " video_format
video_format=${video_format,,}  # Kleinbuchstaben
if [[ -z "$video_format" ]]; then
  video_format="mp4"
fi

case "$video_format" in
  mp4|mkv)
    echo "💾 Recording format: $video_format"
    ;;
  *)
    echo "⚠️ Invalid format '$video_format'. Fallback: mp4"
    video_format="mp4"
    ;;
esac

# Audio-Auswahl
echo
echo "🔊 Audiorecord:"
echo "  [1] No Audio"
echo "  [2] Systemaudio (PulseAudio)"
echo "  [3] Microphone (useing ALSA/arecord for better quality)"
echo "  [4] Mix both (System + Microphone)"
echo " "
echo -ne "\033[1;33m      choose option [1-4]: \033[0m"
read -r audio_choice
echo " "

# stelle sicher das 📁 Zielverzeichnis existiert
mkdir -p "$output_dir"

# 📼 Dateiname mit Zeitstempel
timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
video_file="$output_dir/screencast_${timestamp}_video.$video_format"
sysaudio_file="$output_dir/screencast_${timestamp}_sysaudio.wav"
mic_file="$output_dir/screencast_${timestamp}_mic.wav"
final_file="$output_dir/screencast_${timestamp}.$video_format"

echo "Screencast will be saved in = '$final_file'"
echo " "

if [ $audio_choice -gt 2 ];then
# Mikrofonlautstärke auf 35 % setzen
pactl set-source-volume alsa_input.pci-0000_05_00.6.analog-stereo 50%
fi

# Audioaufnahme vorbereiten (PulseAudio-basiert)
mic_source=$(pactl list short sources | grep input | awk '{print $2}' | head -n 1)
monitor_source=$(pactl list short sources | grep monitor | awk '{print $2}' | head -n1)


case "$audio_choice" in
  2)
    echo "🎧 Recording system audio (PulseAudio)..."
    ffmpeg -hide_banner -loglevel warning -f pulse -i "$monitor_source" "$sysaudio_file" &
    SYS_PID=$!
    ;;
  3)
    get_mic_source
    echo "🎤 Microphone recording (parec)..."
    parec --device="$mic_source" --format=s16le --rate=44100 --channels=2 2>/dev/null | \
    ffmpeg -hide_banner -loglevel warning -f s16le -ar 44100 -ac 2 -i - -acodec pcm_s16le "$mic_file" &
    MIC_PID=$!
    ;;
  4)
    get_mic_source
    echo "🎧🔈 Mix of system + microphone (PulseAudio)..."
    ffmpeg -hide_banner -loglevel warning -f pulse -i "$monitor_source" "$sysaudio_file" &
    SYS_PID=$!
    parec --device="$mic_source" --format=s16le --rate=44100 --channels=2 2>/dev/null | \
    ffmpeg -hide_banner -loglevel warning -f s16le -ar 44100 -ac 2 -i - -acodec pcm_s16le "$mic_file" &
    MIC_PID=$!
    ;;
  *)
    echo "📷 Only video without audio is recorded."
    ;;
esac


# ▶️ Aufnahme starten
echo " "
echo "---------------------------------"
echo -e "        🎥 \e[31m Recording starts... \e[0m"
echo " "
echo "=> End recording again with [ENTER] <="


# Starte Videoaufnahme (ohne Audio)
ffmpeg -hide_banner -loglevel warning -f x11grab -r 30 -s "${width}x${height}" -i ":0.0+${pos_x},${pos_y}" \
  -c:v libx264 -preset ultrafast -crf 18 \
  "$video_file" &
VIDEO_PID=$!


echo

echo "⏺️ Recording in progress... Press [ENTER] to stop."
# Warten auf ENTER, aber sauber im Hintergrund
read -r < /dev/tty

echo "🔻 Recording will be terminated..."

set +e

# Prozesse sanft beenden, wenn sie noch laufen
if kill -0 "$VIDEO_PID" 2>/dev/null; then
  kill "$VIDEO_PID"
  wait "$VIDEO_PID"
fi

if [[ -n "$SYS_PID" && $(ps -p "$SYS_PID" -o comm= 2>/dev/null) ]]; then
  kill "$SYS_PID"
  wait "$SYS_PID"
fi

if [[ -n "$MIC_PID" && $(ps -p "$MIC_PID" -o comm= 2>/dev/null) ]]; then
  kill "$MIC_PID"
  wait "$MIC_PID"
fi

# Verarbeitung
echo "⚙️  Processing ..."

if [[ "$audio_choice" == "2" ]]; then
  ffmpeg -hide_banner -loglevel warning -y -i "$video_file" -i "$sysaudio_file" -c:v copy -c:a aac -b:a 192k "$final_file"
  rm -f "$sysaudio_file" "$video_file"
elif [[ "$audio_choice" == "3" ]]; then
  ffmpeg -hide_banner -loglevel warning -y -i "$video_file" -i "$mic_file" -c:v copy -c:a aac -b:a 192k "$final_file"
  rm -f "$mic_file" "$video_file"
elif [[ "$audio_choice" == "4" ]]; then
  ffmpeg -hide_banner -loglevel warning -y -i "$video_file" -i "$sysaudio_file" -i "$mic_file" \
    -filter_complex "[1:a][2:a]amix=inputs=2:duration=first[a]" -map 0:v -map "[a]" \
    -c:v copy -c:a aac -b:a 192k "$final_file"
  rm -f "$mic_file" "$sysaudio_file" "$video_file"
else
  mv "$video_file" "$final_file"
fi

echo "✅ Done! File saved: $final_file"

xdg-open $final_file
