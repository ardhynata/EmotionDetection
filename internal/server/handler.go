package server

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os/exec"
	"sync"
	"text/template"
)

const refreshInterval = 0.1 // seconds

var (
	pyStdin  io.WriteCloser
	pyStdout *bufio.Reader
	pyLock   sync.Mutex

	frameCounter int
	bufferSize   = 50
)

func StartPython() {
	cmd := exec.Command("python", "python/yolo_emotion.py")

	var err error
	pyStdin, err = cmd.StdinPipe()
	if err != nil {
		log.Fatal("Failed to get stdin pipe:", err)
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal("Failed to get stdout pipe:", err)
	}

	pyStdout = bufio.NewReader(stdoutPipe)

	if err := cmd.Start(); err != nil {
		log.Fatal("Failed to start Python process:", err)
	}

	log.Println("Persistent Python process started")
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles("templates/index.html")
	if err != nil {
		http.Error(w, "failed to load template", http.StatusInternalServerError)
		return
	}

	data := struct{ RefreshInterval float32 }{RefreshInterval: refreshInterval}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = tmpl.Execute(w, data)
}

func handleAnalyzeFrameHTML(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles("templates/analyze_frame.html")
	if err != nil {
		http.Error(w, "failed to load analyze frame template", http.StatusInternalServerError)
		return
	}

	data := struct{ RefreshInterval float32 }{RefreshInterval: refreshInterval}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = tmpl.Execute(w, data)
}

func handleAnalyzeFrame(w http.ResponseWriter, r *http.Request) {
	file, _, err := r.FormFile("frame")
	if err != nil {
		http.Error(w, "no frame uploaded", http.StatusBadRequest)
		return
	}
	defer file.Close()

	imgBytes, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "read error", http.StatusInternalServerError)
		return
	}

	encoded := base64.StdEncoding.EncodeToString(imgBytes)

	pyLock.Lock()
	defer pyLock.Unlock()

	frameCounter++
	currentFrame := frameCounter

	// Send the image to Python
	_, _ = pyStdin.Write([]byte(encoded + "\n"))

	if currentFrame < bufferSize {
		log.Printf("Buffered frame %d/%d", currentFrame, bufferSize)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "waiting",
			"frames": currentFrame,
		})
		return
	}

	// Reset counter once we hit 10 frames
	frameCounter = 0

	// Wait for Python to output the averaged result
	result, err := pyStdout.ReadString('\n')
	if err != nil {
		log.Println("Python read error:", err)
		http.Error(w, "processing error", http.StatusInternalServerError)
		return
	}

	log.Println("=== JSON Output from Python ===")
	log.Println(result)

	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(result))
}
