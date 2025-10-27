package server

import (
	"bufio"
	"encoding/base64"
	"io"
	"log"
	"net/http"
	"os/exec"
	"sync"
	"text/template"
)

const refreshInterval = 7 // seconds

var (
	pyStdin  io.WriteCloser
	pyStdout *bufio.Reader
	pyLock   sync.Mutex
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

	data := struct {
		RefreshInterval int
	}{
		RefreshInterval: refreshInterval,
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := tmpl.Execute(w, data); err != nil {
		http.Error(w, "failed to render template", http.StatusInternalServerError)
	}
}

func handleAnalyzeFrameHTML(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles("templates/analyze_frame.html")
	if err != nil {
		http.Error(w, "failed to load analyze frame template", http.StatusInternalServerError)
		return
	}

	data := struct {
		RefreshInterval int
	}{
		RefreshInterval: refreshInterval,
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := tmpl.Execute(w, data); err != nil {
		http.Error(w, "failed to render analyze frame", http.StatusInternalServerError)
	}
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

	pyLock.Lock()
	defer pyLock.Unlock()

	// Send image to Python process
	encoded := base64.StdEncoding.EncodeToString(imgBytes)
	pyStdin.Write([]byte(encoded + "\n"))

	// Read JSON output from Python
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
