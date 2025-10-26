package server

import (
	"net/http"
)

func NewRouter() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/analyze-frame", handleAnalyzeFrame)
	mux.HandleFunc("/analyze-frame-html", handleAnalyzeFrameHTML)
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
	return mux
}
