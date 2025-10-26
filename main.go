package main

import (
	"emotiondetection/internal/server"
	"log"
	"net/http"
)

func main() {
	server.StartPython()
	mux := server.NewRouter()
	log.Println("Server started on :8080")
	log.Fatal(http.ListenAndServe(":8080", mux))
}
