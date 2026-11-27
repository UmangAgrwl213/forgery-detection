import http.server, subprocess

SUB = "garbage-buckle-darn"

class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        b = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        url = f"https://{SUB}.ngrok-free.dev{self.path}"
        p = subprocess.run(
            ["curl.exe", "--http1.1", "--retry", "2", "-s", "-S", "-X", "POST", url,
             "-H", "Content-Type: application/json",
             "-H", "ngrok-skip-browser-warning: true",
             "--data-binary", "@-"],
            input=b, capture_output=True
        )
        if p.returncode == 0 and p.stdout:
            self.send_response(200)
            ct = "text/event-stream" if b'"stream":true' in b else "application/json"
            self.send_header("Content-Type", ct)
            self.end_headers()
            self.wfile.write(p.stdout)
        else:
            print("CURL ERR:", p.returncode, p.stderr.decode(errors="ignore"))
            self.send_response(502)
            self.end_headers()

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

print(f"Bridge active -> https://{SUB}.ngrok-free.dev")
http.server.ThreadingHTTPServer(("127.0.0.1", 11434), H).serve_forever()
