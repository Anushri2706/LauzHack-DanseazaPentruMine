import { createServer } from "node:http";
import { readFile } from "node:fs/promises";

const files = {
  "/": ["index.html", "text/html; charset=utf-8"],
  "/style.css": ["style.css", "text/css; charset=utf-8"],
  "/app.js": ["app.js", "text/javascript; charset=utf-8"],
  "/simulation.js": ["simulation.js", "text/javascript; charset=utf-8"],
};

createServer(async (request, response) => {
  const file = files[request.url];
  if (!file || !["GET", "HEAD"].includes(request.method)) {
    response.writeHead(404).end("Not found");
    return;
  }
  try {
    const contents = await readFile(new URL(file[0], import.meta.url));
    response.writeHead(200, { "Content-Type": file[1], "X-Content-Type-Options": "nosniff" });
    response.end(request.method === "HEAD" ? undefined : contents);
  } catch {
    response.writeHead(500).end("Unable to load application");
  }
}).listen(3000, "127.0.0.1", () => console.log("Financial Twin is running at http://127.0.0.1:3000"));
