self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open("gausscapture-mobile-v1").then((cache) => cache.addAll(["./", "./index.html", "./style.css", "./app.js", "./manifest.webmanifest"]))
  );
});

self.addEventListener("fetch", (event) => {
  event.respondWith(caches.match(event.request).then((cached) => cached || fetch(event.request)));
});

