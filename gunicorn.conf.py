# gunicorn lee este archivo automáticamente al arrancar (Render: gunicorn app:app)

# Un solo worker: cada worker extra carga otra copia del modelo (512 MB en Render gratis)
workers = 1
threads = 2

# Render gratis tiene 0.1 CPU: cargar el modelo y responder puede pasar de 30 s
timeout = 180

# Carga el modelo antes de crear el worker, así la carga no cuenta para el timeout
preload_app = True
