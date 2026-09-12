"""Módulo de funciones reutilizables para interactuar con el Arcade Learning
Environment (ALE) a través de Gymnasium.

Estas funciones son agnósticas al entorno: funcionan tanto con
``ALE/SpaceInvaders-v5`` como con cualquier otro entorno registrado en
Gymnasium (``CartPole-v1``, ``FrozenLake-v1``, etc.), 
"""

import os
import time

import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

ENTORNO_POR_DEFECTO = "ALE/SpaceInvaders-v5"


def crear_entorno(nombre_entorno=ENTORNO_POR_DEFECTO,
                  render_mode="rgb_array",
                  video_folder=None,
                  name_prefix="rl-video",
                  episode_trigger=None,
                  **kwargs):
    """Crea y retorna un entorno de Gymnasium, opcionalmente con grabación de video.

    Parámetros
    ----------
    nombre_entorno : str
        Id del entorno a crear (p. ej. ``"ALE/SpaceInvaders-v5"`` o ``"CartPole-v1"``).
    render_mode : str o None
        Modo de renderizado. Debe ser ``"rgb_array"`` para poder grabar video;
        si se pasa ``video_folder`` se fuerza automáticamente a ese valor.
    video_folder : str o None
        Carpeta donde se guardarán los videos. Si es ``None`` no se graba nada
        y se retorna el entorno "pelado".
    name_prefix : str
        Prefijo de los archivos ``.mp4`` generados por ``RecordVideo``.
    episode_trigger : callable o None
        Función ``(int) -> bool`` que decide qué episodios se graban según su
        índice (base 0). Por defecto se graban **todos** los episodios.
    **kwargs
        Argumentos adicionales que se pasan tal cual a ``gymnasium.make``
        (p. ej. ``frameskip``, ``repeat_action_probability``,
        ``full_action_space``, ``obs_type``).

    Retorna
    -------
    gymnasium.Env
        El entorno creado, envuelto en ``RecordVideo`` si se pidió grabación.
    """
    if video_folder is not None:
        # RecordVideo necesita frames RGB para poder escribir el video.
        render_mode = "rgb_array"

    env = gym.make(nombre_entorno, render_mode=render_mode, **kwargs)

    if video_folder is not None:
        if episode_trigger is None:
            def episode_trigger(episode_id):
                return True

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            name_prefix=name_prefix,
            disable_logger=True,
        )

    return env


def agente_aleatorio(observation, env):
    return env.action_space.sample()


# --- Constantes de color del frame RGB de ALE/SpaceInvaders-v5 -------------
COLOR_JUGADOR = (50, 132, 50)     # cañón láser del jugador (verde)
COLOR_ALIENS = (134, 134, 29)     # fila de invasores (amarillo)
FILAS_JUGADOR = (185, 195)        # banda de filas donde vive el cañón
X_MIN_CANON, X_MAX_CANON = 20, 140  # rango horizontal recorrible por el cañón
MARGEN_ALIEN = 2                    # px de holgura al considerar una columna ocupada


def _columnas_de_color(observation, color, filas=None):
    """Retorna las columnas (x) donde aparece un color dado en el frame.

    Parámetros
    ----------
    observation : np.ndarray
        Frame RGB de forma ``(210, 160, 3)``.
    color : tuple
        Color RGB a buscar.
    filas : tuple o None
        Rango ``(y_min, y_max)`` de filas donde buscar. ``None`` = todo el frame.

    Retorna
    -------
    np.ndarray
        Índices de columna donde el color está presente (puede venir vacío).
    """
    frame = observation if filas is None else observation[filas[0]:filas[1]]
    mascara = np.all(frame == np.array(color, dtype=np.uint8), axis=-1)
    return np.where(mascara.any(axis=0))[0]


def agente_regla_simple(observation, env):
    """Agente heurístico (no aprendido) para ALE/SpaceInvaders-v5.

    Regla: **colocarse en un hueco entre las columnas de invasores y disparar
    sin parar**. En cada paso el agente localiza por color su cañón y todos los
    invasores vivos, calcula las columnas libres (aquellas que no tienen ningún
    alien encima, con un margen de seguridad) y se desplaza hacia la más
    cercana mientras dispara:

    - Hueco a la derecha  -> ``RIGHTFIRE``
    - Hueco a la izquierda -> ``LEFTFIRE``
    - Ya está en un hueco  -> ``FIRE``

    La intuición es doble: los invasores solo lanzan bombas desde su propia
    columna, así que estar en un hueco reduce el riesgo de ser alcanzado, y
    como la formación avanza en horizontal los aliens terminan entrando solos
    en la línea de fuego del cañón.

    La detección se hace por color sobre el frame RGB. Si la observación no es una imagen RGB de Atari, o si
    no se detecta el cañón o los aliens, el agente simplemente dispara.

    Parámetros
    ----------
    observation : np.ndarray
        Observación actual del entorno (frame RGB de 210x160x3).
    env : gymnasium.Env
        Entorno, usado para resolver los índices de las acciones por nombre.

    Retorna
    -------
    int
        Índice de la acción elegida, válido en ``env.action_space``.
    """
    try:
        nombres = env.unwrapped.get_action_meanings()
    except AttributeError:
        return agente_aleatorio(observation, env)

    accion_fire = nombres.index("FIRE") if "FIRE" in nombres else 0
    accion_der = nombres.index("RIGHTFIRE") if "RIGHTFIRE" in nombres else accion_fire
    accion_izq = nombres.index("LEFTFIRE") if "LEFTFIRE" in nombres else accion_fire

    obs = np.asarray(observation)
    if obs.ndim != 3 or obs.shape[-1] != 3:
        return accion_fire

    x_jugador = _columnas_de_color(obs, COLOR_JUGADOR, FILAS_JUGADOR)
    x_aliens = _columnas_de_color(obs, COLOR_ALIENS)
    if len(x_jugador) == 0 or len(x_aliens) == 0:
        return accion_fire

    centro_jugador = x_jugador.mean()

    # Columnas "peligrosas": las que tienen un alien encima (+/- margen).
    ocupadas = np.concatenate(
        [x_aliens + k for k in range(-MARGEN_ALIEN, MARGEN_ALIEN + 1)]
    )
    libres = np.setdiff1d(np.arange(X_MIN_CANON, X_MAX_CANON), ocupadas)
    if len(libres) == 0:
        return accion_fire

    # Hueco más cercano al cañón.
    objetivo = libres[np.argmin(np.abs(libres - centro_jugador))]
    dx = objetivo - centro_jugador

    if abs(dx) <= 1:
        return accion_fire
    return accion_der if dx > 0 else accion_izq


def ejecutar_episodio(env, funcion_agente, max_steps=10000, seed=None, verbose=False):
    """Ejecuta un episodio completo en ``env`` usando ``funcion_agente``.

    El episodio corre hasta que ``terminated`` o ``truncated`` sea verdadero,
    o hasta alcanzar ``max_steps`` pasos.

    Parámetros
    ----------
    env : gymnasium.Env
        Entorno ya creado (puede venir envuelto en ``RecordVideo``).
    funcion_agente : callable
        Función ``(observation, env) -> action``, por ejemplo
        :func:`agente_aleatorio`, :func:`agente_regla_simple` o, en el
        Proyecto 2, una política aprendida.
    max_steps : int
        Número máximo de pasos a ejecutar.
    seed : int o None
        Semilla para ``env.reset`` y para ``env.action_space``, útil para
        reproducir un episodio exacto (incluido el del agente aleatorio).
    verbose : bool
        Si es ``True`` imprime un resumen al terminar el episodio.

    Retorna
    -------
    dict
        Métricas del episodio:

        - ``"pasos"``: número de pasos ejecutados.
        - ``"recompensa_total"``: return acumulado (suma de recompensas).
        - ``"terminated"`` / ``"truncated"``: cómo terminó el episodio.
        - ``"motivo_fin"``: ``"terminated"``, ``"truncated"`` o ``"max_steps"``.
        - ``"recompensas"``: lista con la recompensa de cada paso.
        - ``"info"``: último diccionario ``info`` devuelto por el entorno
          (en Atari incluye ``lives`` y ``frame_number``).
    """
    observation, info = env.reset(seed=seed)
    if seed is not None:
        # El action_space tiene su propio generador aleatorio, independiente
        # del de env.reset: hay que sembrarlo también para que un episodio del
        # agente aleatorio sea exactamente reproducible.
        env.action_space.seed(seed)

    recompensas = []
    recompensa_total = 0.0
    pasos = 0
    terminated = False
    truncated = False

    while not (terminated or truncated) and pasos < max_steps:
        accion = funcion_agente(observation, env)
        observation, reward, terminated, truncated, info = env.step(accion)
        recompensa_total += float(reward)
        recompensas.append(float(reward))
        pasos += 1

    if terminated:
        motivo_fin = "terminated"
    elif truncated:
        motivo_fin = "truncated"
    else:
        motivo_fin = "max_steps"

    resultado = {
        "pasos": pasos,
        "recompensa_total": recompensa_total,
        "terminated": terminated,
        "truncated": truncated,
        "motivo_fin": motivo_fin,
        "recompensas": recompensas,
        "info": info,
    }

    if verbose:
        print(f"Episodio finalizado ({motivo_fin}): "
              f"pasos={pasos}, recompensa_total={recompensa_total:.1f}")

    return resultado


def generar_video_agente(nombre_entorno=ENTORNO_POR_DEFECTO,
                         funcion_agente=agente_aleatorio,
                         video_folder="videos",
                         name_prefix="agente",
                         n_episodios=1,
                         max_steps=10000,
                         seed=None,
                         verbose=True,
                         **kwargs):
    """Ejecuta episodios completos grabando video y retorna rutas y métricas.

    Parámetros
    ----------
    nombre_entorno : str
        Id del entorno (por defecto ``"ALE/SpaceInvaders-v5"``).
    funcion_agente : callable
        Función ``(observation, env) -> action``.
    video_folder : str
        Carpeta destino de los archivos ``.mp4``.
    name_prefix : str
        Prefijo de los videos generados.
    n_episodios : int
        Cantidad de episodios completos a ejecutar y grabar.
    max_steps : int
        Corte de seguridad de pasos por episodio.
    seed : int o None
        Semilla base. El episodio ``i`` usa ``seed + i``, de modo que los
        videos son reproducibles.
    verbose : bool
        Imprime el resumen de cada episodio mientras corre.
    **kwargs
        Argumentos adicionales para ``gymnasium.make`` (``frameskip``,
        ``repeat_action_probability``, ``full_action_space``, ...).

    Retorna
    -------
    dict
        - ``"videos"``: lista de rutas a los ``.mp4`` escritos por esta llamada
          (se puede repetir la corrida sobre una carpeta que ya tenga videos).
        - ``"episodios"``: lista de dicts con las métricas de cada episodio
          (``pasos``, ``recompensa_total``, ``motivo_fin``, ...).
        - ``"recompensa_promedio"`` y ``"pasos_promedio"``: resumen de la corrida.
    """
    # Marca de tiempo para distinguir los videos escritos por esta llamada de
    # los que ya estuvieran en la carpeta. Se compara por fecha de modificación
    # y no por nombre, porque al repetir la corrida RecordVideo sobrescribe los
    # archivos con el mismo prefijo en vez de crear nombres nuevos.
    carpeta = os.path.abspath(video_folder)
    inicio = time.time() - 1  # margen por la granularidad del sistema de archivos

    env = crear_entorno(
        nombre_entorno,
        video_folder=video_folder,
        name_prefix=name_prefix,
        **kwargs,
    )

    episodios = []
    try:
        for i in range(n_episodios):
            semilla = None if seed is None else seed + i
            resultado = ejecutar_episodio(
                env, funcion_agente, max_steps=max_steps, seed=semilla
            )
            resultado["episodio"] = i
            resultado["seed"] = semilla
            episodios.append(resultado)

            if verbose:
                print(f"Episodio {i + 1}/{n_episodios} ({resultado['motivo_fin']}): "
                      f"pasos={resultado['pasos']}, "
                      f"recompensa_total={resultado['recompensa_total']:.1f}")
    finally:
        env.close()

    nuevos = sorted(
        f for f in os.listdir(carpeta)
        if f.startswith(name_prefix) and f.endswith(".mp4")
        and os.path.getmtime(os.path.join(carpeta, f)) >= inicio
    )
    videos = [os.path.join(carpeta, f) for f in nuevos]

    recompensas = [ep["recompensa_total"] for ep in episodios]
    pasos = [ep["pasos"] for ep in episodios]
    resumen = {
        "videos": videos,
        "episodios": episodios,
        "recompensa_promedio": sum(recompensas) / len(recompensas) if recompensas else 0.0,
        "pasos_promedio": sum(pasos) / len(pasos) if pasos else 0.0,
    }

    if verbose:
        print(f"\nVideos generados en {carpeta}:")
        for ruta in videos:
            print("  -", os.path.basename(ruta))
        print(f"Recompensa promedio: {resumen['recompensa_promedio']:.1f} | "
              f"Pasos promedio: {resumen['pasos_promedio']:.1f}")

    return resumen
