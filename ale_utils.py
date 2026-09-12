"""Módulo de funciones reutilizables para interactuar con el Arcade Learning
Environment (ALE) a través de Gymnasium.

Estas funciones son agnósticas al entorno: funcionan tanto con
``ALE/SpaceInvaders-v5`` como con cualquier otro entorno registrado en
Gymnasium (``CartPole-v1``, ``FrozenLake-v1``, etc.), 
"""

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
