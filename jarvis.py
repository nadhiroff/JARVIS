#!/usr/bin/env python3
"""
Jarvis v5 - assistant vocal perso avec contrôle total de l'ordinateur.
Cerveau : plusieurs modèles Gemini (quotas séparés), puis Groq, avec bascule automatique et
          historique partagé. Agent multi-étapes : terminal visible, fichiers, éditeur de code.
Voix    : Gemini TTS (Kore) -> secours : Piper (Siwis, local) -> edge-tts -> espeak.

    python jarvis.py test-cle     -> teste Gemini (et affiche le temps de réponse)
    python jarvis.py test-groq    -> teste chaque modèle Groq
    python jarvis.py test-terminal -> ouvre le terminal de Jarvis et y tape une commande
    python jarvis.py test-voix    -> teste la voix
    python jarvis.py apprendre-voix -> Jarvis apprend TA voix (à faire une fois)
    python jarvis.py test-id      -> montre si Jarvis te reconnais (réglage du seuil)
    python jarvis.py test-micro   -> teste le micro
    python jarvis.py test-outils  -> teste les outils (sans micro, sans IA)
    python jarvis.py              -> lance Jarvis

Les clés se mettent dans l'environnement (export GEMINI_API_KEY=... GROQ_API_KEY=...)
ou dans un fichier .env à côté de ce script. JAMAIS dans le code.

Voix de secours Piper (une fois) :
    ./venv/bin/pip install -U piper-tts edge-tts
    mkdir -p voix && cd voix
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json

Navigateur web piloté (une fois) :
    ./venv/bin/pip install playwright beautifulsoup4 && ./venv/bin/playwright install chromium

Reconnaissance de ta voix + interruption (une fois) :
    ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    ./venv/bin/pip install numpy scipy librosa webrtcvad-wheels
    ./venv/bin/pip install resemblyzer --no-deps
    ./venv/bin/python jarvis.py apprendre-voix
"""

import asyncio
import collections
import datetime
import getpass
import glob
import inspect
import json
import os
import platform
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import webbrowser

# ----------------------------- REGLAGES ------------------------------------
PRENOM = "Nadhir"                         # ton prénom : Jarvis t'appelle comme ça
NOMS = ("jarvis", "jarvice", "djarvis")   # mots d'activation acceptés
# Cerveaux, essayés dans l'ordre. Chaque modèle Gemini a SON PROPRE quota gratuit : quand l'un est
# épuisé, Jarvis passe au suivant, puis à Groq. Un modèle en erreur est mis en pause puis réessayé.
MODELES_GEMINI = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest",
                  "gemini-3.1-flash-lite")
MODELES_GROQ = ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b")
REFLEXION = "low"                         # réflexion de Gemini 3 : "low" (rapide ; "minimal" est refusé par gemini-3.8-flash)
DELAI_CERVEAU = 30                        # secondes max d'attente d'un cerveau avant de passer au suivant
MOT_ACTIVATION_OBLIGATOIRE = False        # False = si c'est clairement TA voix, pas besoin de dire « Jarvis »
MAX_ETAPES = 30                           # nombre max d'outils enchaînés pour UNE demande (ouvrir, créer, écrire, vérifier...)
HISTORIQUE_ECHANGES = 12                  # nombre d'échanges gardés en mémoire (partagés entre Gemini et Groq)
LANGUE = "fr-FR"

# Contrôle de l'ordinateur
CONFIRMER_ACTIONS_SENSIBLES = False       # False = AUCUNE limite. True = il demande "dis oui" avant rm, sudo, etc.
DELAI_COMMANDE = 120                      # secondes max pour une commande

# Écoute
CONVERSATION_SECONDES = 45                # après une réponse, tu peux continuer SANS redire "Jarvis" (0 = toujours dire Jarvis)
FACTEUR_SEUIL = 1.2                       # >1 = ignore mieux les voix faibles/lointaines. Trop haut : il ne t'entend plus (essaie 1.2)
PAUSE_FIN_PHRASE = 1.2                    # secondes de silence avant de considérer que tu as fini (+ bas = répond plus vite)
ATTENTE_SUITE = 0.5                       # après ce silence, il écoute encore un peu si tu reprends la parole
ATTENTE_SUITE_LONGUE = 3.0                # ... et plus longtemps si ta phrase finit par « et », « puis », « de »...
DUREE_MAX_PHRASE = 90                     # durée max d'une demande en secondes (avant : 15 s, il te coupait)

# Ta voix (reconnaissance du locuteur) et interruption
RECONNAISSANCE_VOIX = True                # True = seule TA voix donne des ordres (d'abord : python jarvis.py apprendre-voix)
APPRENTISSAGE_CONTINU = True              # True = il affine ton empreinte tout seul à chaque phrase dont il est sûr que c'est toi
SEUIL_VOIX_COURT = 0.55                   # ressemblance exigée pour une phrase courte (moins de 2 s) : les notes y sont plus basses
SEUIL_VOIX_MOYEN = 0.62                   # entre 2 et 4 s
SEUIL_VOIX = 0.70                         # ressemblance minimale (0 à 1). Il te rejette ? baisse (0.65). Il accepte les autres ? monte (0.8)
INTERRUPTION = True                       # True = tu peux couper Jarvis en parlant (seulement avec TA voix)
SEUIL_INTERRUPTION = 0.58                 # ressemblance pour METTRE EN PAUSE Jarvis quand tu parles (il vérifie ensuite)
FACTEUR_INTERRUPTION = 1.0                # volume minimal avant de tenter de te reconnaître pendant qu'il parle
# Quand il croit t'entendre, il se met en PAUSE et écoute ta phrase sans sa propre voix par-dessus.
# Si c'est vraiment toi et une vraie demande : il s'arrête et te répond. Sinon (bruit, autre voix) : il REPREND.
MOTS_STOP = re.compile(r"\b(stop|arrête|arrêtez|attends|attend|tais[- ]toi|chut|silence|pause|non|jarvis|"
                       r"jarvice|djarvis|ok|bon|écoute|minute)\b", re.IGNORECASE)
DEBUG_INTERRUPTION = True                 # True = affiche la ressemblance pendant qu'il parle (pour régler). Mets False quand c'est bon
UN_INSTANT_APRES = None                   # N = dit « Un instant. » si la réponse met plus de N secondes (None = jamais)
VOIX_FICHIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voix_nadhir.npy")
# Phrases qui terminent la discussion (ensuite il faut redire "Jarvis")
FIN_CONVERSATION = re.compile(r"^(ok |d'accord |bon )?(merci|c'est bon|c'est tout|ça sera tout|ça suffit)( beaucoup| jarvis)*[ .!]*$")

# Voix Gemini (principale)
MODELES_TTS = ("gemini-3.8-flash-tts", "gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts")
VOIX_GEMINI = "Kore"                      # voix féminines : Kore, Zephyr, Aoede, Leda, Autonoe
STYLE_TTS = ""                            # consigne de ton pour la voix Gemini (certains modèles la lisent à voix haute !)
# Voix de secours 1 : Piper (locale, sans quota, sans internet)
PIPER_MODELE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voix", "fr_FR-siwis-medium.onnx")
# Voix de secours 2 : edge-tts (Microsoft, en ligne)
VOIX_EDGE = "fr-FR-DeniseNeural"
# ---------------------------------------------------------------------------

_MIC = {"r": None, "source": None}
_client_unique = None
_gemini_pause_jusqua = 0.0
_edge_pause_jusqua = 0.0
_modele_tts_ok = None
_piper_voice = None
_ENCODEUR = None        # modèle qui reconnaît les voix
_EMPREINTE = None       # ton empreinte vocale (chargée depuis voix_nadhir.npy)
_LECTURE = {"p": None}  # lecteur audio en cours (pour pouvoir le couper)
_INTERROMPU = {"oui": False, "audio": None, "texte": None}   # tu as coupé Jarvis : ta phrase
_REPRISE = threading.Event()   # « clair » quand Jarvis est en pause pour vérifier si tu lui parles
_REPRISE.set()
_APPRIS = {"n": 0}                              # nombre de phrases qui ont affiné ton empreinte
_DERNIER = {"audio": None}                     # dernier son entendu (pour vérifier la voix)
_silence = None
_ack = None            # petit "Un instant." pré-enregistré, joué pendant que le cerveau réfléchit
_SUDO_MDP = None       # mot de passe sudo, gardé UNIQUEMENT en mémoire
_CWD = {"d": os.path.expanduser("~")}   # dossier de travail de Jarvis (garde la position entre deux commandes)
_JOURNAL = []          # outils appelés pendant la demande en cours (évite de refaire une action en cas de bascule)


# =============================== CLES =======================================
def _lire_cle(nom):
    """Cherche la clé dans l'environnement, sinon dans le fichier .env à côté du script."""
    cle = os.environ.get(nom)
    if not cle:
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_file):
            with open(env_file) as f:
                for ligne in f:
                    if ligne.startswith(nom + "="):
                        cle = ligne.strip().split("=", 1)[1].strip('"\'')
                        break
    return cle or None


def _cle_api():
    cle = _lire_cle("GEMINI_API_KEY")
    if not cle:
        print("Erreur : GEMINI_API_KEY n'est pas définie.")
        print('Fais :  export GEMINI_API_KEY="ta_cle"   (ou crée un fichier .env avec GEMINI_API_KEY=ta_cle)')
        sys.exit(1)
    return cle


def _client():
    global _client_unique
    if _client_unique is None:
        from google import genai
        from google.genai import types
        # sans délai max, une requête Gemini saturée pouvait bloquer Jarvis indéfiniment
        _client_unique = genai.Client(api_key=_cle_api(),
                                      http_options=types.HttpOptions(timeout=DELAI_CERVEAU * 1000))
    return _client_unique


# =============================== BOUCHE =====================================
def _ecrire_wav(chemin, pcm):
    """Gemini TTS renvoie du PCM brut : 24000 Hz, 16 bits, mono."""
    with wave.open(chemin, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)


def _synth_gemini(texte):
    """Voix Gemini. Renvoie le chemin d'un .wav, ou None."""
    global _gemini_pause_jusqua, _modele_tts_ok
    if time.time() < _gemini_pause_jusqua:
        return None
    from google.genai import types
    config = types.GenerateContentConfig(
        http_options=types.HttpOptions(timeout=15000),   # voix trop lente -> Piper prend le relais
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOIX_GEMINI)
            )
        ),
    )
    modeles = [_modele_tts_ok] if _modele_tts_ok else list(MODELES_TTS)
    erreur = None
    for modele in modeles:
        try:
            rep = _client().models.generate_content(
                model=modele, contents=STYLE_TTS + texte, config=config)
            pcm = rep.candidates[0].content.parts[0].inline_data.data
            if not pcm:
                raise RuntimeError("aucun audio reçu")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                chemin = f.name
            _ecrire_wav(chemin, pcm)
            _modele_tts_ok = modele
            return chemin
        except Exception as e:
            erreur = e
            if "404" in str(e) or "NOT_FOUND" in str(e):
                continue
            break
    limite = "429" in str(erreur) or "RESOURCE_EXHAUSTED" in str(erreur)
    print(f"[Voix Gemini {'(quota atteint)' if limite else 'indisponible'} : {type(erreur).__name__}]")
    par_jour = "perday" in str(erreur).lower().replace(" ", "")
    _gemini_pause_jusqua = time.time() + (3 * 3600 if par_jour else 600 if limite else 60)   # en attendant : voix Piper, instantanée
    return None


def _synth_piper(texte):
    """Voix de secours locale (Piper, voix féminine Siwis). Renvoie le chemin d'un .wav, ou None."""
    global _piper_voice
    if not os.path.exists(PIPER_MODELE):
        return None
    chemin = None
    try:
        if _piper_voice is None:
            from piper import PiperVoice
            _piper_voice = PiperVoice.load(PIPER_MODELE)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            chemin = f.name
        with wave.open(chemin, "wb") as w:
            if hasattr(_piper_voice, "synthesize_wav"):
                _piper_voice.synthesize_wav(texte, w)      # piper-tts récent
            else:
                _piper_voice.synthesize(texte, w)          # ancienne version
        if os.path.getsize(chemin) < 100:
            raise RuntimeError("fichier audio vide")
        return chemin
    except Exception as e:
        print(f"[Voix Piper indisponible : {type(e).__name__}: {e}]")
        if chemin and os.path.exists(chemin):
            os.remove(chemin)
        return None


async def _synthese_edge(texte, chemin):
    import edge_tts
    await edge_tts.Communicate(texte, VOIX_EDGE).save(chemin)


def _synth_edge(texte):
    """Voix de secours Microsoft. Renvoie le chemin d'un .mp3, ou None."""
    global _edge_pause_jusqua
    if time.time() < _edge_pause_jusqua:
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        chemin = f.name
    try:
        asyncio.run(_synthese_edge(texte, chemin))
        if os.path.getsize(chemin) == 0:
            raise RuntimeError("fichier audio vide")
        return chemin
    except Exception as e:
        print(f"[Voix de secours edge-tts indisponible : {type(e).__name__}]")
        _edge_pause_jusqua = time.time() + 600
        if os.path.exists(chemin):
            os.remove(chemin)
        return None


def _fichier_silence():
    """Court silence joué avant la voix : évite que le début de la phrase soit coupé."""
    global _silence
    if _silence is None or not os.path.exists(_silence):
        _silence = os.path.join(tempfile.gettempdir(), "jarvis_silence.wav")
        _ecrire_wav(_silence, b"\x00\x00" * int(24000 * 0.3))
    return _silence


def _commande_lecture(chemin, silence):
    if shutil.which("mpv"):
        fichiers = ([_fichier_silence()] if silence else []) + [chemin]
        return ["mpv", "--no-video", "--really-quiet"] + fichiers
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", chemin]
    return None


def _lancer_et_attendre(cmd):
    """Lance un lecteur/synthétiseur audio et attend la fin. _couper_lecture() peut l'arrêter."""
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _LECTURE["p"] = p
    if _INTERROMPU["oui"]:
        p.terminate()
    elif not _REPRISE.is_set():
        _pause_lecture(True)                 # démarré pile pendant une vérification : en pause aussi
    p.wait()
    _LECTURE["p"] = None


def _couper_lecture():
    p = _LECTURE["p"]
    if p and p.poll() is None:
        try:
            p.send_signal(__import__("signal").SIGCONT)   # un lecteur en pause doit pouvoir mourir
        except Exception:
            pass
        p.terminate()


def _pause_lecture(pause):
    """Met en pause (SIGSTOP) ou relance (SIGCONT) la voix de Jarvis, sans perdre où il en était."""
    import signal
    if pause:
        _REPRISE.clear()
    p = _LECTURE["p"]
    if p and p.poll() is None:
        try:
            p.send_signal(signal.SIGSTOP if pause else signal.SIGCONT)
        except Exception:
            pass
    if not pause:
        _REPRISE.set()


def _jouer(chemin, silence=True):
    cmd = _commande_lecture(chemin, silence)
    if not cmd:
        print("[Aucun lecteur audio : sudo apt install mpv]")
        return False
    _lancer_et_attendre(cmd)
    return True


def _jouer_en_fond(chemin):
    """Lance la lecture sans attendre (renvoie le processus, ou None)."""
    cmd = _commande_lecture(chemin, True)
    if not cmd:
        return None
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _espeak(texte):
    for prog in ("espeak-ng", "espeak"):
        if shutil.which(prog):
            _lancer_et_attendre([prog, "-v", "fr+f3", "-s", "150", texte])
            return
    print("[espeak-ng non installé : sudo apt install espeak-ng]")


def _decouper(texte):
    """Nettoie le texte et le coupe en 2 : 1re phrase (courte, donc rapide à générer) + le reste."""
    texte = re.sub(r"https?://\S+", "le lien", texte)
    texte = re.sub(r"\[(\d+|réf\. nécessaire|note \d+|[a-z])\]", "", texte)    # renvois de notes [2]
    texte = re.sub(r"[*_#`>~|]", "", texte)
    texte = re.sub("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]", "", texte)   # les émojis ne se lisent pas
    texte = re.sub(r"\s+", " ", texte).strip()
    morceaux = re.split(r"(?<=[.!?…])\s+", texte, maxsplit=1)
    return [m for m in morceaux if m.strip()]


def _synth(texte):
    """Ordre des voix : Gemini (Kore) -> Piper (Siwis) -> edge-tts. espeak est le dernier recours."""
    return _synth_gemini(texte) or _synth_piper(texte) or _synth_edge(texte)


def parler(texte, silence=True):
    """Dit le texte. La 1re phrase démarre vite ; la suite se prépare pendant qu'elle est dite.
    Si tu parles pendant ce temps (et que c'est bien ta voix), Jarvis se tait et t'écoute."""
    print(f"Jarvis : {texte}")
    morceaux = _decouper(texte)
    if not morceaux:
        return
    _INTERROMPU["oui"], _INTERROMPU["audio"], _INTERROMPU["texte"] = False, None, None
    _REPRISE.set()
    t0 = time.time()
    q = queue.Queue()

    def producteur():
        for m in morceaux:
            q.put((m, _synth(m)))
        q.put(None)

    threading.Thread(target=producteur, daemon=True).start()
    arret = threading.Event()
    veilleur = None
    if INTERRUPTION and _EMPREINTE is not None and _MIC["r"] is not None:
        veilleur = threading.Thread(target=_surveiller_interruption, args=(arret,), daemon=True)
        veilleur.start()
    premier = True
    try:
        while True:
            item = q.get()
            if item is None:
                break
            morceau, chemin = item
            try:
                _REPRISE.wait(30)                    # s'il vérifie si tu lui parles, la suite attend
                if not _INTERROMPU["oui"]:
                    if premier:
                        print(f"   (voix prête en {time.time() - t0:.1f}s)")
                    if not (chemin and _jouer(chemin, silence=silence and premier)):
                        _espeak(morceau)
            finally:
                if chemin and os.path.exists(chemin):
                    os.remove(chemin)
            premier = False
    finally:
        arret.set()
        if veilleur:
            veilleur.join(timeout=20)
        _REPRISE.set()


def preparer_ack():
    """Génère une fois "Un instant." : joué tout de suite, pendant que le cerveau réfléchit."""
    global _ack
    _ack = _synth("Un instant.")


# ======================= RECONNAISSANCE DE TA VOIX ==========================
def _rms(data):
    """Volume d'un morceau de son (même échelle que le seuil d'énergie du micro)."""
    import numpy as np
    a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def _encodeur():
    global _ENCODEUR
    if _ENCODEUR is None:
        from resemblyzer import VoiceEncoder
        _ENCODEUR = VoiceEncoder("cpu", verbose=False)
    return _ENCODEUR


def _pcm16k(audio):
    return audio.get_raw_data(convert_rate=16000, convert_width=2)


def _empreinte_de(pcm16k):
    """Empreinte vocale (vecteur) d'un son PCM 16 bits mono 16 kHz, ou None si trop court."""
    import numpy as np
    wav = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32) / 32768.0
    try:
        from resemblyzer import preprocess_wav
        net = preprocess_wav(wav, source_sr=16000)
        if len(net) >= 8000:            # on garde la version nettoyée si assez de parole reste
            wav = net
    except Exception:
        pass
    if len(wav) < 8000:
        return None
    return _encodeur().embed_utterance(wav)


def _score_emb(audio):
    """Renvoie (empreinte, ressemblance 0 à 1 avec ta voix) ou (None, None) si impossible à juger."""
    try:
        import numpy as np
        emb = _empreinte_de(_pcm16k(audio))
        if emb is None:
            return None, None
        return emb, float(np.dot(emb, _EMPREINTE))
    except Exception as e:
        print(f"[Vérification de voix impossible : {type(e).__name__}: {e}]")
        return None, None


def _score(audio):
    return _score_emb(audio)[1]


def _seuil_pour(duree):
    """Plus la phrase est courte, moins la note est fiable : on est moins exigeant."""
    if duree < 2.0:
        return SEUIL_VOIX_COURT
    if duree < 4.0:
        return SEUIL_VOIX_MOYEN
    return SEUIL_VOIX


def _apprendre(emb, s, duree):
    """Affine ton empreinte avec une phrase dont on est presque sûr que c'est toi (et la sauvegarde)."""
    global _EMPREINTE
    if not APPRENTISSAGE_CONTINU or emb is None or duree < 1.5 or s < 0.70:
        return
    try:
        import numpy as np
        nouv = 0.95 * _EMPREINTE + 0.05 * emb
        _EMPREINTE = nouv / np.linalg.norm(nouv)
        np.save(VOIX_FICHIER, _EMPREINTE)
        _APPRIS["n"] += 1
        print(f"   (j'ai affiné ta voix : {_APPRIS['n']} fois depuis le lancement)")
    except Exception as e:
        print(f"[Apprentissage continu impossible : {type(e).__name__}: {e}]")


def _est_nadhir(audio, avec_nom=False):
    """True si c'est ta voix (ou si on ne peut pas juger : voix non apprise, son trop court).
    Si tu as dit « Jarvis », on est un peu moins exigeant.
    _DERNIER["sur"] = True seulement si ta voix a vraiment été reconnue (pas juste « impossible à juger »)."""
    _DERNIER["sur"] = False
    if _EMPREINTE is None or audio is None:
        return True
    duree = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
    if duree < 1.0:
        return True                      # moins d'1 seconde : trop court pour être fiable
    emb, s = _score_emb(audio)
    if s is None:
        return True
    seuil = _seuil_pour(duree) - (0.05 if avec_nom else 0.0)
    ok = s >= seuil
    print(f"   (ressemblance avec ta voix : {s:.2f} / exigé : {seuil:.2f})")
    if ok:
        _DERNIER["sur"] = True
        _apprendre(emb, s, duree)
    return ok


def _charger_empreinte():
    global _EMPREINTE
    if not RECONNAISSANCE_VOIX:
        return
    if not os.path.exists(VOIX_FICHIER):
        print("[Ta voix n'est pas encore apprise : lance « python jarvis.py apprendre-voix »]")
        return
    try:
        import numpy as np
        _EMPREINTE = np.load(VOIX_FICHIER)
        _encodeur()
        print(f"Reconnaissance de la voix de {PRENOM} : activée.")
    except Exception as e:
        print(f"[Reconnaissance de la voix indisponible : {type(e).__name__}: {e}]")
        _EMPREINTE = None


def _vraie_interruption(r, source, debut):
    """Jarvis est en pause : on enregistre ta phrase (sans sa voix par-dessus), on vérifie que c'est toi
    et que tu dis vraiment quelque chose. Renvoie le texte si c'est une vraie interruption, sinon None."""
    import speech_recognition as sr
    cps = source.SAMPLE_RATE / source.CHUNK
    morceaux, calmes, parle = [debut], 0, 0
    fin = time.time() + 20
    while time.time() < fin:
        d = source.stream.read(source.CHUNK)
        morceaux.append(d)
        fort = _rms(d) >= r.energy_threshold
        calmes = 0 if fort else calmes + 1
        parle += fort
        if calmes >= int(1.0 * cps):
            break
    if parle < int(0.25 * cps):
        print("   (fausse alerte : plus rien entendu, je reprends)")
        return None
    audio = sr.AudioData(b"".join(morceaux), source.SAMPLE_RATE, source.SAMPLE_WIDTH)
    texte = _transcrire(r, audio)
    if not texte:
        print("   (fausse alerte : rien de compréhensible, je reprends)")
        return None
    if not MOTS_STOP.search(texte) and len(texte.split()) < 2:
        print(f"   (fausse alerte : « {texte} », je reprends)")
        return None
    duree = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
    s = _score(audio)
    if s is not None and s < _seuil_pour(duree) - 0.08:
        print(f"   (« {texte} » : ce n'est pas ta voix ({s:.2f}), je reprends)")
        return None
    _DERNIER["audio"] = audio
    return texte


def _surveiller_interruption(arret):
    """Tourne PENDANT que Jarvis parle. Si ça ressemble à TA voix, il se met en PAUSE et t'écoute.
    Vraie demande -> il s'arrête et te répond. Fausse alerte (bruit, sa propre voix, autre personne) ->
    il reprend sa phrase là où il en était."""
    import speech_recognition as sr
    r, source = _MIC["r"], _MIC["source"]
    cps = source.SAMPLE_RATE / source.CHUNK              # morceaux de son par seconde
    taille = int(0.9 * cps)                              # on analyse les 0,9 dernières secondes
    tampon = collections.deque(maxlen=taille)
    seuil = r.energy_threshold * FACTEUR_INTERRUPTION
    depuis, reussites = 0, 0
    try:
        while not arret.is_set():
            tampon.append(source.stream.read(source.CHUNK))
            depuis += 1
            if len(tampon) < taille or depuis < int(0.3 * cps):
                continue                                 # on teste environ 3 fois par seconde
            if sum(1 for d in tampon if _rms(d) >= seuil) < taille * 0.4:
                reussites = 0
                continue
            depuis = 0
            audio = sr.AudioData(b"".join(tampon), source.SAMPLE_RATE, source.SAMPLE_WIDTH)
            s = _score(audio)
            if DEBUG_INTERRUPTION and s is not None:
                print(f"   [coupure ? ressemblance {s:.2f} / seuil {SEUIL_INTERRUPTION:.2f}]")
            reussites = reussites + 1 if (s is not None and s >= SEUIL_INTERRUPTION) else 0
            if reussites >= 2 or (s is not None and s >= SEUIL_INTERRUPTION + 0.08):
                _pause_lecture(True)
                print("   (pause : je vérifie si tu me parles)")
                texte = _vraie_interruption(r, source, audio.frame_data)
                if texte:
                    _INTERROMPU["oui"], _INTERROMPU["audio"], _INTERROMPU["texte"] = True, audio.frame_data, texte
                    _couper_lecture()
                    _REPRISE.set()
                    return
                _pause_lecture(False)                    # fausse alerte : il reprend où il en était
                tampon.clear()
                reussites, depuis = 0, 0
    except Exception as e:
        _pause_lecture(False)
        print(f"[Écoute pendant la parole arrêtée : {type(e).__name__}: {e}]")


# =============================== OREILLES ===================================
def _suite_phrase(recognizer, source, debut_audio, limite=DUREE_MAX_PHRASE):
    """Continue d'enregistrer ta phrase déjà commencée (quand tu as coupé Jarvis), jusqu'au silence."""
    import speech_recognition as sr
    cps = source.SAMPLE_RATE / source.CHUNK
    morceaux, calmes = [debut_audio], 0
    fin = time.time() + limite
    while time.time() < fin:
        d = source.stream.read(source.CHUNK)
        morceaux.append(d)
        calmes = calmes + 1 if _rms(d) < recognizer.energy_threshold else 0
        if calmes >= int(PAUSE_FIN_PHRASE * cps):
            break
    return sr.AudioData(b"".join(morceaux), source.SAMPLE_RATE, source.SAMPLE_WIDTH)


# Une phrase qui se termine par un de ces mots n'est sûrement pas finie : on attend la suite.
_PHRASE_PAS_FINIE = re.compile(
    r"(\b(et|puis|ensuite|après|aussi|ou|mais|donc|alors|de|du|des|le|la|les|l|un|une|à|au|aux|en|dans|"
    r"sur|sous|pour|par|avec|sans|qui|que|qu|quoi|dont|où|si|comme|mon|ma|mes|ton|ta|tes|son|sa|ses|ce|cet|"
    r"cette|ces|euh|hum|genre|c'est|il|elle|je|tu|on|nous|vous)|,)\s*$", re.IGNORECASE)


def _joindre(a, b):
    import speech_recognition as sr
    return sr.AudioData(a.frame_data + b.frame_data, a.sample_rate, a.sample_width)


def _prolonger(recognizer, source, audio, attente):
    """Après une pause, écoute encore `attente` secondes : si tu reprends la parole, on recolle la suite."""
    import speech_recognition as sr
    recu = False
    while True:
        duree = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
        reste = DUREE_MAX_PHRASE - duree
        if reste < 2:
            return audio, recu
        try:
            suite = recognizer.listen(source, timeout=attente, phrase_time_limit=reste)
        except sr.WaitTimeoutError:
            return audio, recu
        audio, recu = _joindre(audio, suite), True
        print("   (tu continues : je t'écoute toujours)")


def _transcrire(recognizer, audio):
    import speech_recognition as sr
    try:
        return recognizer.recognize_google(audio, language=LANGUE)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"[Reconnaissance vocale indisponible (internet ?) : {e}]")
        return None


def ecouter(recognizer, source, timeout=5, limite=DUREE_MAX_PHRASE, prefixe=None):
    """Écoute une phrase ENTIÈRE et renvoie le texte (ou None).
    Une courte pause ne coupe pas : il attend de voir si tu continues, surtout si ta phrase
    s'arrête sur « et », « puis », « de »...
    prefixe = début de phrase déjà enregistré (quand tu coupes Jarvis en pleine parole)."""
    import speech_recognition as sr
    if prefixe is not None:
        audio = _suite_phrase(recognizer, source, prefixe, limite)
    else:
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=limite)
        except sr.WaitTimeoutError:
            return None
    audio, _ = _prolonger(recognizer, source, audio, ATTENTE_SUITE)
    texte = _transcrire(recognizer, audio)
    for _ in range(3):                    # la phrase a l'air coupée : on laisse le temps de finir
        if not texte or not _PHRASE_PAS_FINIE.search(texte):
            break
        print(f"   (« {texte} »... j'attends la suite)")
        audio, prolonge = _prolonger(recognizer, source, audio, ATTENTE_SUITE_LONGUE)
        if not prolonge:
            break
        texte = _transcrire(recognizer, audio) or texte
    _DERNIER["audio"] = audio
    return texte


def _ecouter_suite(recognizer, source, timeout=8, limite=DUREE_MAX_PHRASE):
    """Comme ecouter(), mais reprend ta phrase si tu viens de couper Jarvis."""
    if _INTERROMPU["oui"]:
        pre, texte, _INTERROMPU["oui"] = _INTERROMPU["audio"], _INTERROMPU["texte"], False
        return texte or ecouter(recognizer, source, prefixe=pre, limite=limite)
    return ecouter(recognizer, source, timeout=timeout, limite=limite)


def _confirmer(question):
    """Demande une confirmation vocale (ou au clavier en mode test)."""
    parler(question + " Dis oui pour confirmer.")
    r, source = _MIC["r"], _MIC["source"]
    if r is None:
        return input(question + " (o/n) ").strip().lower().startswith("o")
    rep = _ecouter_suite(r, source, timeout=8, limite=5)
    return bool(rep) and bool(re.search(r"\b(oui|ok|okay|d'accord|vas-y|confirme)\b", rep.lower()))


# =============================== OUTILS PC ==================================
# Ces fonctions sont données au cerveau (Gemini ou Groq) : il décide lui-même laquelle appeler.

DANGEREUX = re.compile(
    r"\bsudo\b|\brm\b|\brmdir\b|\bmkfs|\bdd\b|\bshred\b|\bshutdown\b|\breboot\b|\bpoweroff\b|\bhalt\b|"
    r"\bchmod\b|\bchown\b|\bkill(all)?\b|\bpkill\b|\bmv\b|\bpasswd\b|:\(\)\s*\{|>\s*/dev/|"
    r"\bapt(-get)?\s+(remove|purge|autoremove)|\bsnap\s+remove|\bpip\s+uninstall|\|\s*(ba)?sh\b|"
    r"\bsystemctl\s+(stop|disable|mask)",
    re.IGNORECASE,
)

# "sudo" utilisé comme commande (début, ou après ; & | ou $( ), pas dans un simple texte
_SUDO_CMD = re.compile(r"(^|[;&|]\s*|\$\(\s*)sudo\s")


def _autorise(commande):
    if CONFIRMER_ACTIONS_SENSIBLES and DANGEREUX.search(commande):
        return _confirmer(f"Cette commande est sensible : {commande[:100]}.")
    return True


def _chemin(chemin):
    """Chemin absolu : ~ = dossier personnel, un chemin relatif part du dossier de travail de Jarvis."""
    chemin = os.path.expanduser((chemin or ".").strip())
    if chemin.startswith("file://"):
        chemin = chemin[7:]
    return os.path.normpath(chemin if os.path.isabs(chemin) else os.path.join(_CWD["d"], chemin))


def _cwd_valide():
    if not os.path.isdir(_CWD["d"]):
        _CWD["d"] = os.path.expanduser("~")
    return _CWD["d"]


_MARQUE_PWD = "__JARVIS_PWD__"


def executer_commande(commande: str) -> str:
    """Exécute une commande shell (bash) en arrière-plan, SANS fenêtre visible, et renvoie sa sortie.
    Sert à tout : fichiers et dossiers, installation de paquets (sudo fonctionne), infos système,
    réseau, git, npm, pip, scripts, etc. Attend la fin de la commande.
    Elle s'exécute dans le dossier de travail actuel, et un « cd » est conservé pour les commandes suivantes.
    Si l'utilisateur veut VOIR la commande dans un terminal, utilise plutôt taper_dans_terminal.
    Pour un serveur ou un programme qui ne s'arrête pas, utilise lancer_en_arriere_plan."""
    if not _autorise(commande):
        return "REFUSÉ : l'utilisateur n'a pas confirmé cette commande."
    cmd, kw = commande, {"stdin": subprocess.DEVNULL}
    if _SUDO_MDP and _SUDO_CMD.search(commande):
        cmd = _SUDO_CMD.sub(r"\1sudo -S -p '' ", commande)
        kw = {"input": _SUDO_MDP + "\n"}
    script = f'{cmd}\n__rc=$?\nprintf "\\n{_MARQUE_PWD}%s" "$PWD"\nexit $__rc'
    t0 = time.time()
    try:
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                           timeout=DELAI_COMMANDE, cwd=_cwd_valide(), **kw)
        print(f"   [commande exécutée en {time.time() - t0:.1f}s : {commande[:70]}]")
        out, _, pwd = p.stdout.rpartition(_MARQUE_PWD)
        if _ and os.path.isdir(pwd.strip()):
            _CWD["d"] = pwd.strip()
        else:
            out = p.stdout
        sortie = (out + p.stderr).strip()
        if len(sortie) > 6000:
            sortie = sortie[:2500] + "\n[...]\n" + sortie[-3000:]
        return f"code de retour {p.returncode} (dossier : {_CWD['d']})\n{sortie}"
    except subprocess.TimeoutExpired:
        return (f"La commande dépasse {DELAI_COMMANDE} secondes et a été arrêtée. "
                "Utilise lancer_en_arriere_plan si elle doit tourner longtemps.")
    except Exception as e:
        return f"Erreur : {e}"


def lancer_en_arriere_plan(commande: str) -> str:
    """Lance une commande shell en arrière-plan sans attendre sa fin (serveur web, npm run dev,
    téléchargement long, programme graphique). Renvoie immédiatement le chemin d'un fichier journal :
    lis-le avec lire_fichier quelques secondes après pour vérifier que tout va bien."""
    if not _autorise(commande):
        return "REFUSÉ : l'utilisateur n'a pas confirmé cette commande."
    journal = os.path.join(tempfile.gettempdir(), f"jarvis-fond-{int(time.time())}.log")
    try:
        with open(journal, "w") as f:
            p = subprocess.Popen(["bash", "-c", commande], stdout=f, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True, cwd=_cwd_valide())
        return f"Commande lancée en arrière-plan (pid {p.pid}). Journal : {journal}"
    except Exception as e:
        return f"Erreur : {e}"


def changer_dossier(chemin: str) -> str:
    """Change le dossier de travail de Jarvis (comme « cd »). Les chemins relatifs, les commandes
    et le terminal partent ensuite de ce dossier. Crée le dossier s'il n'existe pas."""
    d = _chemin(chemin)
    try:
        os.makedirs(d, exist_ok=True)
        _CWD["d"] = d
        return f"Dossier de travail : {d}"
    except Exception as e:
        return f"Erreur : {e}"


def creer_dossier(chemin: str) -> str:
    """Crée un dossier (et les dossiers parents manquants)."""
    d = _chemin(chemin)
    try:
        os.makedirs(d, exist_ok=True)
        return f"Dossier créé : {d}"
    except Exception as e:
        return f"Erreur : {e}"


_IGNORES = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".next", ".dart_tool",
            ".idea", ".vscode", "target", ".cache"}


def lister_dossier(chemin: str = ".", profondeur: int = 2) -> str:
    """Liste le contenu d'un dossier sous forme d'arbre (profondeur 1 à 4). Ignore node_modules, .git, venv..."""
    racine = _chemin(chemin)
    if not os.path.isdir(racine):
        return f"Ce dossier n'existe pas : {racine}"
    lignes, prof = [racine + "/"], max(1, min(4, int(profondeur)))
    for d, dossiers, fichiers in os.walk(racine):
        niveau = d[len(racine):].count(os.sep)
        dossiers[:] = sorted(x for x in dossiers if x not in _IGNORES)
        if niveau >= prof:
            dossiers[:] = []
            continue
        retrait = "  " * (niveau + 1)
        if d != racine:
            lignes.append("  " * niveau + os.path.basename(d) + "/")
        lignes += [retrait + f for f in sorted(fichiers)[:60]]
        if len(lignes) > 300:
            lignes.append("[... liste tronquée]")
            break
    return "\n".join(lignes)


def lire_fichier(chemin: str) -> str:
    """Lit le contenu d'un fichier texte (jusqu'à 20000 caractères), avec les numéros de ligne.
    ~ désigne le dossier personnel ; un chemin relatif part du dossier de travail."""
    try:
        with open(_chemin(chemin), encoding="utf-8", errors="replace") as f:
            texte = f.read(20001)
        lignes = texte[:20000].split("\n")
        corps = "\n".join(f"{i:4} | {l}" for i, l in enumerate(lignes, 1))
        return corps + ("\n[... fichier tronqué à 20000 caractères]" if len(texte) > 20000 else "")
    except Exception as e:
        return f"Erreur : {e}"


def ecrire_fichier(chemin: str, contenu: str, ajouter: bool = False) -> str:
    """Écrit un fichier COMPLET (ou ajoute à la fin si ajouter=True). Crée les dossiers manquants.
    Utilise-le pour créer des fichiers de code, des pages web (HTML, CSS, JS), des scripts, etc.
    Pour changer seulement un passage d'un fichier existant, utilise plutôt modifier_fichier."""
    chemin = _chemin(chemin)
    if CONFIRMER_ACTIONS_SENSIBLES and os.path.exists(chemin) and not ajouter:
        if not _confirmer(f"Le fichier {os.path.basename(chemin)} existe déjà. Je l'écrase ?"):
            return "REFUSÉ : l'utilisateur n'a pas confirmé l'écrasement."
    try:
        dossier = os.path.dirname(chemin)
        if dossier:
            os.makedirs(dossier, exist_ok=True)
        with open(chemin, "a" if ajouter else "w", encoding="utf-8") as f:
            f.write(contenu)
        return f"Fichier écrit : {chemin} ({len(contenu)} caractères)"
    except Exception as e:
        return f"Erreur : {e}"


def modifier_fichier(chemin: str, ancien_texte: str, nouveau_texte: str) -> str:
    """Remplace dans un fichier un passage exact (ancien_texte, copié tel quel depuis lire_fichier,
    SANS les numéros de ligne) par nouveau_texte. Sert à corriger un bug ou modifier du code
    sans réécrire tout le fichier. L'éditeur de code ouvert se met à jour tout seul."""
    chemin = _chemin(chemin)
    try:
        with open(chemin, encoding="utf-8") as f:
            texte = f.read()
    except Exception as e:
        return f"Erreur : {e}"
    n = texte.count(ancien_texte)
    if n == 0:
        return ("Passage introuvable : relis le fichier avec lire_fichier et copie le texte exact "
                "(espaces et indentation compris, sans les numéros de ligne).")
    if n > 1:
        return f"Ce passage apparaît {n} fois : donne un passage plus long pour qu'il soit unique."
    try:
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(texte.replace(ancien_texte, nouveau_texte, 1))
        return f"Fichier modifié : {chemin}"
    except Exception as e:
        return f"Erreur : {e}"


def verifier_code(chemin: str) -> str:
    """Vérifie un fichier de code (ou tous ceux d'un dossier) : erreurs de syntaxe et problèmes
    détectables automatiquement (Python, JavaScript, TypeScript, shell, JSON, PHP, Dart).
    Après ça, lis aussi le code avec lire_fichier pour juger la logique."""
    cible = _chemin(chemin)
    if os.path.isdir(cible):
        fichiers = []
        for d, dossiers, noms in os.walk(cible):
            dossiers[:] = [x for x in dossiers if x not in _IGNORES]
            fichiers += [os.path.join(d, n) for n in noms]
        fichiers = [f for f in fichiers if os.path.splitext(f)[1] in _VERIFS][:40]
        if not fichiers:
            return "Aucun fichier de code que je sais vérifier automatiquement dans ce dossier."
        return "\n\n".join(_verifier_un(f) for f in fichiers)
    if not os.path.exists(cible):
        return f"Fichier introuvable : {cible}"
    return _verifier_un(cible)


def _verif_json(f):
    try:
        with open(f, encoding="utf-8") as h:
            json.load(h)
        return None
    except Exception as e:
        return str(e)


_VERIFS = {
    ".py": [["python3", "-m", "py_compile"], ["python3", "-m", "pyflakes"], ["ruff", "check", "--quiet"]],
    ".js": [["node", "--check"]], ".mjs": [["node", "--check"]], ".cjs": [["node", "--check"]],
    ".ts": [["npx", "--no-install", "tsc", "--noEmit", "--skipLibCheck"]],
    ".sh": [["bash", "-n"]], ".php": [["php", "-l"]], ".dart": [["dart", "analyze"]],
    ".json": _verif_json, ".html": None, ".css": None,
}


def _verifier_un(fichier):
    regles = _VERIFS.get(os.path.splitext(fichier)[1])
    nom = os.path.relpath(fichier, _CWD["d"])
    if regles is None:
        return f"{nom} : pas de vérificateur automatique, lis le fichier pour le relire toi-même."
    if callable(regles):
        err = regles(fichier)
        return f"{nom} : {'OK' if not err else 'ERREUR ' + err}"
    resultats = []
    for cmd in regles:
        if not shutil.which(cmd[0]):
            continue
        try:
            p = subprocess.run(cmd + [fichier], capture_output=True, text=True, timeout=60,
                               cwd=os.path.dirname(fichier))
        except Exception as e:
            resultats.append(f"{cmd[0]} : impossible ({e})")
            continue
        sortie = (p.stdout + p.stderr).strip()
        if "No module named" in sortie:
            continue                                   # outil non installé (pyflakes...) : on passe
        resultats.append(f"{' '.join(cmd[:3])} : " + ("OK" if p.returncode == 0 and not sortie
                                                        else sortie[:1500]))
    return f"{nom} :\n  " + ("\n  ".join(resultats) if resultats else "aucun outil de vérification installé")


# ----------------------- Terminal visible piloté par Jarvis ------------------
# Une vraie fenêtre de terminal s'ouvre ; Jarvis y « tape » les commandes (elles s'affichent et
# s'exécutent sous tes yeux) et lit leur résultat. Aucun outil de clavier virtuel n'est nécessaire,
# donc ça marche aussi sous Wayland.
_SCRIPT_TERMINAL = r'''
fifo="$1"; log="$2"; cd "$3" 2>/dev/null || cd ~
echo $$ > "$fifo.pid"
printf '\033]0;Terminal de Jarvis\007'
echo "=== Terminal de Jarvis : les commandes que tu me demandes s'exécutent ici ==="
exec 3<>"$fifo"
tail -n0 -F "$log" 2>/dev/null | grep --line-buffered -v '^__JARVIS_FIN__' &
while IFS= read -r ligne <&3; do
  id="${ligne%%:*}"; mode="${ligne#*:}"; mode="${mode%%:*}"; b64="${ligne#*:*:}"
  cmd="$(printf '%s' "$b64" | base64 -d)"
  printf '\n\033[1;32m%s\033[0m$ %s\n' "$PWD" "$cmd" >> "$log"
  if [ "$mode" = "i" ]; then sleep 0.3; eval "$cmd"; else eval "$cmd" >> "$log" 2>&1 < /dev/tty; fi
  echo "__JARVIS_FIN__ $id $? $PWD" >> "$log"
done
'''


def _fichiers_terminal():
    d = os.path.join(tempfile.gettempdir(), f"jarvis-terminal-{os.getuid()}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "commandes.fifo"), os.path.join(d, "sortie.log"), os.path.join(d, "terminal.sh")


def _ouvrir_fifo():
    """Renvoie un descripteur d'écriture vers le terminal de Jarvis, ou None s'il n'est pas ouvert."""
    fifo = _fichiers_terminal()[0]
    if not os.path.exists(fifo):
        return None
    try:
        return os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return None


def _commande_terminal(script, args, dossier):
    """Commande pour ouvrir une fenêtre de terminal qui exécute `bash script args`."""
    interne = ["bash", script] + args
    for prog, avant in (("ptyxis", ["--new-window", "-d", dossier, "--"]),
                        ("gnome-terminal", ["--window", f"--working-directory={dossier}", "--"]),
                        ("kgx", ["--"]), ("konsole", ["-e"]), ("xfce4-terminal", ["-x"]),
                        ("tilix", ["-e"]), ("alacritty", ["-e"]), ("kitty", []),
                        ("xterm", ["-e"]), ("x-terminal-emulator", ["-e"])):
        if shutil.which(prog):
            return [prog] + avant + interne
    return None


def ouvrir_terminal(dossier: str = "") -> str:
    """Ouvre une fenêtre de terminal VISIBLE contrôlée par Jarvis (dans le dossier donné, sinon le
    dossier de travail). Ensuite, utilise taper_dans_terminal pour y taper et exécuter des commandes.
    À utiliser quand l'utilisateur dit « ouvre le terminal » (et éventuellement « tape ... »)."""
    if dossier:
        changer_dossier(dossier)
    fd = _ouvrir_fifo()
    if fd is not None:
        os.close(fd)
        if dossier:
            return taper_dans_terminal(f"cd {shlex.quote(_CWD['d'])}", 5)
        return "Le terminal de Jarvis est déjà ouvert."
    fifo, log, script = _fichiers_terminal()
    if os.path.exists(fifo):
        os.remove(fifo)
    os.mkfifo(fifo, 0o600)
    with open(log, "w"):
        pass
    with open(script, "w") as f:
        f.write(_SCRIPT_TERMINAL)
    cmd = _commande_terminal(script, [fifo, log, _cwd_valide()], _CWD["d"])
    if not cmd:
        return "Aucun terminal graphique trouvé (installe gnome-terminal ou ptyxis)."
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):                        # on attend que la fenêtre soit prête (5 s max)
        time.sleep(0.1)
        fd = _ouvrir_fifo()
        if fd is not None:
            os.close(fd)
            return f"Terminal ouvert dans {_CWD['d']}."
    return "J'ai lancé le terminal mais il ne répond pas."


def taper_dans_terminal(commande: str, attendre: int = 30, interactif: bool = False) -> str:
    """Tape et exécute une commande dans le terminal VISIBLE de Jarvis (l'ouvre si besoin), puis
    renvoie ce qu'elle affiche. Le « cd » est conservé d'une commande à l'autre.
    attendre = secondes max d'attente du résultat (la commande continue si elle est plus longue).
    interactif=True pour un programme plein écran ou qui pose des questions (nano, htop, ssh, python...) :
    l'utilisateur le voit et peut l'utiliser lui-même, mais tu ne lis pas sa sortie."""
    if not _autorise(commande):
        return "REFUSÉ : l'utilisateur n'a pas confirmé cette commande."
    fd = _ouvrir_fifo()
    if fd is None:
        r = ouvrir_terminal()
        fd = _ouvrir_fifo()
        if fd is None:
            return r
    import base64
    fifo, log, _ = _fichiers_terminal()
    ident = f"{time.time():.6f}".replace(".", "")
    debut = os.path.getsize(log)
    ligne = f"{ident}:{'i' if interactif else 'n'}:{base64.b64encode(commande.encode()).decode()}\n"
    try:
        os.write(fd, ligne.encode())
    finally:
        os.close(fd)
    print(f"   [tapé dans le terminal : {commande[:70]}]")
    if interactif:
        return "Commande lancée dans le terminal (mode interactif : l'utilisateur la voit et l'utilise)."
    fin_attente = time.time() + max(1, min(int(attendre), DELAI_COMMANDE))
    marque = f"__JARVIS_FIN__ {ident} "
    while time.time() < fin_attente:
        time.sleep(0.2)
        with open(log, encoding="utf-8", errors="replace") as f:
            f.seek(debut)
            texte = f.read()
        if marque in texte:
            sortie, _, reste = texte.partition(marque)
            code, _, pwd = reste.split("\n", 1)[0].partition(" ")
            if os.path.isdir(pwd.strip()):
                _CWD["d"] = pwd.strip()
            sortie = re.sub(r"\x1b\[[0-9;]*m", "", sortie).strip()
            sortie = "\n".join(sortie.split("\n")[1:])          # sans la ligne « dossier$ commande »
            if len(sortie) > 6000:
                sortie = sortie[:2500] + "\n[...]\n" + sortie[-3000:]
            return f"code de retour {code} (dossier : {_CWD['d']})\n{sortie}"
    return ("La commande tourne toujours dans le terminal (visible par l'utilisateur). "
            "Ne la relance pas ; elle continue toute seule.")


# ----------------------------- Éditeurs de code -----------------------------
_EDITEURS = (  # (noms possibles, commandes, dossier de configuration)
    (("devin", "windsurf"), ("windsurf", "devin-desktop", "/usr/share/devin-desktop/devin-desktop"),
     ("Devin", "Windsurf")),
    (("vscode", "vs code", "visual studio code", "code"), ("code", "/snap/bin/code"), ("Code",)),
    (("cursor",), ("cursor",), ("Cursor",)),
    (("vscodium", "codium"), ("codium",), ("VSCodium",)),
)


def ouvrir_dans_editeur(chemin: str, editeur: str = "") -> str:
    """Ouvre un fichier ou un dossier de projet dans l'éditeur de code (Devin/Windsurf, VS Code,
    Cursor). editeur = nom de l'éditeur voulu, vide = celui qui est installé."""
    cible = _chemin(chemin)
    if not os.path.exists(cible):
        return f"Ce chemin n'existe pas : {cible}"
    voulu = editeur.lower().strip()
    candidats = [e for e in _EDITEURS if not voulu or any(n in voulu or voulu in n for n in e[0])] or _EDITEURS
    for _, commandes, _ in candidats:
        for c in commandes:
            exe = shutil.which(c) or (c if os.path.isabs(c) and os.path.exists(c) else None)
            if exe:
                subprocess.Popen([exe, cible], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
                if os.path.isdir(cible):
                    _CWD["d"] = cible
                return f"Ouvert dans l'éditeur ({os.path.basename(exe)}) : {cible}"
    return "Aucun éditeur de code trouvé (Devin, Windsurf, VS Code, Cursor)."


def _fichiers_recents(dossier, n=6):
    trouves = []
    for d, dossiers, noms in os.walk(dossier):
        dossiers[:] = [x for x in dossiers if x not in _IGNORES and not x.startswith(".")]
        for nom in noms:
            p = os.path.join(d, nom)
            try:
                trouves.append((os.path.getmtime(p), p))
            except OSError:
                pass
        if len(trouves) > 5000:
            break
    return [p for _, p in sorted(trouves, reverse=True)[:n]]


def contexte_editeur() -> str:
    """Dit quel projet est ouvert dans l'éditeur de code (Devin/Windsurf, VS Code, Cursor), les derniers
    fichiers ouverts dedans et les derniers fichiers modifiés. À utiliser quand l'utilisateur parle de
    « mon code », « ce fichier », « le projet » sans donner de chemin."""
    import sqlite3
    import urllib.parse
    espaces = []
    for _, _, configs in _EDITEURS:
        for nom in configs:
            base = os.path.expanduser(f"~/.config/{nom}/User/workspaceStorage")
            for d in glob.glob(os.path.join(base, "*")):
                ws, db = os.path.join(d, "workspace.json"), os.path.join(d, "state.vscdb")
                if os.path.exists(ws) and os.path.exists(db):
                    espaces.append((os.path.getmtime(db), nom, ws, db))
    if not espaces:
        return "Je ne trouve aucun projet ouvert dans un éditeur de code."
    lignes = []
    for _, nom, ws, db in sorted(espaces, reverse=True)[:2]:
        try:
            with open(ws) as f:
                info = json.load(f)
            dossier = urllib.parse.unquote(urllib.parse.urlparse(info.get("folder") or info.get("workspace", "")).path)
        except Exception:
            continue
        ouverts = []
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
            ligne = con.execute("select value from ItemTable where key='history.entries'").fetchone()
            con.close()
            for e in json.loads(ligne[0]) if ligne else []:
                res = (e.get("editor") or {}).get("resource", "")
                if res.startswith("file://"):
                    ouverts.append(urllib.parse.unquote(res[7:]))
        except Exception:
            pass
        lignes.append(f"Éditeur {nom} - projet : {dossier}")
        if ouverts:
            lignes.append("  derniers fichiers ouverts (le 1er est le plus récent) : " + ", ".join(ouverts[:6]))
        if os.path.isdir(dossier):
            lignes.append("  derniers fichiers modifiés : " + ", ".join(_fichiers_recents(dossier)))
    return "\n".join(lignes) or "Je ne trouve aucun projet ouvert dans un éditeur de code."


def _dossiers_applications():
    return ["/usr/share/applications", "/usr/local/share/applications",
            os.path.expanduser("~/.local/share/applications"),
            "/var/lib/snapd/desktop/applications",
            "/var/lib/flatpak/exports/share/applications",
            os.path.expanduser("~/.local/share/flatpak/exports/share/applications")]


def _lire_desktop(chemin):
    """Lit un fichier .desktop. Renvoie (liste de noms, commande Exec) ou None."""
    noms, exec_, dans = [], None, False
    try:
        with open(chemin, encoding="utf-8", errors="ignore") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne.startswith("["):
                    if ligne == "[Desktop Entry]":
                        dans = True
                        continue
                    if dans:
                        break
                    continue
                if not dans:
                    continue
                if ligne.lower() == "nodisplay=true":
                    return None
                if ligne.startswith(("Name=", "Name[fr]=", "GenericName=", "GenericName[fr]=")):
                    noms.append(ligne.split("=", 1)[1].strip().lower())
                elif ligne.startswith("Exec=") and exec_ is None:
                    exec_ = ligne.split("=", 1)[1].strip()
    except OSError:
        return None
    return (noms, exec_) if exec_ else None


def _trouver_application(nom, dossiers=None):
    """Cherche dans les applications installées (fichiers .desktop) la commande correspondant au nom."""
    nom = nom.lower().strip()
    meilleur, score_max = None, 0
    for d in (dossiers or _dossiers_applications()):
        for chemin in glob.glob(os.path.join(d, "*.desktop")):
            infos = _lire_desktop(chemin)
            if not infos:
                continue
            noms, exec_ = infos
            base = os.path.basename(chemin)[:-8].lower()
            score = 0
            for n in noms + [base]:
                if n == nom:
                    score = max(score, 3)
                elif n.startswith(nom):
                    score = max(score, 2)
                elif nom in n:
                    score = max(score, 1)
            if score > score_max:
                meilleur, score_max = exec_, score
    return meilleur


def ouvrir_application(nom: str) -> str:
    """Ouvre une application installée sur l'ordinateur à partir de son nom
    (ex : Firefox, Chrome, calculatrice, Spotify, VLC, fichiers, paramètres).
    Pour un terminal, utilise ouvrir_terminal ; pour un éditeur de code, ouvrir_dans_editeur."""
    if re.search(r"\b(terminal|console|shell|ptyxis)\b", nom.lower()):
        return ouvrir_terminal()
    exec_ = _trouver_application(nom)
    try:
        if exec_:
            cmd = shlex.split(re.sub(r"%[a-zA-Z%]", "", exec_))
        else:
            direct = shutil.which(nom.lower().replace(" ", "-"))
            if not direct:
                return f"Aucune application nommée « {nom} » n'est installée."
            cmd = [direct]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        return f"Application « {nom} » lancée."
    except Exception as e:
        return f"Erreur au lancement de {nom} : {e}"


def ouvrir_url(url: str) -> str:
    """Ouvre une adresse web (site, page, vidéo) ou un fichier local (ex : index.html d'un site
    que tu viens de créer) dans le navigateur par défaut."""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        local = _chemin(url)
        url = ("file://" + local) if os.path.exists(local) else "https://" + url
    webbrowser.open(url)
    return f"Adresse ouverte : {url}"


def chercher_sur_internet(requete: str) -> str:
    """Ouvre une recherche Google dans le navigateur pour la requête donnée."""
    import urllib.parse
    webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote(requete))
    return f"Recherche lancée : {requete}"


def regler_volume(pourcentage: int) -> str:
    """Règle le volume général de l'ordinateur à un pourcentage précis (0 à 100)."""
    p = max(0, min(150, int(pourcentage)))
    if shutil.which("pactl"):
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{p}%"], check=False)
        return f"Volume réglé à {p} pour cent."
    if shutil.which("amixer"):
        subprocess.run(["amixer", "set", "Master", f"{p}%"], check=False)
        return f"Volume réglé à {p} pour cent."
    return "Impossible de régler le volume (pactl et amixer introuvables)."


def date_et_heure() -> str:
    """Donne la date et l'heure actuelles de l'ordinateur."""
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    n = datetime.datetime.now()
    return f"{jours[n.weekday()]} {n.strftime('%d/%m/%Y')}, {n.strftime('%H:%M')}"


# ------------------------- Fermer, supprimer, quitter ------------------------
import signal as _signal

_ETAT = {"quitter": False}


def _ancetres():
    """PID de Jarvis et de tous ses parents (le terminal / l'éditeur dans lequel il tourne)."""
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    return pids


def _vivant(pid):
    """True si le processus tourne encore (un « zombie » déjà arrêté compte comme fermé)."""
    try:
        os.waitpid(pid, os.WNOHANG)       # récupère les applis lancées par Jarvis lui-même
    except (ChildProcessError, OSError):
        pass
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        return False


def _tuer(pids, attendre=3.0):
    """SIGTERM (fermeture propre), puis SIGKILL pour ceux qui résistent. Renvoie le nb de processus arrêtés."""
    for p in pids:
        try:
            os.kill(p, _signal.SIGTERM)
        except OSError:
            pass
    fin = time.time() + attendre
    while time.time() < fin and any(_vivant(p) for p in pids):
        time.sleep(0.1)
    for p in pids:
        if _vivant(p):
            try:
                os.kill(p, _signal.SIGKILL)
            except OSError:
                pass
    time.sleep(0.2)
    return sum(1 for p in pids if not _vivant(p))


def fermer_terminal() -> str:
    """Ferme la fenêtre du terminal de Jarvis (celui ouvert avec ouvrir_terminal)."""
    fifo = _fichiers_terminal()[0]
    try:
        with open(fifo + ".pid") as f:
            pid = int(f.read().strip())
    except Exception:
        pid = None
    if not pid or not _vivant(pid):
        return "Le terminal de Jarvis n'était pas ouvert (rien à fermer)."
    try:
        os.killpg(os.getpgid(pid), _signal.SIGHUP)
    except OSError:
        pass
    _tuer([pid], 2)
    if _vivant(pid):
        return "ÉCHEC : le terminal de Jarvis est toujours ouvert."
    if os.path.exists(fifo):
        os.remove(fifo)
    return "Terminal de Jarvis fermé (vérifié : il ne tourne plus)."


def _noms_processus(nom):
    """Noms de programmes possibles pour une application (« devin » -> devin-desktop, etc.)."""
    nom = nom.lower().strip()
    noms = {nom, nom.replace(" ", "-"), nom.replace(" ", "")}
    exec_ = _trouver_application(nom)
    if exec_:
        try:
            noms.add(os.path.basename(shlex.split(re.sub(r"%[a-zA-Z%]", "", exec_))[0]).lower())
        except Exception:
            pass
    alias = {"chrome": "chrome", "google chrome": "chrome", "vscode": "code", "vs code": "code",
             "visual studio code": "code", "devin": "devin-desktop", "windsurf": "devin-desktop",
             "calculatrice": "gnome-calculator", "fichiers": "nautilus", "navigateur": "firefox"}
    if nom in alias:
        noms.add(alias[nom])
    return {n for n in noms if len(n) >= 3}


def fermer_application(nom: str, forcer: bool = False) -> str:
    """Ferme (quitte) une application ouverte : Firefox, Chrome, Spotify, VLC, Devin, VS Code, calculatrice...
    Ferme TOUTES ses fenêtres. Pour le terminal de Jarvis, utilise fermer_terminal.
    forcer=True seulement si l'utilisateur confirme vouloir fermer l'application dans laquelle Jarvis
    tourne lui-même (Jarvis s'arrêtera alors aussi)."""
    if re.search(r"\b(terminal|console|ptyxis)\b", nom.lower()) and not forcer:
        return fermer_terminal()
    noms = _noms_processus(nom)
    moi = _ancetres()
    trouves, proteges = [], []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        pid = int(d)
        try:
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip().lower()
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                exe = os.path.basename(f.read().split(b"\0")[0].decode(errors="ignore")).lower()
            if os.stat(f"/proc/{pid}").st_uid != os.getuid():
                continue
        except Exception:
            continue
        if any(n == comm or n == exe or (len(n) >= 4 and comm.startswith(n[:15])) for n in noms):
            (proteges if pid in moi else trouves).append(pid)
    if proteges and not forcer:
        return (f"ATTENTION : Jarvis tourne à l'intérieur de « {nom} ». Le fermer arrêterait Jarvis aussi. "
                "Demande confirmation à l'utilisateur ; s'il confirme, rappelle cet outil avec forcer=True.")
    cibles = trouves + (proteges if forcer else [])
    if not cibles:
        return f"Aucune application « {nom} » n'est ouverte (rien à fermer)."
    if forcer and proteges:
        parler(f"Je ferme {nom}, je vais donc m'arrêter aussi. À bientôt.")
    arretes = _tuer(cibles)
    if arretes < len(cibles):
        return f"ÉCHEC partiel : {arretes} processus sur {len(cibles)} de « {nom} » arrêtés."
    return f"« {nom} » est fermé (vérifié : {arretes} processus arrêtés)."


def supprimer(chemin: str) -> str:
    """Supprime définitivement un fichier ou un dossier (avec tout son contenu)."""
    cible = _chemin(chemin)
    interdits = {"/", os.path.expanduser("~"), "/home", "/etc", "/usr", "/bin", "/boot", "/var", "/lib"}
    if cible in interdits:
        return f"REFUSÉ : je ne supprime pas {cible} en entier, c'est un dossier vital."
    if not os.path.lexists(cible):
        return f"Rien à supprimer : {cible} n'existe pas."
    if CONFIRMER_ACTIONS_SENSIBLES and not _confirmer(f"Je supprime {os.path.basename(cible)} ?"):
        return "REFUSÉ : l'utilisateur n'a pas confirmé."
    try:
        if os.path.isdir(cible) and not os.path.islink(cible):
            shutil.rmtree(cible)
        else:
            os.remove(cible)
    except Exception as e:
        return f"Erreur : {e}"
    if os.path.lexists(cible):
        return f"ÉCHEC : {cible} existe toujours."
    if _CWD["d"].startswith(cible):
        _CWD["d"] = os.path.dirname(cible)
    return f"Supprimé (vérifié) : {cible}"


def quitter_jarvis() -> str:
    """Arrête complètement Jarvis (le programme s'éteint après avoir répondu). À utiliser quand
    l'utilisateur dit « quitte », « éteins-toi », « arrête-toi », « au revoir Jarvis »..."""
    _ETAT["quitter"] = True
    return "Jarvis va s'arrêter juste après cette réponse : dis au revoir en une phrase courte."


# ============================ NAVIGATEUR WEB ================================
# Jarvis pilote un vrai navigateur Chromium VISIBLE (Playwright) : tu le vois ouvrir les pages,
# défiler, cliquer et taper. Le profil est gardé : tes connexions aux sites restent ouvertes.
#   Installation (une fois) : ./venv/bin/pip install playwright beautifulsoup4
#                             ./venv/bin/playwright install chromium
PROFIL_WEB = os.path.expanduser("~/.config/jarvis-navigateur")
DOSSIER_DONNEES = os.path.expanduser("~/Jarvis_donnees")      # où vont les données récoltées
_WEB = {"pw": None, "ctx": None, "page": None, "exec": None}

_JS_TEXTE_VISIBLE = r"""() => {
  const H = innerHeight, blocs = [], cache = new Map();
  const bloc = el => {            // bloc parent (paragraphe, titre, cellule...) : on y recolle les morceaux
    let e = el;
    while (e && e !== document.body) {
      if (!cache.has(e)) cache.set(e, !getComputedStyle(e).display.startsWith('inline'));
      if (cache.get(e)) return e;
      e = e.parentElement;
    }
    return document.body;
  };
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n, dernier = null;
  while ((n = w.nextNode())) {
    const t = n.textContent.replace(/\s+/g, ' ').trim();
    const el = n.parentElement;
    if (!t || !el || ['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(el.tagName)) continue;
    const r = el.getBoundingClientRect();
    if (r.bottom < 0 || r.top > H || r.width === 0 || r.height === 0) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none') continue;
    const b = bloc(el);
    if (b === dernier && blocs.length) blocs[blocs.length - 1] += ' ' + t;
    else blocs.push(t);
    dernier = b;
  }
  return blocs.join('\n').replace(/ ([,.;:!?)])/g, '$1');
}"""
_JS_POSITION = "() => [Math.round(scrollY), Math.max(document.body.scrollHeight, document.documentElement.scrollHeight), innerHeight]"


def _web(fn, delai=120):
    """Playwright n'aime pas changer de thread : tout passe par un seul thread dédié."""
    import concurrent.futures
    if _WEB["exec"] is None:
        _WEB["exec"] = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="web")
    try:
        return _WEB["exec"].submit(fn).result(timeout=delai)
    except ModuleNotFoundError:
        return ("Le module navigateur n'est pas installé : ./venv/bin/pip install playwright beautifulsoup4 "
                "puis ./venv/bin/playwright install chromium")
    except concurrent.futures.TimeoutError:
        return "Le navigateur met trop de temps à répondre."
    except Exception as e:
        return f"Erreur navigateur : {type(e).__name__}: {str(e)[:300]}"


def _page():
    """La page active du navigateur de Jarvis (le lance si besoin, ou le relance si tu l'as fermé)."""
    ctx = _WEB["ctx"]
    if ctx is not None:
        try:
            pages = [p for p in ctx.pages if not p.is_closed()]
            if pages:
                if _WEB["page"] not in pages:
                    _WEB["page"] = pages[-1]
                return _WEB["page"]
        except Exception:
            pass
        try:
            ctx.close()
        except Exception:
            pass
        _WEB["ctx"] = None
    if _WEB["pw"] is None:
        from playwright.sync_api import sync_playwright
        _WEB["pw"] = sync_playwright().start()
    ctx = _WEB["pw"].chromium.launch_persistent_context(
        PROFIL_WEB, headless=False, no_viewport=True, locale="fr-FR",
        args=["--start-maximized", "--disable-blink-features=AutomationControlled"])
    ctx.on("page", lambda p: _WEB.__setitem__("page", p))      # nouvel onglet ouvert par un lien
    _WEB["ctx"] = ctx
    _WEB["page"] = ctx.pages[0] if ctx.pages else ctx.new_page()
    return _WEB["page"]


def _attendre_page(page):
    try:
        page.wait_for_load_state("domcontentloaded", timeout=15000)
        page.wait_for_load_state("networkidle", timeout=4000)
    except Exception:
        pass


def _texte_visible(page, limite=5000):
    try:
        t = page.evaluate(_JS_TEXTE_VISIBLE)
    except Exception:
        t = ""
    return t[:limite] + ("\n[...]" if len(t) > limite else "")


def _etat_page(page, avec_texte=True):
    try:
        y, h, vh = page.evaluate(_JS_POSITION)
        pos = f"position {min(100, round(100 * (y + vh) / max(h, 1)))} % de la page"
    except Exception:
        pos = ""
    s = f"Page : {page.title()[:120]} | {page.url[:200]} | {pos}"
    return s + ("\nTexte visible à l'écran :\n" + _texte_visible(page) if avec_texte else "")


def _surligner(loc):
    """Entoure en rouge l'élément sur lequel Jarvis va agir : tu vois ce qu'il fait."""
    try:
        loc.scroll_into_view_if_needed(timeout=3000)
        loc.evaluate("e => { e.style.outline = '3px solid #e53935'; e.style.outlineOffset = '2px'; "
                     "setTimeout(() => { e.style.outline = ''; }, 1500); }")
        time.sleep(0.4)
    except Exception:
        pass


def web_ouvrir(adresse: str) -> str:
    """Ouvre une page web dans le navigateur VISIBLE de Jarvis (adresse, ou des mots à chercher sur Google).
    Renvoie le titre et le texte visible. Ensuite : web_defiler, web_lire, web_cliquer, web_taper..."""
    def f():
        page = _page()
        a = adresse.strip()
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", a):
            url = a
        elif re.match(r"^[\w-]+(\.[\w-]+)+(/\S*)?$", a):
            url = "https://" + a
        else:
            import urllib.parse
            url = "https://www.google.com/search?hl=fr&q=" + urllib.parse.quote(a)
        page.bring_to_front()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        _attendre_page(page)
        return _etat_page(page)
    return _web(f)


def web_defiler(direction: str = "bas", ecrans: float = 1.0) -> str:
    """Fait défiler la page sous les yeux de l'utilisateur et renvoie le texte maintenant visible
    (pour lire une page petit à petit). direction : "bas", "haut", "debut" ou "fin"."""
    def f():
        page = _page()
        d = direction.lower()
        if "deb" in d or "déb" in d or "top" in d:
            page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        elif "fin" in d or "bottom" in d:
            page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")
        else:
            signe = -1 if "haut" in d or "up" in d or "monte" in d else 1
            page.evaluate(f"window.scrollBy({{top: {signe} * innerHeight * 0.85 * {float(ecrans)}, behavior: 'smooth'}})")
        time.sleep(0.9)
        return _etat_page(page)
    return _web(f)


def web_lire(tout: bool = False) -> str:
    """Lit la page web ouverte. tout=False : seulement ce qui est à l'écran ; tout=True : la page entière
    (jusqu'à 15000 caractères), pour résumer ou chercher une information."""
    def f():
        page = _page()
        if not tout:
            return _etat_page(page)
        texte = page.inner_text("body", timeout=10000)
        texte = re.sub(r"\n\s*\n+", "\n", texte)
        return _etat_page(page, False) + "\n" + texte[:15000] + ("\n[... page tronquée]" if len(texte) > 15000 else "")
    return _web(f)


def _trouver(page, texte, champ=False):
    """Cherche un élément par son texte, son étiquette, son placeholder... (le 1er visible)."""
    if champ:
        essais = [lambda: page.get_by_label(texte), lambda: page.get_by_placeholder(texte),
                  lambda: page.get_by_role("searchbox", name=texte), lambda: page.get_by_role("textbox", name=texte),
                  lambda: page.get_by_role("combobox", name=texte),
                  lambda: page.locator(f'input[name*="{texte}" i], textarea[name*="{texte}" i], '
                                       f'input[id*="{texte}" i], textarea[id*="{texte}" i]')]
    else:
        essais = [lambda: page.get_by_role("button", name=texte), lambda: page.get_by_role("link", name=texte),
                  lambda: page.get_by_role("tab", name=texte), lambda: page.get_by_role("menuitem", name=texte),
                  lambda: page.get_by_label(texte), lambda: page.get_by_text(texte, exact=True),
                  lambda: page.get_by_text(texte), lambda: page.get_by_title(texte),
                  lambda: page.get_by_alt_text(texte)]
    for essai in essais:
        try:
            loc = essai()
            for i in range(min(loc.count(), 10)):
                el = loc.nth(i)
                if el.is_visible():
                    return el
        except Exception:
            continue
    return None


def web_cliquer(texte: str) -> str:
    """Clique sur un lien, un bouton, un onglet... de la page web, désigné par le texte qu'on voit dessus
    (ou son étiquette). L'élément est entouré en rouge avant le clic."""
    def f():
        page = _page()
        loc = _trouver(page, texte)
        if loc is None:
            return f"Je ne trouve rien de cliquable avec le texte « {texte} » à l'écran. Défile ou lis la page."
        _surligner(loc)
        loc.click(timeout=8000)
        time.sleep(0.5)
        page = _page()
        _attendre_page(page)
        return "Cliqué. " + _etat_page(page)
    return _web(f)


def web_taper(texte: str, champ: str = "", valider: bool = False) -> str:
    """Tape du texte dans un champ de la page web (recherche, formulaire, connexion, message...).
    champ = étiquette ou placeholder du champ (« Rechercher », « E-mail », « Nom »...) ; vide = le champ
    actif ou le premier champ de saisie visible. valider=True appuie sur Entrée après."""
    def f():
        page = _page()
        loc = _trouver(page, champ, champ=True) if champ else None
        if loc is None and not champ:
            actif = page.evaluate("() => { const e = document.activeElement; return !!e && "
                                  "(['INPUT','TEXTAREA'].includes(e.tagName) || e.isContentEditable); }")
            if not actif:
                cand = page.locator('input[type="search"], input[type="text"], input[type="email"], '
                                    'input:not([type]), textarea, [contenteditable="true"]')
                for i in range(min(cand.count(), 20)):
                    if cand.nth(i).is_visible():
                        loc = cand.nth(i)
                        break
        if champ and loc is None:
            bouton = _trouver(page, champ)               # ex : loupe « Rechercher » qui ouvre le champ
            if bouton is not None:
                _surligner(bouton)
                bouton.click(timeout=5000)
                time.sleep(0.6)
                loc = _trouver(page, champ, champ=True)
                if loc is None and page.evaluate("() => ['INPUT','TEXTAREA'].includes(document.activeElement?.tagName)"):
                    loc = page.locator(":focus")
        if champ and loc is None:
            return f"Je ne trouve pas de champ « {champ} » à l'écran."
        if loc is not None:
            _surligner(loc)
            loc.click(timeout=8000)
            loc.fill("")
        page.keyboard.type(texte, delay=35)          # tapé lettre par lettre : l'utilisateur le voit
        if valider:
            page.keyboard.press("Enter")
            time.sleep(0.8)
            page = _page()
            _attendre_page(page)
        return "Texte tapé. " + _etat_page(page)
    return _web(f)


def web_touche(touche: str) -> str:
    """Appuie sur une touche dans le navigateur : Enter, Escape, Tab, PageDown, ArrowDown, Control+a..."""
    def f():
        page = _page()
        page.keyboard.press(touche)
        time.sleep(0.5)
        return _etat_page(page)
    return _web(f)


def web_retour() -> str:
    """Revient à la page web précédente (bouton Précédent)."""
    def f():
        page = _page()
        page.go_back(wait_until="domcontentloaded", timeout=20000)
        _attendre_page(page)
        return _etat_page(page)
    return _web(f)


def web_liens(filtre: str = "") -> str:
    """Liste les liens de la page web ouverte (texte et adresse). filtre = mot à chercher dans les liens."""
    def f():
        page = _page()
        liens = page.evaluate("() => [...document.querySelectorAll('a[href]')].map(a => "
                              "[a.innerText.replace(/\\s+/g,' ').trim(), a.href]).filter(x => x[0] || x[1])")
        filt = filtre.lower()
        lignes, vus = [], set()
        for t, h in liens:
            if (filt and filt not in t.lower() and filt not in h.lower()) or h in vus:
                continue
            vus.add(h)
            lignes.append(f"- {t[:80] or '(sans texte)'} -> {h[:150]}")
        return f"{len(lignes)} liens :\n" + "\n".join(lignes[:120])
    return _web(f)


def web_fermer() -> str:
    """Ferme le navigateur de Jarvis."""
    def f():
        if _WEB["ctx"] is None:
            return "Le navigateur de Jarvis n'était pas ouvert."
        try:
            _WEB["ctx"].close()
        finally:
            _WEB["ctx"], _WEB["page"] = None, None
        return "Navigateur fermé."
    return _web(f)


# ------------------------ Lecture à voix haute en défilant ------------------
_JS_BLOCS = r"""(depuisDebut) => {
  const racine = document.querySelector('article, main, [role=main], #content, #mw-content-text') || document.body;
  const exclus = 'nav, footer, aside, header, [role=navigation], [role=banner], [role=contentinfo], .navbox, .infobox, .reference, .mw-editsection';
  const els = [...racine.querySelectorAll('h1, h2, h3, h4, p, li, blockquote, dd, figcaption')];
  const out = [];
  els.forEach(e => {
    if (e.closest(exclus)) return;
    if (e.tagName === 'LI' && e.querySelector('p, li')) return;
    const t = e.innerText.replace(/\s+/g, ' ').trim();
    const titre = /^H\d$/.test(e.tagName);
    if (!t || (!titre && t.length < 40)) return;
    const r = e.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    if (!depuisDebut && r.bottom < 0) return;
    e.setAttribute('data-jarvis-bloc', out.length);
    out.push(t);
  });
  return out;
}"""


def web_lire_a_voix_haute(depuis_debut: bool = False, blocs_max: int = 30) -> str:
    """Lit À VOIX HAUTE le contenu de la page web ouverte en la faisant défiler automatiquement, petit à
    petit, sous les yeux de l'utilisateur (le passage lu est surligné). Commence à l'endroit affiché
    (ou au début si depuis_debut=True). L'utilisateur peut l'interrompre en parlant.
    Après cet outil, réponds juste une phrase très courte (le texte a déjà été lu)."""
    blocs = _web(lambda: _page().evaluate(_JS_BLOCS, bool(depuis_debut)))
    if isinstance(blocs, str):
        return blocs
    if not blocs:
        return "Je ne trouve pas de texte à lire sur cette page."
    # on regroupe les petits blocs (environ 450 caractères) pour une lecture fluide
    groupes, courant = [], []
    for i, t in enumerate(blocs[:max(1, int(blocs_max))]):
        courant.append((i, t))
        if sum(len(x) for _, x in courant) > 450 or re.match(r"^\S", t) and len(t) < 80 and len(courant) > 1:
            groupes.append(courant)
            courant = []
    if courant:
        groupes.append(courant)
    lus = 0
    for g in groupes:
        premier = g[0][0]

        def montrer(i=premier, n=len(g)):
            page = _page()
            page.evaluate("([i, n]) => { document.querySelectorAll('[data-jarvis-lu]').forEach(e => {"
                          "e.style.background = ''; e.removeAttribute('data-jarvis-lu'); });"
                          "for (let k = i; k < i + n; k++) { const e = document.querySelector(`[data-jarvis-bloc=\"${k}\"]`);"
                          "if (!e) continue; if (k === i) e.scrollIntoView({behavior: 'smooth', block: 'center'});"
                          "e.style.background = 'rgba(255, 235, 59, .35)'; e.setAttribute('data-jarvis-lu', 1); } }",
                          [i, n])
        _web(montrer)
        parler(" ".join(t for _, t in g))
        lus += len(g)
        if _INTERROMPU["oui"]:
            return (f"L'utilisateur m'a interrompu pendant la lecture (après {lus} passages). "
                    "N'ajoute rien : il va te dire ce qu'il veut.")
    reste = len(blocs) - lus
    return (f"Lecture faite ({lus} passages lus à voix haute). " +
            (f"Il reste {reste} passages : propose de continuer." if reste > 0 else "Fin de la page atteinte."))


# ----------------------- Demander une info à l'utilisateur ------------------
def demander_utilisateur(question: str) -> str:
    """Pose une question à voix haute à l'utilisateur et renvoie sa réponse, SANS arrêter la tâche en cours.
    Sert pendant une démarche : informations pour remplir un formulaire, code de vérification reçu par
    SMS ou e-mail, choix à faire, validation d'un message avant de l'envoyer, captcha à résoudre..."""
    parler(question)
    r, source = _MIC["r"], _MIC["source"]
    if r is None:
        try:
            return input(question + " > ").strip() or "(pas de réponse)"
        except EOFError:
            return "(pas de réponse)"
    for _ in range(2):
        rep = _ecouter_suite(r, source, timeout=20)
        if rep:
            print(f"   (réponse : {rep})")
            return rep
        parler("Je n'ai pas entendu, tu peux répéter ?")
    return "(l'utilisateur n'a pas répondu)"


# ------------------------------ Voir l'écran ---------------------------------
def web_regarder(question: str = "Décris ce qu'on voit à l'écran.") -> str:
    """Prend une capture du navigateur de Jarvis et la fait analyser par l'IA (images, captcha, mise en
    page, graphiques, boutons sans texte...). question = ce qu'il faut chercher ou décrire."""
    img = _web(lambda: _page().screenshot(type="jpeg", quality=70))
    if isinstance(img, str):
        return img
    from google.genai import types
    for modele in MODELES_GEMINI:
        try:
            rep = _client().models.generate_content(
                model=modele, contents=[types.Part.from_bytes(data=img, mime_type="image/jpeg"),
                                        "Réponds en français, de façon précise et concise. " + question])
            return rep.text or "Je ne vois rien de particulier."
        except Exception as e:
            if any(k in str(e) for k in ("429", "404", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                continue
            return f"Analyse de l'écran impossible : {e}"
    return "Analyse de l'écran impossible : quotas Gemini épuisés pour le moment."


# -------------------------------- WhatsApp Web ------------------------------
_WA_RECHERCHE = ('#side div[contenteditable="true"], div[contenteditable="true"][data-tab="3"], '
                 'input[aria-label*="Rechercher"], input[aria-label*="Search"]')
_WA_SAISIE = ('#main footer div[contenteditable="true"], footer div[contenteditable="true"][data-tab="10"], '
              '#main div[contenteditable="true"][role="textbox"]')


def _wa_page():
    """Onglet WhatsApp Web (ouvert si besoin). Renvoie (page, message d'erreur ou None)."""
    page = _page()
    if "web.whatsapp.com" not in page.url:
        page.bring_to_front()
        page.goto("https://web.whatsapp.com/", wait_until="domcontentloaded", timeout=30000)
    for _ in range(40):
        if page.locator("#side, #pane-side").count():
            return page, None
        if page.locator("canvas[aria-label], [data-ref]").count() or "Scan" in page.content()[:20000]:
            time.sleep(1)
            if not page.locator("#side, #pane-side").count():
                return page, ("WhatsApp Web demande de scanner le QR code : sur le téléphone, ouvre WhatsApp > "
                              "Appareils connectés > Connecter un appareil, puis scanne l'écran. C'est à faire "
                              "une seule fois. Demande à l'utilisateur de le faire puis réessaie.")
        time.sleep(0.5)
    return page, "WhatsApp Web ne s'est pas chargé."


def whatsapp_ouvrir_discussion(nom: str) -> str:
    """Ouvre WhatsApp Web dans le navigateur de Jarvis et la discussion avec un contact ou un groupe
    (par son nom). Renvoie les derniers messages de la discussion."""
    def f():
        page, err = _wa_page()
        if err:
            return err
        boite = page.locator(_WA_RECHERCHE).first
        boite.click(timeout=8000)
        page.keyboard.press("Control+a")
        page.keyboard.press("Backspace")
        page.keyboard.type(nom, delay=40)
        time.sleep(1.5)
        res = page.locator(f'#pane-side span[title*="{nom}" i]').first
        if res.count():
            _surligner(res)
            res.click()
        else:
            page.keyboard.press("Enter")
        time.sleep(1.2)
        if not page.locator("#main").count():
            return f"Je ne trouve pas de discussion « {nom} » sur WhatsApp."
        return _wa_messages(page, 10)
    return _web(f)


def _wa_messages(page, n):
    msgs = page.evaluate("""(n) => {
      const rows = [...document.querySelectorAll('#main [data-pre-plain-text]')].slice(-n);
      return rows.map(r => {
        const entete = r.getAttribute('data-pre-plain-text') || '';
        const texte = [...r.querySelectorAll('span.selectable-text, span[dir]')].map(s => s.innerText).join(' ').trim()
                      || r.innerText.trim();
        const moi = !!r.closest('.message-out');
        return (moi ? '[MOI] ' : '') + entete.trim() + ' ' + texte;
      });
    }""", n)
    titre = page.evaluate("() => (document.querySelector('#main header span[title], #main header span[dir=auto]') "
                          "|| {}).innerText || ''")
    if not msgs:
        visible = _texte_visible(page, 3000)
        return f"Discussion : {titre}\n(je n'arrive pas à isoler les messages, voici le texte visible)\n{visible}"
    return f"Discussion : {titre}\nDerniers messages (du plus ancien au plus récent) :\n" + "\n".join(msgs)


def whatsapp_lire_messages(nombre: int = 10) -> str:
    """Lit les derniers messages de la discussion WhatsApp ouverte ([MOI] = messages de l'utilisateur)."""
    def f():
        page, err = _wa_page()
        if err:
            return err
        if not page.locator("#main").count():
            return "Aucune discussion n'est ouverte : utilise whatsapp_ouvrir_discussion."
        return _wa_messages(page, max(1, min(int(nombre), 50)))
    return _web(f)


def whatsapp_discussions_non_lues() -> str:
    """Liste les discussions WhatsApp qui ont des messages non lus (avec le dernier message)."""
    def f():
        page, err = _wa_page()
        if err:
            return err
        lignes = page.evaluate("""() => [...document.querySelectorAll('#pane-side [role="listitem"], #pane-side [role="row"]')]
          .map(r => r.innerText.replace(/\\n+/g, ' | ').trim())
          .filter(t => /\\|\\s*\\d+\\s*$/.test(t) || /non lu|unread/i.test(t)).slice(0, 20)""")
        return ("Discussions non lues :\n" + "\n".join(lignes)) if lignes else "Aucun message non lu visible."
    return _web(f)


def whatsapp_envoyer(message: str) -> str:
    """Écrit et ENVOIE un message (émojis acceptés 😀👍🔥) dans la discussion WhatsApp ouverte.
    Avant d'envoyer, fais valider le texte par l'utilisateur avec demander_utilisateur, sauf s'il a
    dicté exactement le message et dit de l'envoyer."""
    def f():
        page, err = _wa_page()
        if err:
            return err
        saisie = page.locator(_WA_SAISIE).first
        if not saisie.count():
            return "Aucune discussion n'est ouverte : utilise whatsapp_ouvrir_discussion."
        _surligner(saisie)
        saisie.click(timeout=8000)
        lignes = message.split("\n")
        for i, l in enumerate(lignes):
            page.keyboard.insert_text(l)              # insert_text gère les émojis
            if i < len(lignes) - 1:
                page.keyboard.press("Shift+Enter")
        time.sleep(0.4)
        page.keyboard.press("Enter")
        time.sleep(1.0)
        dernier = _wa_messages(page, 1)
        return "Message envoyé (vérifié dans la discussion) : " + dernier.split("\n")[-1]
    return _web(f)


def whatsapp_envoyer_sticker(numero: int = 1, recherche: str = "") -> str:
    """Envoie un sticker dans la discussion WhatsApp ouverte : numero = position dans les stickers
    (1 = le premier / plus récent), recherche = mot pour chercher un sticker (ex : « merci », « rire »)."""
    def f():
        page, err = _wa_page()
        if err:
            return err
        if not page.locator("#main footer").count():
            return "Aucune discussion n'est ouverte."
        for sel in ('#main footer [aria-label*="moji" i]', '#main footer [data-icon*="emoji"]',
                    '#main footer [aria-label*="xpression" i]', '#main footer [data-icon*="expressions"]'):
            b = page.locator(sel).first
            if b.count() and b.is_visible():
                b.click()
                break
        else:
            return "Je ne trouve pas le bouton émojis/stickers de WhatsApp."
        time.sleep(0.8)
        onglet = _trouver(page, "Stickers") or _trouver(page, "Autocollants")
        if onglet is not None:
            onglet.click()
            time.sleep(1.2)
        if recherche:
            champ = page.locator('[aria-label*="echercher" i][contenteditable="true"], input[placeholder*="echercher" i], '
                                 'input[placeholder*="earch" i]').first
            if champ.count():
                champ.click()
                page.keyboard.type(recherche, delay=40)
                time.sleep(1.5)
        stickers = page.locator('img[src*="sticker" i], [data-testid*="sticker" i] img, img[alt*="sticker" i], '
                                'div[role="button"] img[draggable="false"]')
        n = stickers.count()
        if n == 0:
            page.keyboard.press("Escape")
            return "Je ne trouve aucun sticker (la liste est peut-être vide sur ce compte)."
        cible = stickers.nth(max(0, min(int(numero) - 1, n - 1)))
        _surligner(cible)
        cible.click()
        time.sleep(1.0)
        page.keyboard.press("Escape")
        return f"Sticker n°{numero} envoyé ({n} stickers disponibles)."
    return _web(f)


# ----------------------------- Récolte de données ---------------------------
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}")
_RE_TEL = re.compile(r"(?:\+|00)?\d[\d .\-()]{7,18}\d")
_EXT_FICHIERS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".zip", ".rar", ".mp4", ".mp3",
                 ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".exe", ".dmg", ".apk", ".css", ".js")


def _analyser_html(html, url):
    """Extrait d'une page : titre, texte, titres, liens, e-mails, téléphones, images, tableaux."""
    import urllib.parse
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "svg", "iframe"]):
        t.decompose()
    texte = re.sub(r"\n\s*\n+", "\n", soup.get_text("\n", strip=True))
    liens = []
    for a in soup.find_all("a", href=True):
        h = urllib.parse.urljoin(url, a["href"]).split("#")[0]
        liens.append((a.get_text(" ", strip=True)[:120], h))
    emails = set(_RE_EMAIL.findall(texte))
    emails |= {h[7:].split("?")[0] for _, h in liens if h.lower().startswith("mailto:")}
    tels = {h[4:] for _, h in liens if h.lower().startswith("tel:")}
    for m in _RE_TEL.findall(texte):
        m = m.strip()
        chiffres = re.sub(r"\D", "", m)
        groupes = [g for g in re.split(r"[ .\-()]+", m) if g]
        if not 8 <= len(chiffres) <= 15 or re.search(r"\b1[0-9]{3}\s*[-–]\s*[12][0-9]{3}\b", m):
            continue                       # trop court/long, ou une période « 1645-1689 »
        if m.startswith(("+", "00")) or (len(groupes) >= 3 and all(len(g) <= 4 for g in groupes)):
            tels.add(m)
    tableaux = []
    for tab in soup.find_all("table"):
        lignes = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in tab.find_all("tr")]
        lignes = [l for l in lignes if any(l)]
        if len(lignes) >= 2:
            tableaux.append(lignes)
    return {
        "url": url,
        "titre": (soup.title.get_text(strip=True) if soup.title else "")[:300],
        "description": ((soup.find("meta", attrs={"name": "description"}) or {}).get("content") or "")[:500],
        "titres": [h.get_text(" ", strip=True)[:200] for h in soup.find_all(["h1", "h2", "h3"])][:60],
        "texte": texte,
        "emails": sorted(emails), "telephones": sorted(tels),
        "images": [urllib.parse.urljoin(url, i["src"]) for i in soup.find_all("img", src=True)][:100],
        "liens": liens, "tableaux": tableaux,
    }


def _rendre_avec_navigateur(url):
    """Pour les sites faits en JavaScript : on charge la page dans un onglet du navigateur de Jarvis."""
    def f():
        _page()
        p = _WEB["ctx"].new_page()
        try:
            p.goto(url, wait_until="domcontentloaded", timeout=30000)
            _attendre_page(p)
            return p.content()
        finally:
            p.close()
    r = _web(f, 60)
    return r if isinstance(r, str) and r.lstrip().startswith("<") else None


def recolter_site(adresse: str, pages_max: int = 30, avec_navigateur: bool = False) -> str:
    """Récolte automatiquement les données d'un site web entier (ou d'une seule page avec pages_max=1) :
    texte de chaque page, titres, e-mails, téléphones, tableaux, liens, images. Visite les pages du
    même site (en respectant robots.txt) et enregistre tout dans ~/Jarvis_donnees/<site>/ :
    texte.md (tout le texte), pages.json, contacts.csv, liens.csv et les tableaux en CSV.
    avec_navigateur=True pour les sites qui s'affichent en JavaScript (plus lent)."""
    import csv
    import urllib.parse
    import urllib.robotparser
    import requests
    debut = adresse.strip()
    if not re.match(r"^https?://", debut):
        debut = "https://" + debut
    dom = urllib.parse.urlparse(debut).netloc
    base = f"{urllib.parse.urlparse(debut).scheme}://{dom}"
    robots = urllib.robotparser.RobotFileParser(base + "/robots.txt")
    try:
        robots.read()
    except Exception:
        robots = None
    agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"
    session = requests.Session()
    session.headers.update({"User-Agent": agent, "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.6"})
    a_voir, vus, pages, ignorees = [debut], set(), [], 0
    pages_max = max(1, min(int(pages_max), 300))
    t0 = time.time()
    while a_voir and len(pages) < pages_max and time.time() - t0 < 600:
        url = a_voir.pop(0)
        if url in vus:
            continue
        vus.add(url)
        if robots and not robots.can_fetch(agent, url):
            ignorees += 1
            continue
        try:
            r = session.get(url, timeout=15)
            if "html" not in r.headers.get("content-type", ""):
                continue
            html = r.text
        except Exception as e:
            print(f"   [récolte : {url[:80]} inaccessible : {type(e).__name__}]")
            if not avec_navigateur:
                continue
            html = ""
        info = _analyser_html(html, url) if html else None
        if avec_navigateur or info is None or len(info["texte"]) < 300:      # page vide : sûrement du JavaScript
            rendu = _rendre_avec_navigateur(url)
            if rendu:
                info = _analyser_html(rendu, url)
        if info is None:
            continue
        pages.append(info)
        print(f"   [récolte {len(pages)}/{pages_max} : {url[:90]}]")
        for _, h in info["liens"]:
            p = urllib.parse.urlparse(h)
            if (p.scheme in ("http", "https") and p.netloc == dom and h not in vus and h not in a_voir
                    and not p.path.lower().endswith(_EXT_FICHIERS)):
                a_voir.append(h)
        time.sleep(0.4)                     # politesse : on ne surcharge pas le site
    if not pages:
        return f"Je n'ai pu récolter aucune page de {dom} (site inaccessible, bloqué ou interdit par robots.txt)."
    dossier = os.path.join(DOSSIER_DONNEES, re.sub(r"[^\w.-]", "_", dom) + "_" +
                           datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(dossier, exist_ok=True)
    with open(os.path.join(dossier, "pages.json"), "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=1)
    with open(os.path.join(dossier, "texte.md"), "w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"# {p['titre'] or p['url']}\n{p['url']}\n\n{p['texte']}\n\n---\n\n")
    emails = sorted({e for p in pages for e in p["emails"]})
    tels = sorted({t for p in pages for t in p["telephones"]})
    with open(os.path.join(dossier, "contacts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "valeur", "page"])
        for p in pages:
            for e in p["emails"]:
                w.writerow(["email", e, p["url"]])
            for t in p["telephones"]:
                w.writerow(["telephone", t, p["url"]])
    with open(os.path.join(dossier, "liens.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "texte du lien", "adresse"])
        for p in pages:
            for t, h in p["liens"]:
                w.writerow([p["url"], t, h])
    n_tab = 0
    for p in pages:
        for tab in p["tableaux"]:
            n_tab += 1
            with open(os.path.join(dossier, f"tableau_{n_tab}.csv"), "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(tab)
    _CWD["d"] = dossier
    resume = (f"Récolte terminée : {len(pages)} pages de {dom}, {len(emails)} e-mails, {len(tels)} téléphones, "
              f"{n_tab} tableaux. Enregistré dans {dossier} (texte.md, pages.json, contacts.csv, liens.csv"
              f"{', tableau_N.csv' if n_tab else ''}).")
    if ignorees:
        resume += f" {ignorees} pages ignorées car interdites par robots.txt."
    if emails:
        resume += " E-mails : " + ", ".join(emails[:10])
    if tels:
        resume += " | Téléphones : " + ", ".join(tels[:10])
    resume += "\nDébut du texte récolté :\n" + pages[0]["texte"][:2500]
    return resume


def web_extraire_page() -> str:
    """Récolte les données de la page ACTUELLEMENT ouverte dans le navigateur de Jarvis (utile après
    s'être connecté ou avoir fait une recherche) : texte, e-mails, téléphones, tableaux, liens.
    Enregistre tout dans ~/Jarvis_donnees/."""
    def f():
        page = _page()
        return page.content(), page.url
    r = _web(f)
    if isinstance(r, str):
        return r
    html, url = r
    import csv
    import urllib.parse
    info = _analyser_html(html, url)
    dom = urllib.parse.urlparse(url).netloc or "page"
    dossier = os.path.join(DOSSIER_DONNEES, re.sub(r"[^\w.-]", "_", dom) + "_page_" +
                           datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dossier, exist_ok=True)
    with open(os.path.join(dossier, "page.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    with open(os.path.join(dossier, "texte.md"), "w", encoding="utf-8") as f:
        f.write(f"# {info['titre']}\n{url}\n\n{info['texte']}\n")
    for i, tab in enumerate(info["tableaux"], 1):
        with open(os.path.join(dossier, f"tableau_{i}.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(tab)
    _CWD["d"] = dossier
    return (f"Données de la page enregistrées dans {dossier} : {len(info['emails'])} e-mails, "
            f"{len(info['telephones'])} téléphones, {len(info['tableaux'])} tableaux, {len(info['liens'])} liens. "
            f"E-mails : {', '.join(info['emails'][:10])} | Téléphones : {', '.join(info['telephones'][:10])}\n"
            f"Début du texte :\n{info['texte'][:2500]}")


def _resume_appel(nom, kwargs):
    """Résumé court d'un appel d'outil (sans les gros contenus), pour le journal et l'historique."""
    morceaux = []
    for k, v in kwargs.items():
        if k in ("contenu", "nouveau_texte", "ancien_texte"):
            v = f"<{len(str(v))} caractères>"
        v = str(v).replace("\n", " ")
        morceaux.append(v[:80] + ("…" if len(v) > 80 else ""))
    return f"{nom}({', '.join(morceaux)})"


def _avec_journal(fn):
    """Enregistre chaque appel d'outil dans _JOURNAL (et l'affiche), sans changer la signature."""
    import functools

    @functools.wraps(fn)
    def outil(*args, **kwargs):
        noms = list(inspect.signature(fn).parameters)
        kwargs.update(zip(noms, args))
        resume = _resume_appel(fn.__name__, kwargs)
        print(f"   -> {resume}")
        try:
            resultat = fn(**kwargs)
        except Exception as e:
            resultat = f"Erreur : {type(e).__name__}: {e}"
        _JOURNAL.append(resume)
        return resultat
    return outil


_OUTILS_BRUTS = [executer_commande, lancer_en_arriere_plan, ouvrir_terminal, taper_dans_terminal,
                 ouvrir_application, ouvrir_dans_editeur, contexte_editeur, ouvrir_url, chercher_sur_internet,
                 changer_dossier, creer_dossier, lister_dossier, lire_fichier, ecrire_fichier,
                 modifier_fichier, verifier_code, supprimer, fermer_terminal, fermer_application,
                 quitter_jarvis, regler_volume, date_et_heure,
                 web_ouvrir, web_defiler, web_lire, web_cliquer, web_taper, web_touche, web_retour,
                 web_liens, web_extraire_page, recolter_site, web_fermer, web_lire_a_voix_haute,
                 web_regarder, demander_utilisateur, whatsapp_ouvrir_discussion, whatsapp_lire_messages,
                 whatsapp_discussions_non_lues, whatsapp_envoyer, whatsapp_envoyer_sticker]
OUTILS = [_avec_journal(f) for f in _OUTILS_BRUTS]
OUTILS_PAR_NOM = {f.__name__: f for f in OUTILS}


def _demander_sudo():
    """Demande (au clavier, une fois) le mot de passe sudo pour que Jarvis puisse l'utiliser.
    Il reste uniquement en mémoire : jamais écrit sur disque, jamais envoyé à l'IA."""
    global _SUDO_MDP
    if not sys.stdin.isatty():
        return
    mdp = getpass.getpass("Mot de passe sudo pour que Jarvis puisse administrer le PC (Entrée = sans sudo) : ")
    if not mdp:
        return
    subprocess.run(["sudo", "-k"], check=False)
    r = subprocess.run(["sudo", "-S", "-p", "", "-v"], input=mdp + "\n", text=True, capture_output=True)
    if r.returncode == 0:
        _SUDO_MDP = mdp
        print("sudo activé.")
    else:
        print("Mot de passe incorrect : sudo désactivé.")


# =============================== CERVEAU ====================================
def _consigne():
    try:
        systeme = platform.freedesktop_os_release().get("PRETTY_NAME", "Linux")
    except Exception:
        systeme = "Linux"
    bureau = f"{os.environ.get('XDG_CURRENT_DESKTOP', '?')} / {os.environ.get('XDG_SESSION_TYPE', '?')}"
    securite = ("les commandes sensibles déclenchent déjà une confirmation vocale"
                if CONFIRMER_ACTIONS_SENSIBLES else
                "l'utilisateur t'autorise à tout faire sur cet ordinateur, sans limite")
    return (
        f"Tu es Jarvis, l'assistant vocal personnel de {PRENOM}. Tu tournes sur son ordinateur "
        f"({systeme}, bureau {bureau}, dossier personnel {os.path.expanduser('~')}). "
        "Tu es un AGENT : tu contrôles cet ordinateur grâce à tes outils et tu vas au bout des demandes. "
        f"Quand on te demande de faire quelque chose, fais-le immédiatement, sans demander la permission : {securite}.\n"
        "RÈGLES D'ACTION :\n"
        "- Une demande peut contenir plusieurs étapes (« ouvre le terminal et tape ls », « crée un dossier "
        "puis fais-moi un site dedans », « ouvre mon projet et vérifie le code »). Enchaîne TOUS les outils "
        "nécessaires, dans l'ordre, jusqu'à ce que TOUT soit fait. Ne t'arrête jamais après la première étape.\n"
        "- Si un outil échoue, lis l'erreur et essaie une autre méthode avant d'abandonner.\n"
        "- Ne dis jamais que tu as fait quelque chose sans avoir appelé l'outil correspondant.\n"
        "- Terminal : « ouvre le terminal » = ouvrir_terminal ; « tape / lance / exécute X dans le terminal » = "
        "taper_dans_terminal (la commande s'affiche et s'exécute dans la fenêtre visible). Pour ton travail "
        "interne (créer, installer, vérifier) sans que l'utilisateur ait demandé à voir, utilise executer_commande.\n"
        "- Web : pour aller sur un site, lire, faire défiler, cliquer, remplir un formulaire ou chercher "
        "une information, utilise TON navigateur visible : web_ouvrir (adresse ou recherche Google), puis "
        "web_defiler / web_lire / web_cliquer / web_taper / web_touche / web_retour. L'utilisateur voit tout. "
        "« Lis-moi la page » = lis le texte visible et résume-le à voix haute, puis défile si on te demande "
        "la suite. Pour une question d'actualité ou un fait que tu ne connais pas sûrement, va vérifier sur le "
        "web. « Récolte / récupère / scrape les données de tel site » = recolter_site (site entier) ou "
        "web_extraire_page (page affichée), puis dis combien de pages, e-mails, téléphones et où c'est "
        "enregistré. « Lis-moi / lis la page / lis l'article » = web_lire_a_voix_haute (il lit en défilant "
        "tout seul) ; après, une phrase très courte seulement. "
        "Démarches en ligne (créer un compte, s'inscrire, remplir un formulaire, réserver...) : fais-les "
        "toi-même étape par étape dans le navigateur ; chaque fois qu'il te manque une information (nom, date "
        "de naissance, mot de passe souhaité, code de vérification reçu par SMS ou e-mail, choix), appelle "
        "demander_utilisateur puis continue. Un captcha ou une vérification « je ne suis pas un robot » : "
        "demande à l'utilisateur de le faire dans la fenêtre, puis continue. web_regarder sert à voir ce qui "
        "n'est pas du texte (images, captcha, boutons sans texte).\n"
        "- WhatsApp : whatsapp_ouvrir_discussion (lit aussi les derniers messages), whatsapp_lire_messages, "
        "whatsapp_discussions_non_lues, whatsapp_envoyer (émojis bienvenus quand ça colle au ton), "
        "whatsapp_envoyer_sticker. Pour répondre à quelqu'un : lis ses derniers messages, résume-les, propose "
        "une réponse (ou prends celle que l'utilisateur dicte) et fais-la valider avec demander_utilisateur "
        "avant d'envoyer, sauf s'il a dicté le message exact et dit « envoie ». Tu écris au nom de "
        "l'utilisateur : adopte son ton, en français naturel.\n"
        "- Ne dis JAMAIS « c'est fait » sans avoir appelé l'outil dans CETTE demande.\n"
        "- ouvrir_url sert seulement à montrer une page dans le navigateur habituel de l'utilisateur.\n"
        "- Fermer / quitter : « ferme le terminal » = fermer_terminal ; « ferme Firefox / Devin / ... » = "
        "fermer_application ; « supprime ... » = supprimer ; « quitte / éteins-toi / arrête-toi » = quitter_jarvis. "
        "Ne dis JAMAIS qu'une chose est fermée, supprimée ou faite si l'outil a répondu ÉCHEC, ATTENTION ou "
        "« rien à » : répète honnêtement ce que l'outil a dit.\n"
        "- Tu ne peux pas cliquer ni taper au clavier dans les fenêtres graphiques des autres programmes. "
        "Pour agir dans un éditeur de code (Devin, Windsurf, VS Code), modifie directement les fichiers avec "
        "ecrire_fichier / modifier_fichier : l'éditeur se met à jour tout seul.\n"
        "- Code : si l'utilisateur parle de « mon code », « ce fichier », « le projet » sans chemin, appelle "
        "d'abord contexte_editeur pour savoir ce qui est ouvert. « Vérifie le code » = lire_fichier + "
        "verifier_code, puis donne les problèmes trouvés. « Corrige » = corrige avec modifier_fichier puis "
        "revérifie.\n"
        "- Créer un site ou un programme : crée le dossier (creer_dossier ou changer_dossier), écris chaque "
        "fichier en entier avec ecrire_fichier (code complet, propre, moderne, qui fonctionne : pas de "
        "« à compléter »), vérifie-le, puis ouvre le résultat (ouvrir_url sur le index.html, ou "
        "ouvrir_dans_editeur si on te le demande).\n"
        "- Les chemins relatifs partent du dossier de travail actuel (indiqué au début de chaque message).\n"
        "Tu réponds aussi à toutes les questions (culture, conseils, calculs, traductions...). "
        f"Tu l'appelles {PRENOM}, mais seulement de temps en temps, pas à chaque phrase. "
        "S'il te coupe ou te corrige pendant que tu parles, oublie ce que tu disais et fais tout de suite "
        "ce qu'il demande, sans t'excuser longuement.\n"
        "Tes réponses sont lues à voix haute : français naturel et chaleureux, 1 à 2 phrases COURTES "
        "(après une action, dis TOUJOURS clairement que c'est fait et ce que tu as fait, par exemple « C'est fait, "
        "j'ai ouvert le terminal et lancé la commande » ou « Voilà, le dossier et le site sont créés » ; si "
        "quelque chose a échoué, dis-le franchement ; après une vérification de code, "
        "au plus 4 phrases avec l'essentiel), sans markdown, sans listes, sans emojis, sans lire de code, "
        "d'adresses web ni de longs chemins."
    )


class Reponse:
    def __init__(self, text):
        self.text = text


class Indisponible(Exception):
    """Le cerveau n'a pas pu répondre (quota, modèle inexistant, réseau...)."""


def _contenus_gemini(historique):
    from google.genai import types
    contenus = []
    for u, a in historique:
        contenus.append(types.Content(role="user", parts=[types.Part(text=u)]))
        contenus.append(types.Content(role="model", parts=[types.Part(text=a)]))
    return contenus


class Discussion:
    """Un modèle Gemini. Chaque demande repart de l'historique commun (texte + résumé des actions),
    ce qui permet de passer d'un modèle à l'autre sans perdre le fil, et limite les tokens consommés."""

    def __init__(self, modele):
        self.modele = modele
        self.reflexion = True

    def _config(self):
        from google.genai import types
        kw = dict(system_instruction=_consigne(), tools=OUTILS,
                  automatic_function_calling=types.AutomaticFunctionCallingConfig(
                      maximum_remote_calls=MAX_ETAPES))
        if self.reflexion:
            try:
                if "2.5" in self.modele:
                    kw["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=0 if REFLEXION == "minimal" else 1024)
                else:
                    kw["thinking_config"] = types.ThinkingConfig(thinking_level=REFLEXION)
            except Exception:
                pass
        return types.GenerateContentConfig(**kw)

    def send_message(self, texte, historique):
        for _ in range(2):
            chat = _client().chats.create(model=self.modele, config=self._config(),
                                          history=_contenus_gemini(historique))
            try:
                rep = chat.send_message(texte)
            except Exception as e:
                s = str(e).lower()
                if self.reflexion and ("thinking" in s or "budget" in s or "level" in s) and "400" in s:
                    print(f"[{self.modele} refuse le réglage de réflexion : désactivé]")
                    self.reflexion = False
                    continue
                raise
            try:
                texte_rep = rep.text
            except Exception:
                texte_rep = None
            if not texte_rep and _JOURNAL and len(_JOURNAL) >= MAX_ETAPES:
                texte_rep = "J'ai fait beaucoup d'étapes, dis-moi si je dois continuer."
            return Reponse((texte_rep or "C'est fait.").strip())


# ---------------------------- Cerveau de secours : Groq ----------------------
def _schema_outil(fn, court=False):
    """Transforme une fonction Python en description d'outil (format OpenAI/Groq).
    court=True : description réduite à l'essentiel (Groq gratuit est limité en tokens par minute)."""
    types_json = {str: "string", int: "integer", bool: "boolean", float: "number"}
    proprietes, requis = {}, []
    for nom, p in inspect.signature(fn).parameters.items():
        proprietes[nom] = {"type": types_json.get(p.annotation, "string")}
        if p.default is inspect.Parameter.empty:
            requis.append(nom)
    return {"type": "function", "function": {
        "name": fn.__name__,
        "description": _desc_courte(fn.__doc__) if court else " ".join((fn.__doc__ or "").split()),
        "parameters": {"type": "object", "properties": proprietes, "required": requis},
    }}


def _desc_courte(doc):
    texte = " ".join((doc or "").split())
    phrases = re.split(r"(?<=[.!?])\s+", texte)
    court = phrases[0]
    for p in phrases[1:]:
        if len(court) + len(p) > 230:
            break
        court += " " + p
    return court[:300]


# Groq : on n'envoie que les outils utiles à la demande (sinon ~6000 tokens par requête)
_GROUPES_OUTILS = {
    "web": (re.compile(r"web|site|page|internet|google|navigat|cherche|recherche|wiki|compte|inscri|formulaire|"
                       r"défil|descend|scroll|clique|scrap|récolt|donnée|http|www|\.com|\.org|\.fr|lis[- ]moi|"
                       r"article|actualit|youtube|facebook|gmail|connecte|réserv|achète|commande", re.I),
            {"web_ouvrir", "web_defiler", "web_lire", "web_cliquer", "web_taper", "web_touche", "web_retour",
             "web_liens", "web_extraire_page", "recolter_site", "web_fermer", "web_lire_a_voix_haute",
             "web_regarder"}),
    "whatsapp": (re.compile(r"whats|wa\b|message|répond|écri[st]?\s+à|sticker|émoji|emoji|discussion", re.I),
                 {"whatsapp_ouvrir_discussion", "whatsapp_lire_messages", "whatsapp_discussions_non_lues",
                  "whatsapp_envoyer", "whatsapp_envoyer_sticker", "web_regarder", "web_fermer"}),
    "code": (re.compile(r"code|fichier|dossier|projet|site|programme|script|éditeur|devin|vs ?code|windsurf|"
                        r"vérifie|corrige|bug|erreur|crée|écri", re.I),
             {"ouvrir_dans_editeur", "contexte_editeur", "lister_dossier", "lire_fichier", "ecrire_fichier",
              "modifier_fichier", "verifier_code", "creer_dossier", "changer_dossier", "supprimer"}),
}
_OUTILS_TOUJOURS = {"executer_commande", "lancer_en_arriere_plan", "ouvrir_terminal", "taper_dans_terminal",
                    "fermer_terminal", "ouvrir_application", "fermer_application", "ouvrir_url",
                    "demander_utilisateur", "quitter_jarvis", "regler_volume", "date_et_heure",
                    "lire_fichier", "ecrire_fichier", "creer_dossier", "supprimer"}


def _outils_pour(texte, historique):
    """Les outils utiles pour cette demande (en tenant compte des 2 derniers échanges)."""
    contexte = texte + " " + " ".join(u + " " + a for u, a in historique[-2:])
    noms = set(_OUTILS_TOUJOURS)
    for motif, groupe in _GROUPES_OUTILS.values():
        if motif.search(contexte):
            noms |= groupe
    return [_schema_outil(f, court=True) for f in OUTILS if f.__name__ in noms]


class GroqErreur(Exception):
    def __init__(self, message, code=0, attente=None):
        super().__init__(message)
        self.code, self.attente = code, attente


class GroqChat:
    """Un modèle Groq (API compatible OpenAI), avec le même historique commun que Gemini."""
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, modele):
        self.modele = modele
        self.cle = _lire_cle("GROQ_API_KEY")
        self.outils = []

    def disponible(self):
        return bool(self.cle)

    def _requete(self, messages, max_tokens=8000):
        import requests
        for essai in range(3):
            r = requests.post(
                self.URL, timeout=120,
                headers={"Authorization": f"Bearer {self.cle}", "Content-Type": "application/json"},
                json={"model": self.modele, "messages": messages, "tools": self.outils,
                      "tool_choice": "auto", "temperature": 0.4, "max_tokens": max_tokens})
            if r.status_code == 200:
                return r.json()
            if r.status_code == 400 and "tool_use_failed" in r.text and essai < 2:
                continue                      # appel d'outil mal formé par le modèle : on réessaie
            try:
                attente = float(r.headers.get("retry-after", ""))
            except ValueError:
                m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", r.text)
                attente = (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else None
            if r.status_code == 429 and attente is not None and attente <= 3 and essai < 2:
                print(f"   (Groq demande d'attendre {attente:.0f} s : j'attends)")
                time.sleep(attente + 0.5)     # limite par minute : on patiente au lieu d'abandonner
                continue
            if r.status_code == 413 and max_tokens > 2000 and essai < 2:
                max_tokens //= 2
                continue
            raise GroqErreur(f"Groq {r.status_code} : {r.text[:300]}", r.status_code, attente)
        raise GroqErreur("Groq : échec après plusieurs essais", 500)

    def send_message(self, texte, historique):
        messages = [{"role": "system", "content": _consigne()}]
        for u, a in historique:
            messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": texte})
        self.outils = _outils_pour(texte, historique)
        for _ in range(MAX_ETAPES):
            msg = self._requete(messages)["choices"][0]["message"]
            appels = msg.get("tool_calls")
            if not appels:
                return Reponse((msg.get("content") or "").strip() or "C'est fait.")
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": appels})
            for a in appels:
                nom = a["function"]["name"]
                try:
                    args = json.loads(a["function"].get("arguments") or "{}")
                    resultat = str(OUTILS_PAR_NOM[nom](**args))
                except KeyError:
                    resultat = f"Erreur : l'outil {nom} n'existe pas."
                except Exception as e:
                    resultat = f"Erreur : {e}"
                messages.append({"role": "tool", "tool_call_id": a["id"], "content": resultat[:6000]})
        return Reponse("J'ai fait beaucoup d'étapes, dis-moi si je dois continuer.")


def _classer_erreur(e):
    """Renvoie (raison lisible, pause en secondes avant de réessayer ce modèle)."""
    s = str(e)
    b = s.lower().replace(" ", "")
    code = getattr(e, "code", 0) or 0
    if "429" in s or "RESOURCE_EXHAUSTED" in s or code == 429 or "ratelimit" in b:
        if "perday" in b or "tokensperday" in b or "(tpd)" in b or "(rpd)" in b:
            return "quota du jour épuisé", 3 * 3600
        attente = getattr(e, "attente", None)
        m = re.search(r"(?:retry|try again) in ([\d.]+)s|retryDelay['\": ]+(\d+)", s)
        if m:
            attente = float(m.group(1) or m.group(2))
        return "limite par minute", max(20, min(300, attente or 60))
    if "404" in s or "NOT_FOUND" in s or "model_not_found" in s or "decommissioned" in s or code == 404:
        return "modèle introuvable", 24 * 3600
    if any(k in s for k in ("401", "403", "PERMISSION_DENIED", "UNAUTHENTICATED", "API key")) or code in (401, 403):
        return "clé refusée", 24 * 3600
    if code == 413 or "413" in s:
        return "demande trop grosse pour ce modèle", 30
    if "504" in s or "DEADLINE" in s or "Timeout" in type(e).__name__ or "timed out" in s.lower():
        return "trop lent", 300           # un modèle qui traîne : on l'évite 5 min pour répondre vite
    if any(k in s for k in ("500", "502", "503", "UNAVAILABLE", "overloaded", "INTERNAL")) or code >= 500:
        return "serveur saturé", 120
    if "Connection" in type(e).__name__:
        return "réseau", 15
    return f"erreur {type(e).__name__}", 20


_ANNONCE_ACTION = re.compile(
    r"\b(c'est fait|voilà, c'est|j'ai (bien )?(ouvert|fermé|lancé|créé|supprimé|tapé|envoyé|cliqué|écrit|"
    r"modifié|installé|exécuté|réglé|cherché|récolté))", re.IGNORECASE)


class Cerveau:
    """Essaie les modèles dans l'ordre : chaque Gemini (quotas séparés), puis chaque Groq.
    Un modèle en erreur est mis en pause (1 min pour une limite par minute, 3 h pour le quota du jour...)
    et le suivant prend le relais AVEC tout l'historique. Si des actions ont déjà été faites avant la
    panne, le modèle suivant est prévenu pour ne pas les refaire."""

    def __init__(self):
        self.historique = []
        self.cerveaux = [Discussion(m) for m in MODELES_GEMINI if _lire_cle("GEMINI_API_KEY")]
        groq = [GroqChat(m) for m in MODELES_GROQ]
        self.cerveaux += [g for g in groq if g.disponible()]
        self.pause = self._charger_pauses()
        print(f"Cerveaux disponibles (dans l'ordre) : {', '.join(c.modele for c in self.cerveaux)}")
        if not any(isinstance(c, GroqChat) for c in self.cerveaux):
            print("Groq : pas de clé GROQ_API_KEY, pas de cerveau de secours Groq.")
        self.actuel = None

    _FICHIER_PAUSES = os.path.expanduser("~/.cache/jarvis_pauses.json")

    def _charger_pauses(self):
        """Les modèles épuisés restent en pause même après un redémarrage (plus d'essais inutiles)."""
        try:
            with open(self._FICHIER_PAUSES) as f:
                pauses = {k: v for k, v in json.load(f).items() if v > time.time()}
            if pauses:
                print("Modèles encore en pause : " + ", ".join(
                    f"{k} ({int((v - time.time()) / 60)} min)" for k, v in pauses.items()))
            return pauses
        except Exception:
            return {}

    def _mettre_en_pause(self, modele, secondes):
        self.pause[modele] = time.time() + secondes
        try:
            os.makedirs(os.path.dirname(self._FICHIER_PAUSES), exist_ok=True)
            with open(self._FICHIER_PAUSES, "w") as f:
                json.dump(self.pause, f)
        except Exception:
            pass

    def _noter(self, texte, reponse, actions):
        if actions:
            reponse += "\n[Actions faites : " + " ; ".join(actions[-12:]) + "]"
        self.historique.append((texte, reponse))
        self.historique = self.historique[-HISTORIQUE_ECHANGES:]

    def send_message(self, texte, _attendu=False):
        if not _attendu:
            _JOURNAL.clear()
        message = f"[Dossier de travail : {_CWD['d']}]\n{texte}"
        derniere = None
        maintenant = time.time()
        ordre = [c for c in self.cerveaux if self.pause.get(c.modele, 0) <= maintenant]
        if not ordre:                     # tout est en pause : on retente quand même le plus proche
            ordre = sorted(self.cerveaux, key=lambda c: self.pause.get(c.modele, 0))
        essais = [(c, n) for c in ordre for n in (0, 1)]
        for c, n in essais:
            if n == 1 and not (derniere is not None and getattr(derniere, "_modele", None) == c.modele
                               and _classer_erreur(derniere)[0] == "serveur saturé"):
                continue                  # 2e essai du même modèle seulement s'il était juste saturé (503)
            if n == 1:
                time.sleep(2)
            deja = list(_JOURNAL)
            envoi = message
            if deja:
                envoi = (message + "\n\n(Attention : cette demande a déjà été commencée, ces actions sont FAITES : "
                         + " ; ".join(deja) + ". NE LES REFAIS PAS, termine seulement ce qui manque.)")
            try:
                t0 = time.time()
                rep = c.send_message(envoi, self.historique)
                if not _JOURNAL and _ANNONCE_ACTION.search(rep.text or ""):
                    print("   (il annonce une action sans l'avoir faite : je lui demande de la faire)")
                    rep = c.send_message(envoi + "\n\n(ATTENTION : tu n'as appelé AUCUN outil. Si une action est "
                                         "demandée, appelle l'outil maintenant ; sinon réponds sans prétendre "
                                         "avoir agi.)", self.historique)
                duree = time.time() - t0
                if duree > 12 and not _JOURNAL:      # question simple mais modèle surchargé : on l'évite 10 min
                    print(f"   ({c.modele} a été lent ({duree:.0f} s) : je l'évite pendant 10 min)")
                    self._mettre_en_pause(c.modele, 600)
                if self.actuel != c.modele:
                    if self.actuel:
                        print(f"   (cerveau utilisé : {c.modele})")
                    self.actuel = c.modele
                self._noter(texte, rep.text, list(_JOURNAL))
                return rep
            except Exception as e:
                raison, pause = _classer_erreur(e)
                derniere = e
                try:
                    e._modele = c.modele
                except Exception:
                    pass
                if n == 0 and raison == "serveur saturé":
                    print(f"[{c.modele} : saturé, je réessaie dans 2 s]")
                    continue
                self._mettre_en_pause(c.modele, pause)
                print(f"[{c.modele} : {raison} -> pause {int(pause)} s, je passe au suivant] {str(e)[:150]}")
        # tout le monde est en pause pour quelques secondes (limite par minute) : on patiente un peu
        prochain = min((self.pause.get(c.modele, 0) for c in self.cerveaux), default=0) - time.time()
        if not _attendu and 0 < prochain <= 25:
            print(f"   (tous mes cerveaux sont limités : j'attends {prochain:.0f} s)")
            time.sleep(prochain + 0.5)
            return self.send_message(texte, _attendu=True)
        raise derniere or Indisponible("aucun cerveau configuré")


def creer_chat():
    return None, Cerveau()


_ACTIONS_DITES = {
    "ouvrir_terminal": "ouvert le terminal", "taper_dans_terminal": "lancé la commande dans le terminal",
    "executer_commande": "exécuté la commande", "lancer_en_arriere_plan": "lancé le programme",
    "ouvrir_application": "ouvert l'application", "ouvrir_dans_editeur": "ouvert ça dans l'éditeur",
    "ouvrir_url": "ouvert la page", "chercher_sur_internet": "lancé la recherche",
    "creer_dossier": "créé le dossier", "changer_dossier": "changé de dossier",
    "ecrire_fichier": "écrit le fichier", "modifier_fichier": "modifié le fichier",
    "verifier_code": "vérifié le code", "regler_volume": "réglé le volume",
    "supprimer": "supprimé ce que tu m'as demandé", "fermer_terminal": "fermé le terminal",
    "fermer_application": "fermé l'application", "web_ouvrir": "ouvert la page web",
    "web_defiler": "fait défiler la page", "web_cliquer": "cliqué", "web_taper": "tapé le texte",
    "recolter_site": "récolté les données du site", "web_extraire_page": "récolté les données de la page",
    "web_fermer": "fermé le navigateur", "whatsapp_envoyer": "envoyé le message",
    "whatsapp_envoyer_sticker": "envoyé le sticker", "whatsapp_ouvrir_discussion": "ouvert la discussion",
}


def _confirmation(actions):
    """Phrase parlée qui résume ce qui a été fait (quand le cerveau n'a rien dit de précis)."""
    faits = []
    for a in actions:
        dit = _ACTIONS_DITES.get(a.split("(", 1)[0])
        if dit and dit not in faits:
            faits.append(dit)
    if not faits:
        return "C'est fait."
    if len(faits) > 1:
        faits[-1] = "et " + faits[-1]
    return "C'est fait, j'ai " + (", ".join(faits[:-1]) + " " + faits[-1] if len(faits) > 1 else faits[0]) + "."


def demander_ia(client, chat, texte):
    """Interroge le cerveau et traduit les erreurs en phrases claires."""
    try:
        rep = re.sub(r"\[Actions faites[^\]]*\]", "", chat.send_message(texte).text or "").strip()
        if not rep or rep.lower().strip(" .!") in ("c'est fait", "fait", "ok", "voilà"):
            rep = _confirmation(_JOURNAL)          # il dit toujours ce qu'il a fait
        return rep
    except Exception as erreur:
        s = str(erreur)
        print(f"[Erreur IA : {type(erreur).__name__}: {s[:300]}]")
        raison = _classer_erreur(erreur)[0]
        if "quota" in raison or "limite" in raison:
            return "Tous mes cerveaux ont atteint leur limite gratuite. Réessaie dans une minute."
        if "clé" in raison:
            return "Mes clés API sont refusées."
        if "introuvable" in raison:
            return "Les noms de modèles dans mon fichier ne sont pas valides."
        return "Désolé, je n'arrive pas à joindre mon cerveau pour le moment."


def traiter(commande, client, chat):
    """Renvoie False si Jarvis doit s'arrêter."""
    if any(m in commande.lower() for m in ("au revoir", "arrête-toi", "arrête toi", "éteins-toi")):
        parler("À bientôt.")
        return False
    etat = {"proc": None}
    minuteur = None
    if UN_INSTANT_APRES is not None and _ack and os.path.exists(_ack):
        # « Un instant. » seulement si le cerveau met du temps à répondre
        minuteur = threading.Timer(UN_INSTANT_APRES,
                                   lambda: etat.__setitem__("proc", _jouer_en_fond(_ack)))
        minuteur.daemon = True
        minuteur.start()
    t0 = time.time()
    reponse = demander_ia(client, chat, commande)
    if _ETAT["quitter"]:
        parler(reponse)
        return False
    if _INTERROMPU["oui"]:               # coupé pendant une lecture : on écoute au lieu de parler
        print(f"   (réponse non dite, tu m'as coupé : {reponse})")
        return True
    if minuteur:
        minuteur.cancel()
    print(f"   (cerveau + actions : {time.time() - t0:.1f}s)")
    proc = etat["proc"]
    if proc:
        proc.wait()
    parler(reponse, silence=proc is None)
    time.sleep(0.3)      # évite qu'il entende la fin de sa propre voix
    return True


# =============================== PROGRAMME ==================================
def test_cle():
    client, chat = creer_chat()
    t0 = time.time()
    print("Réponse :", demander_ia(client, chat, "Dis bonjour en une phrase."))
    print(f"Temps : {time.time() - t0:.1f}s")


def test_groq():
    if not _lire_cle("GROQ_API_KEY"):
        print("GROQ_API_KEY n'est pas définie.")
        return
    for modele in MODELES_GROQ:
        g = GroqChat(modele)
        t0 = time.time()
        try:
            print(f"{modele} répond :", g.send_message("Dis bonjour en une phrase.", []).text)
            print("   test des outils :", g.send_message("Quelle heure est-il ?", []).text,
                  f"({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"{modele} : erreur {type(e).__name__}: {str(e)[:200]}")


def test_terminal():
    print(ouvrir_terminal())
    print(taper_dans_terminal("echo Bonjour depuis Jarvis && ls | head -5", 10))
    print(taper_dans_terminal("cd /tmp", 5), "| cd conservé ->", _CWD["d"])


def test_voix():
    parler("Bonjour, je suis Jarvis. Si tu m'entends, ma voix fonctionne.")


def test_micro():
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrage (1 seconde)...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Parle maintenant...")
        print("J'ai compris :", ecouter(r, source, timeout=8))


def test_outils():
    print("Date :", date_et_heure())
    print("Commande :", executer_commande("echo salut && uname -r"))
    print("cd conservé :", executer_commande("cd /tmp"), "->", executer_commande("pwd"))
    print("Éditeur :", contexte_editeur())
    for app in ("firefox", "calculatrice", "terminal"):
        print(f"Application « {app} » ->", _trouver_application(app) or "introuvable")


PHRASES_APPRENTISSAGE = (
    f"Bonjour Jarvis, je m'appelle {PRENOM} et ceci est ma voix.",
    "Aujourd'hui, je travaille sur mon ordinateur avec mon assistant personnel.",
    "Ouvre le navigateur, cherche la météo de demain et règle le volume à cinquante pour cent.",
    "Je veux que tu me reconnaisses même quand je parle doucement ou rapidement.",
    "Dis-moi quelle heure il est, puis lis-moi le dernier fichier que j'ai modifié.",
    "Le réseau de l'entreprise fonctionne bien, mais il faut surveiller le serveur de stockage.",
    "Peux-tu lancer la calculatrice, puis ouvrir mon dossier de documents ?",
    "Hier soir, j'ai terminé une partie de mon mémoire sur le cloud computing.",
    "Un, deux, trois, quatre, cinq, six, sept, huit, neuf, dix.",
    "Combien font quarante-sept multiplié par douze, et quelle est la capitale du Bénin ?",
    "Je voudrais installer un nouveau programme, vérifier les mises à jour et redémarrer plus tard.",
    "Cotonou est une ville vivante, avec beaucoup de monde, de bruit et de couleurs.",
    "Mets de la musique, baisse un peu le son, puis dis-moi ce que tu as trouvé.",
    "Quand je te parle, tu dois m'écouter attentivement, même si tu es en train de répondre.",
    "Aujourd'hui, la température est agréable, et j'ai beaucoup de travail à terminer avant ce soir.",
    "Peux-tu chercher sur internet les meilleures solutions pour sauvegarder des fichiers importants ?",
    "Attends, arrête-toi un instant, j'ai quelque chose de plus important à te demander.",
    "Ce projet demande de la patience, de la rigueur et un peu de créativité.",
    "Ouvre mon éditeur de code, crée un nouveau fichier et enregistre-le sur le bureau.",
    f"Merci Jarvis, c'est {PRENOM} qui te parle, et je suis content de ton travail.",
)
MIN_PHRASES_VALIDES = 10          # il faut au moins ça pour enregistrer ta voix


def apprendre_voix():
    """Te fait lire une vingtaine de phrases, une par une, puis enregistre ton empreinte vocale.
    Dès que tu as fini de lire (environ 1 seconde de silence), il passe tout seul à la suivante."""
    import numpy as np
    import speech_recognition as sr
    phrases = PHRASES_APPRENTISSAGE
    total = len(phrases)
    r = sr.Recognizer()
    r.pause_threshold = 1.2           # fin de lecture = 1,2 s de silence -> phrase suivante
    empreintes = []
    with sr.Microphone(sample_rate=16000) as source:
        print("Calibrage du bruit ambiant (2 secondes, ne parle pas)...")
        r.adjust_for_ambient_noise(source, duration=2)
        parler(f"Bonjour {PRENOM}. Je vais apprendre ta voix avec {total} phrases. "
               "Lis chaque phrase affichée, d'une voix normale. Quand tu as fini, je passe à la suivante.")
        try:
            for i, phrase in enumerate(phrases, 1):
                for _ in range(3):
                    print(f"\n[{i}/{total}] Lis :\n    {phrase}")
                    time.sleep(0.5)
                    try:
                        audio = r.listen(source, timeout=20, phrase_time_limit=25)
                    except sr.WaitTimeoutError:
                        print("    (rien entendu, je repose la même phrase)")
                        continue
                    duree = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
                    emb = _empreinte_de(_pcm16k(audio)) if duree >= 2.0 else None
                    if emb is None:
                        print("    (trop court, je repose la même phrase)")
                        continue
                    empreintes.append(emb)
                    print(f"    ✓ bien reçu ({duree:.1f} s)")
                    break
        except KeyboardInterrupt:
            print("\nArrêt demandé : j'enregistre ce que j'ai déjà.")
    if len(empreintes) < MIN_PHRASES_VALIDES:
        print(f"\nSeulement {len(empreintes)} phrases valides (minimum {MIN_PHRASES_VALIDES}). "
              "Relance « python jarvis.py apprendre-voix ».")
        return
    moy = np.mean(empreintes, axis=0)
    moy = moy / np.linalg.norm(moy)
    # on écarte les enregistrements qui ne te ressemblent pas (bruit, quelqu'un d'autre qui parle)
    bons = [e for e in empreintes if float(np.dot(e, moy)) >= 0.65]
    if MIN_PHRASES_VALIDES <= len(bons) < len(empreintes):
        print(f"\n{len(empreintes) - len(bons)} enregistrement(s) douteux écarté(s).")
        empreintes = bons
        moy = np.mean(empreintes, axis=0)
        moy = moy / np.linalg.norm(moy)
    np.save(VOIX_FICHIER, moy)
    scores = [float(np.dot(e, moy)) for e in empreintes]
    print(f"\nEmpreinte enregistrée avec {len(empreintes)} phrases : {VOIX_FICHIER}")
    print(f"Cohérence : min {min(scores):.2f} / moyenne {sum(scores) / len(scores):.2f}")
    parler(f"C'est fait {PRENOM}, je connais ta voix.")


def test_id():
    """Affiche à quel point ce que tu dis ressemble à ta voix (pour régler SEUIL_VOIX)."""
    import speech_recognition as sr
    _charger_empreinte()
    if _EMPREINTE is None:
        return
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("Dis 5 phrases d'au moins 2 secondes. Fais aussi parler quelqu'un d'autre pour comparer.")
        for _ in range(5):
            try:
                audio = r.listen(source, timeout=10, phrase_time_limit=8)
            except sr.WaitTimeoutError:
                continue
            s = _score(audio)
            if s is None:
                print("   trop court")
            else:
                print(f"   ressemblance : {s:.2f}  ->  {'TOI' if s >= _seuil_pour(len(audio.frame_data) / (audio.sample_rate * audio.sample_width)) else 'pas toi'}")


def main():
    import speech_recognition as sr
    _demander_sudo()
    _charger_empreinte()
    client, chat = creer_chat()
    r = sr.Recognizer()
    r.pause_threshold = 1.0
    r.non_speaking_duration = 0.6
    with sr.Microphone(sample_rate=16000) as source:
        print("Calibrage du bruit ambiant (2 secondes, ne parle pas)...")
        r.adjust_for_ambient_noise(source, duration=2)
        r.dynamic_energy_threshold = False
        r.energy_threshold *= FACTEUR_SEUIL
        print(f"(seuil de détection : {r.energy_threshold:.0f})")
        _MIC["r"], _MIC["source"] = r, source
        if UN_INSTANT_APRES is not None:
            preparer_ack()
        def _prechauffer():                      # charge la voix de secours Piper (3 s la 1re fois)
            w = _synth_piper("Prêt.")
            if w and os.path.exists(w):
                os.remove(w)
        threading.Thread(target=_prechauffer, daemon=True).start()
        parler(f"Bonjour {PRENOM}, Jarvis en ligne. Je t'écoute.")
        print(f"(Commence par « {NOMS[0]} ». Ensuite tu peux enchaîner pendant {CONVERSATION_SECONDES} s "
              "sans redire mon nom, et me couper quand je parle. Dis « merci » pour finir la discussion. "
              "Ctrl+C pour quitter)")
        actif_jusqua = 0.0
        while True:
            coupe = _INTERROMPU["oui"]
            if coupe:                                  # tu as coupé Jarvis : on reprend ta phrase
                pre, texte, _INTERROMPU["oui"] = _INTERROMPU["audio"], _INTERROMPU["texte"], False
                print("   (tu m'as coupé : je t'écoute)")
                if texte and _PHRASE_PAS_FINIE.search(texte):   # ta phrase n'est pas finie : on écoute la suite
                    suite = ecouter(r, source, timeout=ATTENTE_SUITE_LONGUE)
                    texte = f"{texte} {suite}" if suite else texte
                elif not texte:
                    texte = ecouter(r, source, prefixe=pre)
            else:
                texte = ecouter(r, source, timeout=5)
            if not texte:
                continue
            print(f"Entendu : {texte}")
            t = texte.lower()
            nom = next((n for n in NOMS if n in t), None)
            if not coupe and not _est_nadhir(_DERNIER["audio"], avec_nom=bool(nom)):
                print(f"   (ignoré : ce n'est pas la voix de {PRENOM})")
                continue
            t = texte.lower()
            en_conversation = coupe or time.time() < actif_jusqua
            if en_conversation and FIN_CONVERSATION.match(t.strip()):
                actif_jusqua = 0.0
                parler("Avec plaisir.")
                continue
            nom = next((n for n in NOMS if n in t), None)
            if nom:
                commande = t.split(nom, 1)[1].strip(" ,.!?")
                if not commande:
                    parler("Oui ?")
                    commande = _ecouter_suite(r, source, timeout=8)
                    if not commande or not _est_nadhir(_DERNIER["audio"]):
                        continue
            elif en_conversation:
                commande = ("(Tu viens d'être coupé en pleine phrase : abandonne ce que tu disais.) " + t) if coupe else t
            elif not MOT_ACTIVATION_OBLIGATOIRE and _DERNIER.get("sur"):
                commande = t                           # c'est clairement ta voix : pas besoin de « Jarvis »
            else:
                print(f"   (ignoré : dis d'abord « {NOMS[0]} »)")
                continue
            if not traiter(commande, client, chat):
                break
            actif_jusqua = time.time() + CONVERSATION_SECONDES


if __name__ == "__main__":
    modes = {"test-cle": test_cle, "test-groq": test_groq, "test-terminal": test_terminal, "test-voix": test_voix,
             "test-micro": test_micro, "test-outils": test_outils,
             "apprendre-voix": apprendre_voix, "test-id": test_id}
    try:
        modes.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
    except KeyboardInterrupt:
        print("\nArrêt de Jarvis.")