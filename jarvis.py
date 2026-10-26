#!/usr/bin/env python3
"""
Jarvis v5 - assistant vocal perso avec contrôle total de l'ordinateur.
Cerveau : Gemini, avec bascule automatique sur Groq si le quota est atteint.
Voix    : Gemini TTS (Kore) -> secours : Piper (Siwis, local) -> edge-tts -> espeak.

    python jarvis.py test-cle     -> teste Gemini (et affiche le temps de réponse)
    python jarvis.py test-groq    -> teste Groq seul
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
MODELE = "gemini-3.8-flash"               # cerveau principal. Si erreur 404 : change ce nom
MODELES_GROQ = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")   # cerveau de secours (le 1er disponible est utilisé)
LANGUE = "fr-FR"

# Contrôle de l'ordinateur
CONFIRMER_ACTIONS_SENSIBLES = False       # False = AUCUNE limite. True = il demande "dis oui" avant rm, sudo, etc.
DELAI_COMMANDE = 120                      # secondes max pour une commande

# Écoute
CONVERSATION_SECONDES = 45                # après une réponse, tu peux continuer SANS redire "Jarvis" (0 = toujours dire Jarvis)
FACTEUR_SEUIL = 1.5                       # >1 = ignore mieux les voix faibles/lointaines. Trop haut : il ne t'entend plus (essaie 1.2)

# Ta voix (reconnaissance du locuteur) et interruption
RECONNAISSANCE_VOIX = True                # True = seule TA voix donne des ordres (d'abord : python jarvis.py apprendre-voix)
APPRENTISSAGE_CONTINU = True              # True = il affine ton empreinte tout seul à chaque phrase dont il est sûr que c'est toi
SEUIL_VOIX_COURT = 0.55                   # ressemblance exigée pour une phrase courte (moins de 2 s) : les notes y sont plus basses
SEUIL_VOIX_MOYEN = 0.62                   # entre 2 et 4 s
SEUIL_VOIX = 0.70                         # ressemblance minimale (0 à 1). Il te rejette ? baisse (0.65). Il accepte les autres ? monte (0.8)
INTERRUPTION = True                       # True = tu peux couper Jarvis en parlant (seulement avec TA voix)
SEUIL_INTERRUPTION = 0.55                 # idem pour couper Jarvis (plus bas : sa voix se mélange à la tienne)
FACTEUR_INTERRUPTION = 0.8                # volume minimal avant de tenter de te reconnaître pendant qu'il parle
DEBUG_INTERRUPTION = True                 # True = affiche la ressemblance pendant qu'il parle (pour régler). Mets False quand c'est bon
UN_INSTANT_APRES = 3.0                    # dit « Un instant. » seulement si la réponse met plus de N secondes (None = jamais)
VOIX_FICHIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voix_nadhir.npy")
# Phrases qui terminent la discussion (ensuite il faut redire "Jarvis")
FIN_CONVERSATION = re.compile(r"^(ok |d'accord |bon )?(merci|c'est bon|c'est tout|ça sera tout|ça suffit)( beaucoup| jarvis)*[ .!]*$")

# Voix Gemini (principale)
MODELES_TTS = ("gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts")
VOIX_GEMINI = "Kore"                      # voix féminines : Kore, Zephyr, Aoede, Leda, Autonoe
STYLE_TTS = "Dis de façon naturelle et chaleureuse : "   # mets "" pour économiser du quota
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
_INTERROMPU = {"oui": False, "audio": None}   # tu as coupé Jarvis : début de ta phrase
_APPRIS = {"n": 0}                              # nombre de phrases qui ont affiné ton empreinte
_DERNIER = {"audio": None}                     # dernier son entendu (pour vérifier la voix)
_silence = None
_ack = None            # petit "Un instant." pré-enregistré, joué pendant que le cerveau réfléchit
_SUDO_MDP = None       # mot de passe sudo, gardé UNIQUEMENT en mémoire


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
        _client_unique = genai.Client(api_key=_cle_api())
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
    _gemini_pause_jusqua = time.time() + (60 if limite else 10)
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
    p.wait()
    _LECTURE["p"] = None


def _couper_lecture():
    p = _LECTURE["p"]
    if p and p.poll() is None:
        p.terminate()


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
    texte = re.sub(r"[*_#`>~|]", "", texte)
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
    _INTERROMPU["oui"], _INTERROMPU["audio"] = False, None
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
            veilleur.join(timeout=2)


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
    if not APPRENTISSAGE_CONTINU or emb is None or duree < 2.5 or s < 0.75:
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
    Si tu as dit « Jarvis », on est un peu moins exigeant."""
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


def _surveiller_interruption(arret):
    """Tourne PENDANT que Jarvis parle. Si c'est TA voix qui s'élève, il se tait et te laisse la parole.
    Sa propre voix (qui revient par le micro) est ignorée car elle ne ressemble pas à la tienne."""
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
            if sum(1 for d in tampon if _rms(d) >= seuil) < taille * 0.3:
                reussites = 0
                continue
            depuis = 0
            audio = sr.AudioData(b"".join(tampon), source.SAMPLE_RATE, source.SAMPLE_WIDTH)
            s = _score(audio)
            if DEBUG_INTERRUPTION and s is not None:
                print(f"   [coupure ? ressemblance {s:.2f} / seuil {SEUIL_INTERRUPTION:.2f}]")
            reussites = reussites + 1 if (s is not None and s >= SEUIL_INTERRUPTION) else 0
            if reussites >= 2 or (s is not None and s >= SEUIL_INTERRUPTION + 0.06):
                _INTERROMPU["oui"], _INTERROMPU["audio"] = True, audio.frame_data
                _couper_lecture()
                return
    except Exception as e:
        print(f"[Écoute pendant la parole arrêtée : {type(e).__name__}: {e}]")


# =============================== OREILLES ===================================
def _suite_phrase(recognizer, source, debut_audio, limite=15):
    """Continue d'enregistrer ta phrase déjà commencée (quand tu as coupé Jarvis), jusqu'au silence."""
    import speech_recognition as sr
    cps = source.SAMPLE_RATE / source.CHUNK
    morceaux, calmes = [debut_audio], 0
    fin = time.time() + limite
    while time.time() < fin:
        d = source.stream.read(source.CHUNK)
        morceaux.append(d)
        calmes = calmes + 1 if _rms(d) < recognizer.energy_threshold else 0
        if calmes >= int(0.9 * cps):
            break
    return sr.AudioData(b"".join(morceaux), source.SAMPLE_RATE, source.SAMPLE_WIDTH)


def ecouter(recognizer, source, timeout=5, limite=15, prefixe=None):
    """Écoute une phrase et renvoie le texte (ou None).
    prefixe = début de phrase déjà enregistré (quand tu coupes Jarvis en pleine parole)."""
    import speech_recognition as sr
    if prefixe is not None:
        audio = _suite_phrase(recognizer, source, prefixe, limite)
    else:
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=limite)
        except sr.WaitTimeoutError:
            return None
    _DERNIER["audio"] = audio
    try:
        return recognizer.recognize_google(audio, language=LANGUE)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"[Reconnaissance vocale indisponible (internet ?) : {e}]")
        return None


def _ecouter_suite(recognizer, source, timeout=8, limite=15):
    """Comme ecouter(), mais reprend ta phrase si tu viens de couper Jarvis."""
    if _INTERROMPU["oui"]:
        pre, _INTERROMPU["oui"] = _INTERROMPU["audio"], False
        return ecouter(recognizer, source, prefixe=pre, limite=limite)
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


def executer_commande(commande: str) -> str:
    """Exécute une commande shell (bash) sur l'ordinateur de l'utilisateur et renvoie sa sortie.
    Sert à tout : fichiers et dossiers, installation de paquets (sudo fonctionne), infos système,
    réseau, captures d'écran, git, scripts, etc. Attend la fin de la commande.
    Pour lancer une application graphique, utilise plutôt ouvrir_application."""
    if not _autorise(commande):
        return "REFUSÉ : l'utilisateur n'a pas confirmé cette commande."
    cmd, kw = commande, {"stdin": subprocess.DEVNULL}
    if _SUDO_MDP and _SUDO_CMD.search(commande):
        cmd = _SUDO_CMD.sub(r"\1sudo -S -p '' ", commande)
        kw = {"input": _SUDO_MDP + "\n"}
    t0 = time.time()
    try:
        p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                           timeout=DELAI_COMMANDE, cwd=os.path.expanduser("~"), **kw)
        print(f"   [commande exécutée en {time.time() - t0:.1f}s : {commande[:70]}]")
        sortie = (p.stdout + p.stderr).strip()
        return f"code de retour {p.returncode}\n{sortie[:3000]}"
    except subprocess.TimeoutExpired:
        return (f"La commande dépasse {DELAI_COMMANDE} secondes et a été arrêtée. "
                "Utilise lancer_en_arriere_plan si elle doit tourner longtemps.")
    except Exception as e:
        return f"Erreur : {e}"


def lancer_en_arriere_plan(commande: str) -> str:
    """Lance une commande shell en arrière-plan sans attendre sa fin (serveur, téléchargement long,
    programme graphique). Renvoie immédiatement."""
    if not _autorise(commande):
        return "REFUSÉ : l'utilisateur n'a pas confirmé cette commande."
    try:
        subprocess.Popen(["bash", "-c", commande], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         cwd=os.path.expanduser("~"))
        return "Commande lancée en arrière-plan."
    except Exception as e:
        return f"Erreur : {e}"


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
    (ex : Firefox, Chrome, calculatrice, Spotify, VLC, Visual Studio Code, terminal, fichiers)."""
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
    """Ouvre une adresse web (site, page, vidéo) dans le navigateur par défaut."""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url
    webbrowser.open(url)
    return f"Adresse ouverte : {url}"


def chercher_sur_internet(requete: str) -> str:
    """Ouvre une recherche Google dans le navigateur pour la requête donnée."""
    import urllib.parse
    webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote(requete))
    return f"Recherche lancée : {requete}"


def lire_fichier(chemin: str) -> str:
    """Lit le contenu d'un fichier texte (jusqu'à 5000 caractères). ~ désigne le dossier personnel."""
    try:
        with open(os.path.expanduser(chemin), encoding="utf-8", errors="replace") as f:
            return f.read(5000)
    except Exception as e:
        return f"Erreur : {e}"


def ecrire_fichier(chemin: str, contenu: str, ajouter: bool = False) -> str:
    """Écrit (ou ajoute à la fin si ajouter=True) du texte dans un fichier. Crée les dossiers manquants."""
    chemin = os.path.expanduser(chemin)
    if CONFIRMER_ACTIONS_SENSIBLES and os.path.exists(chemin) and not ajouter:
        if not _confirmer(f"Le fichier {os.path.basename(chemin)} existe déjà. Je l'écrase ?"):
            return "REFUSÉ : l'utilisateur n'a pas confirmé l'écrasement."
    try:
        dossier = os.path.dirname(chemin)
        if dossier:
            os.makedirs(dossier, exist_ok=True)
        with open(chemin, "a" if ajouter else "w", encoding="utf-8") as f:
            f.write(contenu)
        return f"Fichier écrit : {chemin}"
    except Exception as e:
        return f"Erreur : {e}"


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


OUTILS = [executer_commande, lancer_en_arriere_plan, ouvrir_application, ouvrir_url,
          chercher_sur_internet, lire_fichier, ecrire_fichier, regler_volume, date_et_heure]
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
    securite = ("les commandes sensibles déclenchent déjà une confirmation vocale"
                if CONFIRMER_ACTIONS_SENSIBLES else
                "l'utilisateur t'autorise à tout faire sur cet ordinateur, sans limite")
    return (
        f"Tu es Jarvis, l'assistant vocal personnel de {PRENOM}. Tu tournes sur son ordinateur "
        f"({systeme}, dossier personnel {os.path.expanduser('~')}). "
        "Tu contrôles cet ordinateur grâce à tes outils (commandes shell, applications, navigateur, "
        "fichiers, volume). Quand on te demande de faire quelque chose, fais-le immédiatement et "
        f"directement avec les outils, sans demander la permission : {securite}. "
        "Sois efficace : un seul appel d'outil quand c'est possible, sans vérifications inutiles. "
        "Si une commande échoue, lis l'erreur et essaie une autre méthode. "
        "Ne dis jamais que tu as fait quelque chose sans avoir appelé l'outil. "
        "Tu ne peux pas cliquer ni taper dans les fenêtres d'autres programmes : dis-le honnêtement. "
        "Tu réponds aussi à toutes les questions (culture, conseils, calculs, traductions...). "
        f"Tu l'appelles {PRENOM}, mais seulement de temps en temps, pas à chaque phrase. "
        "S'il te coupe ou te corrige pendant que tu parles, oublie ce que tu disais et fais tout de suite "
        "ce qu'il demande, sans t'excuser longuement. "
        "Tes réponses sont lues à voix haute : français naturel et chaleureux, 1 à 2 phrases COURTES "
        "(après une action, juste « C'est fait » ou une phrase très brève), sans markdown, sans listes, "
        "sans emojis, sans lire d'adresses web ni de longs chemins."
    )


class Reponse:
    def __init__(self, text):
        self.text = text


class Discussion:
    """Conversation avec Gemini, avec réflexion réduite pour répondre vite.
    Si le modèle refuse ce réglage, on le retire automatiquement et on réessaie."""

    def __init__(self):
        self.reduite = True
        self.chat = self._creer()

    def _creer(self):
        from google.genai import types
        kw = dict(system_instruction=_consigne(), tools=OUTILS)
        if self.reduite:
            try:
                if "2.5" in MODELE:
                    kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                else:
                    niveau = getattr(getattr(types, "ThinkingLevel", None), "MINIMAL", "minimal")
                    kw["thinking_config"] = types.ThinkingConfig(thinking_level=niveau)
            except Exception:
                pass
        return _client().chats.create(model=MODELE, config=types.GenerateContentConfig(**kw))

    def send_message(self, texte):
        try:
            return self.chat.send_message(texte)
        except Exception as e:
            s = str(e).lower()
            if self.reduite and ("thinking" in s or "budget" in s or "level" in s):
                print("[Réglage de réflexion rapide refusé par ce modèle : désactivé]")
                self.reduite = False
                self.chat = self._creer()
                return self.chat.send_message(texte)
            raise


# ---------------------------- Cerveau de secours : Groq ----------------------
def _schema_outil(fn):
    """Transforme une fonction Python en description d'outil (format OpenAI/Groq)."""
    types_json = {str: "string", int: "integer", bool: "boolean", float: "number"}
    proprietes, requis = {}, []
    for nom, p in inspect.signature(fn).parameters.items():
        proprietes[nom] = {"type": types_json.get(p.annotation, "string")}
        if p.default is inspect.Parameter.empty:
            requis.append(nom)
    return {"type": "function", "function": {
        "name": fn.__name__,
        "description": " ".join((fn.__doc__ or "").split()),
        "parameters": {"type": "object", "properties": proprietes, "required": requis},
    }}


class GroqErreur(Exception):
    pass


class GroqChat:
    """Même rôle que Discussion, mais avec Groq (API compatible OpenAI, via 'requests')."""
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        self.cle = _lire_cle("GROQ_API_KEY")
        self.messages = [{"role": "system", "content": _consigne()}]
        self.modele_ok = None
        self.outils = [_schema_outil(f) for f in OUTILS]

    def disponible(self):
        return bool(self.cle)

    def noter(self, utilisateur, assistant):
        """Garde en mémoire un échange déjà traité par Gemini, pour que Groq suive la conversation."""
        if self.cle and assistant:
            self.messages += [{"role": "user", "content": utilisateur},
                              {"role": "assistant", "content": assistant}]
            self._tailler()

    def _tailler(self, garder=8):
        """Ne garde que les derniers échanges (le palier gratuit limite les tokens par minute)."""
        idx = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        if len(idx) > garder:
            self.messages = self.messages[:1] + self.messages[idx[-garder]:]

    def _requete(self):
        import requests
        ordre = list(MODELES_GROQ)
        if self.modele_ok in ordre:
            ordre.remove(self.modele_ok)
            ordre.insert(0, self.modele_ok)
        derniere = None
        for modele in ordre:
            for essai in range(2):
                r = requests.post(
                    self.URL, timeout=60,
                    headers={"Authorization": f"Bearer {self.cle}", "Content-Type": "application/json"},
                    json={"model": modele, "messages": self.messages, "tools": self.outils,
                          "tool_choice": "auto", "temperature": 0.5, "max_tokens": 500})
                if r.status_code == 200:
                    self.modele_ok = modele
                    return r.json()
                derniere = GroqErreur(f"Groq {r.status_code} : {r.text[:200]}")
                if r.status_code == 400 and "tool_use_failed" in r.text and essai == 0:
                    continue                      # le modèle a mal formé l'appel d'outil : on réessaie
                break
            if r.status_code in (404, 429) or "decommissioned" in r.text or "model_not_found" in r.text:
                continue                          # modèle indisponible ou limite atteinte : on essaie le suivant
            raise derniere
        raise derniere

    def send_message(self, texte):
        debut = len(self.messages)
        self.messages.append({"role": "user", "content": texte})
        try:
            for _ in range(8):
                msg = self._requete()["choices"][0]["message"]
                appels = msg.get("tool_calls")
                if not appels:
                    rep = (msg.get("content") or "").strip() or "C'est fait."
                    self.messages.append({"role": "assistant", "content": rep})
                    self._tailler()
                    return Reponse(rep)
                self.messages.append({"role": "assistant", "content": msg.get("content") or "",
                                      "tool_calls": appels})
                for a in appels:
                    nom = a["function"]["name"]
                    try:
                        args = json.loads(a["function"].get("arguments") or "{}")
                        fn = OUTILS_PAR_NOM[nom]
                        resultat = str(fn(**args))
                    except Exception as e:
                        resultat = f"Erreur : {e}"
                    self.messages.append({"role": "tool", "tool_call_id": a["id"],
                                          "content": resultat[:2000]})
            self.messages.append({"role": "assistant", "content": "J'ai fait ce que j'ai pu."})
            return Reponse("J'ai fait ce que j'ai pu.")
        except Exception:
            del self.messages[debut:]             # annule l'échange raté pour ne pas casser l'historique
            raise


class Cerveau:
    """Gemini en priorité ; si son quota est atteint (ou s'il est saturé), bascule sur Groq."""

    def __init__(self):
        self.gemini = Discussion()
        self.groq = GroqChat()
        self.pause_gemini_jusqua = 0.0
        if self.groq.disponible():
            print("Groq : clé trouvée, bascule automatique activée.")
        else:
            print("Groq : pas de clé GROQ_API_KEY, pas de cerveau de secours.")

    def send_message(self, texte):
        if time.time() >= self.pause_gemini_jusqua or not self.groq.disponible():
            try:
                rep = self.gemini.send_message(texte)
                self.groq.noter(texte, rep.text)
                return rep
            except Exception as e:
                s = str(e)
                sature = any(k in s for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded"))
                if not (sature and self.groq.disponible()):
                    raise
                par_jour = "perday" in s.lower().replace(" ", "")
                self.pause_gemini_jusqua = time.time() + (3600 if par_jour else 90)
                print(f"[Gemini saturé ({'quota du jour' if par_jour else 'limite par minute'}) : "
                      f"je passe sur Groq]")
        return self.groq.send_message(texte)


def creer_chat():
    return _client(), Cerveau()


def demander_ia(client, chat, texte):
    """Interroge le cerveau et traduit les erreurs en phrases claires."""
    erreur = None
    for essai in range(2):
        try:
            return chat.send_message(texte).text or "C'est fait."
        except Exception as e:
            erreur = e
            if any(k in str(e) for k in ("500", "502", "503", "UNAVAILABLE", "overloaded")) and essai == 0:
                time.sleep(1.5)
                continue
            break
    s = str(erreur)
    print(f"[Erreur IA : {type(erreur).__name__}: {s[:300]}]")
    if "429" in s or "RESOURCE_EXHAUSTED" in s:
        return "J'ai atteint la limite gratuite. Réessaie dans une minute."
    if "404" in s or "NOT_FOUND" in s:
        return "Le nom du modèle dans mon fichier n'est pas valide."
    if any(k in s for k in ("API key", "401", "403", "PERMISSION_DENIED", "UNAUTHENTICATED")):
        return "Ma clé API est refusée."
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
    g = GroqChat()
    if not g.disponible():
        print("GROQ_API_KEY n'est pas définie.")
        return
    t0 = time.time()
    try:
        print("Groq répond :", g.send_message("Dis bonjour en une phrase.").text)
        print("Test des outils :", g.send_message("Quelle heure est-il ?").text)
        print(f"Modèle utilisé : {g.modele_ok} | Temps : {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"Erreur : {type(e).__name__}: {e}")


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
        parler(f"Bonjour {PRENOM}, Jarvis en ligne. Je t'écoute.")
        print(f"(Commence par « {NOMS[0]} ». Ensuite tu peux enchaîner pendant {CONVERSATION_SECONDES} s "
              "sans redire mon nom, et me couper quand je parle. Dis « merci » pour finir la discussion. "
              "Ctrl+C pour quitter)")
        actif_jusqua = 0.0
        while True:
            coupe = _INTERROMPU["oui"]
            if coupe:                                  # tu as coupé Jarvis : on reprend ta phrase
                pre, _INTERROMPU["oui"] = _INTERROMPU["audio"], False
                print("   (tu m'as coupé : je t'écoute)")
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
            else:
                print(f"   (ignoré : dis d'abord « {NOMS[0]} »)")
                continue
            if not traiter(commande, client, chat):
                break
            actif_jusqua = time.time() + CONVERSATION_SECONDES


if __name__ == "__main__":
    modes = {"test-cle": test_cle, "test-groq": test_groq, "test-voix": test_voix,
             "test-micro": test_micro, "test-outils": test_outils,
             "apprendre-voix": apprendre_voix, "test-id": test_id}
    try:
        modes.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
    except KeyboardInterrupt:
        print("\nArrêt de Jarvis.")