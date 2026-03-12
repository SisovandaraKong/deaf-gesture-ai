/**
 * app.js — UI logic for Sign Language Recognition
 *
 * Responsibilities:
 *  1. Show the MJPEG stream (the <img> tag handles that automatically)
 *  2. Poll /api/status + /api/sentence every 500 ms and update the sidebar
 *  3. Handle button clicks: Speak KM, Speak EN, Clear
 *  4. Handle keyboard shortcuts: Space, E, C
 *  5. Play TTS audio returned from /api/speak via HTML5 Audio API
 *  6. Animate confidence bar, hold-progress circle, sign flash
 */

"use strict";

// ── Constants ─────────────────────────────────────────────────────────────────
const POLL_INTERVAL_MS  = 500;   // How often to fetch state from Flask
const HOLD_CIRCUMFERENCE = 2 * Math.PI * 20; // SVG circle r=20 → 125.66

// ── State ──────────────────────────────────────────────────────────────────────
let currentState = {
    current_sign:     "",
    confidence:       0,
    hold_progress:    0,
    sentence:         [],
    khmer_translation:"",
    status_message:   "Show your hand to start",
    is_speaking:      false,
};

let prevSign   = "";      // Detect when a new sign is first confirmed
let audioRef   = null;    // Currently playing Audio instance

// ── DOM References ─────────────────────────────────────────────────────────────
const elSign         = () => document.getElementById("current-sign");
const elConfFill     = () => document.getElementById("confidence-fill");
const elConfText     = () => document.getElementById("confidence-text");
const elHoldCircle   = () => document.getElementById("hold-circle");
const elSentence     = () => document.getElementById("sentence-text");
const elKhmer        = () => document.getElementById("khmer-text");
const elStatus       = () => document.getElementById("status-message");
const elSpinner      = () => document.getElementById("loading-spinner");
const elVideoOverlay = () => document.getElementById("video-overlay");
const elVideoContainer = () => document.getElementById("video-container");
const elBtnKm        = () => document.getElementById("btn-speak-km");
const elBtnEn        = () => document.getElementById("btn-speak-en");
const elBtnClear     = () => document.getElementById("btn-clear");

// ── Polling ────────────────────────────────────────────────────────────────────
async function pollState() {
    try {
        // Fetch both endpoints in parallel
        const [statusRes, sentenceRes] = await Promise.all([
            fetch("/api/status"),
            fetch("/api/sentence"),
        ]);

        if (!statusRes.ok || !sentenceRes.ok) return;

        const status   = await statusRes.json();
        const sentence = await sentenceRes.json();

        // Merge into a single state object
        currentState = { ...currentState, ...status, ...sentence };
        updateUI();
    } catch (_) {
        // Network errors are silently ignored (camera might still be starting)
    }
}

// ── UI Update ──────────────────────────────────────────────────────────────────
function updateUI() {
    const {
        current_sign, confidence, hold_progress,
        sentence, khmer_translation, status_message,
    } = currentState;

    // ── Current sign ────────────────────────────────────────────────────────
    const signEl = elSign();
    const displaySign = current_sign || "...";
    if (signEl.textContent !== displaySign) {
        signEl.textContent = displaySign;
        // Flash animation when a new sign appears
        if (current_sign && current_sign !== prevSign) {
            signEl.classList.remove("flash");
            void signEl.offsetWidth; // reflow to restart animation
            signEl.classList.add("flash");
        }
        prevSign = current_sign;
    }

    // ── Confidence bar ───────────────────────────────────────────────────────
    const pct = Math.round(confidence * 100);
    elConfFill().style.width = `${pct}%`;
    elConfText().textContent = `${pct}%`;

    // Color: green >80%, amber 50–80%, red <50%
    if (confidence > 0.8)       elConfFill().style.backgroundColor = "#00DC82";
    else if (confidence > 0.5)  elConfFill().style.backgroundColor = "#F59E0B";
    else                        elConfFill().style.backgroundColor = "#EF4444";

    // ── Hold-progress circle ─────────────────────────────────────────────────
    const circle = elHoldCircle();
    if (circle) {
        circle.style.strokeDashoffset =
            HOLD_CIRCUMFERENCE * (1 - (hold_progress || 0));
    }

    // ── Video container border glow when a sign is detected ──────────────────
    const container = elVideoContainer();
    if (container) {
        container.classList.toggle("active", !!current_sign);
    }

    // ── Sentence ─────────────────────────────────────────────────────────────
    const sentText = Array.isArray(sentence) && sentence.length > 0
        ? sentence.join(" ")
        : "(empty)";
    elSentence().textContent = sentText;

    // ── Khmer translation ────────────────────────────────────────────────────
    elKhmer().textContent = khmer_translation || "(will appear after sign)";

    // ── Status message ───────────────────────────────────────────────────────
    elStatus().textContent = status_message || "Show your hand to start";
}

// ── Video feed overlay ─────────────────────────────────────────────────────────
function initVideoOverlay() {
    const img     = document.getElementById("video-feed");
    const overlay = elVideoOverlay();

    // Hide the "Connecting…" overlay once the MJPEG stream loads
    img.addEventListener("load", () => {
        overlay && overlay.classList.add("hidden");
    }, { once: true });

    // If stream fails show a message instead of a broken image
    img.addEventListener("error", () => {
        if (overlay) {
            overlay.querySelector("p").textContent =
                "Camera unavailable — check server logs.";
        }
    });
}

// ── TTS Speak ──────────────────────────────────────────────────────────────────
async function speakText(lang) {
    // Determine what to speak
    let text;
    if (lang === "km") {
        text = currentState.khmer_translation
            || (currentState.sentence || []).join(" ")
            || "";
    } else {
        text = (currentState.sentence || []).join(" ") || "";
    }

    if (!text || text === "(empty)") {
        elStatus().textContent = "Nothing to speak yet.";
        return;
    }

    // Show loading overlay
    const spinner = elSpinner();
    spinner.classList.add("visible");
    disableSpeakButtons(true);

    try {
        const response = await fetch("/api/speak", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ text, lang }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            elStatus().textContent = `TTS error: ${err.error || response.status}`;
            return;
        }

        // Stop any currently playing audio
        if (audioRef) {
            audioRef.pause();
            URL.revokeObjectURL(audioRef.src);
        }

        // Create a Blob URL and play via HTML5 Audio
        const blob     = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        audioRef = new Audio(audioUrl);
        audioRef.play();

        audioRef.addEventListener("ended", () => {
            URL.revokeObjectURL(audioUrl);
            audioRef = null;
        }, { once: true });

    } catch (exc) {
        elStatus().textContent = `Network error: ${exc.message}`;
    } finally {
        spinner.classList.remove("visible");
        disableSpeakButtons(false);
    }
}

// ── Sentence clear ─────────────────────────────────────────────────────────────
async function clearSentence() {
    try {
        await fetch("/api/sentence/clear", { method: "POST" });
        // State will refresh on next poll cycle
    } catch (exc) {
        elStatus().textContent = `Clear error: ${exc.message}`;
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function disableSpeakButtons(state) {
    document.querySelectorAll(".speak-btn").forEach(btn => {
        btn.disabled = state;
    });
}

// ── Keyboard shortcuts ─────────────────────────────────────────────────────────
function initKeyboard() {
    document.addEventListener("keydown", (e) => {
        // Ignore events when focus is inside an input/textarea
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

        switch (e.code) {
            case "Space":
                e.preventDefault();
                speakText("km");
                break;
            case "KeyE":
                speakText("en");
                break;
            case "KeyC":
                clearSentence();
                break;
        }
    });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initVideoOverlay();
    initKeyboard();

    // Wire up buttons
    elBtnKm()?.addEventListener("click", () => speakText("km"));
    elBtnEn()?.addEventListener("click", () => speakText("en"));
    elBtnClear()?.addEventListener("click", clearSentence);

    // Start polling server state every 500 ms
    pollState();                                // immediate first call
    setInterval(pollState, POLL_INTERVAL_MS);   // recurring
});
