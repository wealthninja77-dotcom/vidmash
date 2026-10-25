/**
 * VidMash Client-Side Utilities
 */

// Form Submission & Pipeline Launch
async function submitVideo(event) {
    if (event) event.preventDefault();
    
    const input = document.getElementById("video-url-input");
    const errorEl = document.getElementById("error-message");
    const submitBtn = document.getElementById("submit-button");
    const spinner = document.getElementById("button-spinner");
    const btnText = document.getElementById("button-text");

    if (!input || !input.value.trim()) {
        showError("Please enter a valid YouTube video URL.");
        return;
    }

    const url = input.value.trim();

    // Set loading state
    if (submitBtn) submitBtn.disabled = true;
    if (spinner) spinner.classList.remove("hidden");
    if (btnText) btnText.textContent = "Initiating Cortex Engine...";
    if (errorEl) errorEl.classList.add("hidden");

    try {
        const response = await fetch("/api/process", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ url: url }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || data.message || "Failed to start processing job.");
        }

        if (data.redirect) {
            window.location.href = data.redirect;
        } else if (data.job_id) {
            window.location.href = `/processing/${data.job_id}`;
        }
    } catch (err) {
        showError(err.message || "An unexpected error occurred. Please check the URL.");
        if (submitBtn) submitBtn.disabled = false;
        if (spinner) spinner.classList.add("hidden");
        if (btnText) btnText.textContent = "🚀 Transform Video";
    }
}

function showError(msg) {
    const errorEl = document.getElementById("error-message");
    if (errorEl) {
        errorEl.textContent = msg;
        errorEl.classList.remove("hidden");
    } else {
        alert(msg);
    }
}

// Copy to clipboard helper
function copyToClipboard(text) {
    if (!navigator.clipboard) {
        fallbackCopyTextToClipboard(text);
        return;
    }
    navigator.clipboard.writeText(text).then(() => {
        showToast("Copied to clipboard!");
    });
}

function fallbackCopyTextToClipboard(text) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
        document.execCommand('copy');
        showToast("Copied to clipboard!");
    } catch (err) {
        console.error('Fallback copy failed', err);
    }
    document.body.removeChild(textArea);
}

function showToast(message) {
    const toast = document.createElement("div");
    toast.className = "fixed bottom-5 right-5 z-50 bg-slate-900 text-white px-4 py-2.5 rounded-xl text-sm font-semibold shadow-lg border border-slate-700 animate-bounce";
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.remove();
    }, 2500);
}
