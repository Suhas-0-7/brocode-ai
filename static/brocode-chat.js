// static/brocode-chat.js
document.addEventListener("DOMContentLoaded", () =>
{
    const openBtn = document.getElementById("brocode-button");
    const widget = document.getElementById("brocode-widget");
    const closeBtn = document.getElementById("brocode-close");
    const sendBtn = document.getElementById("brocode-send");
    const input = document.getElementById("brocode-input");
    const log = document.getElementById("brocode-chatlog");

    function addMessage (role, text)
    {
        const p = document.createElement("div");
        p.style.marginBottom = "8px";
        p.innerHTML = `<strong style="display:block">${role}</strong><div style="white-space:pre-wrap; margin-top:4px;">${text}</div>`;
        log.appendChild(p);
        log.scrollTop = log.scrollHeight;
    }

    openBtn.addEventListener("click", () =>
    {
        widget.style.display = "flex";
    });

    closeBtn.addEventListener("click", () =>
    {
        widget.style.display = "none";
    });

    async function sendQuery (q)
    {
        addMessage("You", q);
        addMessage("Brocode AI", "…thinking (querying live data)");
        try
        {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ q })
            });
            const j = await res.json();
            log.lastChild.remove(); // remove thinking
            if (j.answer)
            {
                addMessage("Brocode AI", j.answer);
            } else
            {
                addMessage("Brocode AI", "No answer returned.");
            }
        } catch (err)
        {
            addMessage("Brocode AI", "Error: " + String(err));
        }
    }

    sendBtn.addEventListener("click", () =>
    {
        const v = input.value.trim();
        if (!v) return;
        input.value = "";
        sendQuery(v);
    });

    input.addEventListener("keydown", (e) =>
    {
        if (e.key === "Enter" && !e.shiftKey)
        {
            e.preventDefault();
            sendBtn.click();
        }
    });

});
