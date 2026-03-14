document.addEventListener("DOMContentLoaded", () => {
    const inputText = document.getElementById("input-text");
    const analyzeBtn = document.getElementById("analyze-btn");
    const status = document.getElementById("status");
    const loader = document.getElementById("loader");

    const resultCard = document.getElementById("result-card");
    const entityList = document.getElementById("entity-list");
    const tokenTableBody = document.getElementById("token-table-body");

    const missingBox = document.getElementById("missing-box");
    const missingText = document.getElementById("missing-text");
    const missingList = document.getElementById("missing-list");

    let expectedFields = ["ROLE", "SKILL", "LOC", "EXP", "SALARY"];

    const fieldGuidance = {
        ROLE: "add target role (e.g., Backend Developer)",
        SKILL: "add core skills (e.g., Python, AWS, SQL)",
        LOC: "add location (e.g., Hanoi, Remote)",
        EXP: "add years of experience (e.g., 3 years)",
        SALARY: "add salary expectation (e.g., $1500/month)",
    };

    init();

    async function init() {
        try {
            const response = await fetch("/labels");
            if (response.ok) {
                const data = await response.json();
                if (Array.isArray(data.required_fields) && data.required_fields.length > 0) {
                    expectedFields = data.required_fields;
                }
            }
        } catch (_) {
            // Keep fallback labels when /labels is not available.
        }
    }

    analyzeBtn.addEventListener("click", async () => {
        hideStatus();
        resultCard.classList.add("hidden");

        const text = inputText.value.trim();
        if (!text) {
            showStatus("Please enter text before running NER.", true);
            return;
        }

        showLoader(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                throw new Error("NER API error. Check model loading in backend logs.");
            }

            const data = await response.json();
            renderResults(data);
            showStatus("NER completed successfully.", false);
        } catch (error) {
            showStatus(error.message || "Failed to run NER.", true);
        } finally {
            showLoader(false);
        }
    });

    function renderResults(data) {
        const entities = Array.isArray(data.entities) ? data.entities : [];
        const tokens = Array.isArray(data.tokens) ? data.tokens : [];
        const tags = Array.isArray(data.tags) ? data.tags : [];

        entityList.innerHTML = "";
        tokenTableBody.innerHTML = "";
        missingList.innerHTML = "";
        missingBox.classList.add("hidden");

        if (entities.length === 0) {
            const empty = document.createElement("p");
            empty.className = "status warning";
            empty.textContent = "No entities detected from the current text.";
            entityList.appendChild(empty);
        } else {
            entities.forEach((entity) => {
                const pill = document.createElement("span");
                pill.className = "entity-pill";
                const score = Number.isFinite(entity.score) ? ` (${(entity.score * 100).toFixed(1)}%)` : "";
                pill.textContent = `${entity.type}: ${entity.text}${score}`;
                entityList.appendChild(pill);
            });
        }

        const rowCount = Math.min(tokens.length, tags.length);
        for (let i = 0; i < rowCount; i += 1) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>${i + 1}</td><td>${tokens[i]}</td><td>${tags[i]}</td>`;
            tokenTableBody.appendChild(tr);
        }

        renderMissingGuidance(entities);
        resultCard.classList.remove("hidden");
        resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderMissingGuidance(entities) {
        const detected = new Set(
            entities
                .map((entity) => String(entity.type || "").toUpperCase())
                .filter(Boolean)
        );

        const missing = expectedFields.filter((field) => !detected.has(field));
        if (missing.length === 0) {
            return;
        }

        missingText.textContent = "Model did not clearly detect these fields. Add more detail to improve extraction:";

        missing.forEach((field) => {
            const item = document.createElement("div");
            item.className = "missing-item";
            item.innerHTML = `<span class="badge-ner">${field}</span>${fieldGuidance[field] || "add more detail"}`;
            missingList.appendChild(item);
        });

        missingBox.classList.remove("hidden");
    }

    function showLoader(isLoading) {
        loader.classList.toggle("hidden", !isLoading);
    }

    function showStatus(message, isWarning) {
        status.textContent = message;
        status.classList.remove("hidden");
        status.classList.toggle("warning", Boolean(isWarning));
    }

    function hideStatus() {
        status.textContent = "";
        status.classList.add("hidden");
        status.classList.remove("warning");
    }
});
