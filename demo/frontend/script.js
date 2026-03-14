document.addEventListener("DOMContentLoaded", () => {
    const inputText = document.getElementById("input-text");
    const analyzeBtn = document.getElementById("analyze-btn");
    const status = document.getElementById("status");
    const loader = document.getElementById("loader");

    const resultCard = document.getElementById("result-card");
    const entityList = document.getElementById("entity-list");
    const skillCoverage = document.getElementById("skill-coverage");
    const matchGrid = document.getElementById("match-grid");

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
            const response = await fetch("/match", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                throw new Error("Matching API error. Check model loading in backend logs.");
            }

            const data = await response.json();
            renderResults(data);
            showStatus("Matching completed successfully.", false);
        } catch (error) {
            showStatus(error.message || "Failed to run matching.", true);
        } finally {
            showLoader(false);
        }
    });

    function renderResults(data) {
        const entities = Array.isArray(data.candidate_entities) ? data.candidate_entities : [];
        const matches = Array.isArray(data.matches) ? data.matches : [];
        const coverage = Array.isArray(data.skill_coverage) ? data.skill_coverage : [];

        entityList.innerHTML = "";
        skillCoverage.innerHTML = "";
        matchGrid.innerHTML = "";
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

    renderSkillCoverage(coverage);
    renderMatches(matches);

        renderMissingGuidance(entities);
        resultCard.classList.remove("hidden");
        resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderMatches(matches) {
        if (matches.length === 0) {
            const empty = document.createElement("p");
            empty.className = "status warning";
            empty.textContent = "No company matches available for the current input.";
            matchGrid.appendChild(empty);
            return;
        }

        matches.slice(0, 6).forEach((match) => {
            const card = document.createElement("article");
            card.className = "match-card";

            const matched = Array.isArray(match.matched_skills) ? match.matched_skills : [];
            const missing = Array.isArray(match.missing_skills) ? match.missing_skills : [];

            const matchedHtml = matched
                .map((skill) => `<span class="pill ok">${escapeHtml(skill)}</span>`)
                .join("");
            const missingHtml = missing
                .map((skill) => `<span class="pill miss">${escapeHtml(skill)}</span>`)
                .join("");

            card.innerHTML = `
                <div class="match-head">
                    <strong>${escapeHtml(match.company_name || "Unknown company")}</strong>
                    <span class="score">${Number(match.match_score || 0).toFixed(1)}%</span>
                </div>
                <p class="skill-ratio">Matched skills: ${Number(match.matched_skill_count || 0)}/${Number(match.required_skill_count || 0)}</p>
                <p>${escapeHtml(match.description || "No description available.")}</p>
                <div class="pills">${matchedHtml}${missingHtml}</div>
            `;
            matchGrid.appendChild(card);
        });
    }

    function renderSkillCoverage(coverage) {
        if (coverage.length === 0) {
            const empty = document.createElement("p");
            empty.className = "status warning";
            empty.textContent = "No detected skills available to compare across companies.";
            skillCoverage.appendChild(empty);
            return;
        }

        coverage.forEach((item) => {
            const row = document.createElement("article");
            row.className = "coverage-item";
            const count = Number(item.match_count || 0);
            const total = Number(item.total_companies || 0);
            const ratio = total > 0 ? (count / total) * 100 : 0;

            row.innerHTML = `
                <div class="coverage-head">
                    <strong>${escapeHtml(item.skill || "unknown")}</strong>
                    <span>${count}/${total} companies</span>
                </div>
                <div class="coverage-bar"><span style="width:${ratio.toFixed(1)}%"></span></div>
            `;
            skillCoverage.appendChild(row);
        });
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

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }
});
