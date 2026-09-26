/**
 * PROPERTY INTELLIGENCE — Property Match Client Script
 * Handles multi-attribute simulator submission, centroid proximity matching,
 * pre-fitted StandardScaler + log1p transformation, and nearest-neighbor peer lookup.
 */

document.addEventListener("DOMContentLoaded", function () {
    // 1. Check URL parameters for direct property ID query
    const urlParams = new URLSearchParams(window.location.search);
    const propIdFromUrl = urlParams.get("id");
    if (propIdFromUrl) {
        const queryInput = document.getElementById("queryPropId");
        if (queryInput) {
            queryInput.value = propIdFromUrl;
            fetchSimilarProperties(propIdFromUrl);
        }
    }

    // 2. Form Submission
    const form = document.getElementById("propertyMatchForm");
    if (form) {
        form.addEventListener("submit", function (e) {
            e.preventDefault();
            executePropertyMatch();
        });
    }

    // 3. Similar Properties Search
    const searchBtn = document.getElementById("btnFindSimilar");
    if (searchBtn) {
        searchBtn.addEventListener("click", function () {
            const queryId = document.getElementById("queryPropId").value.trim();
            if (queryId) fetchSimilarProperties(queryId);
        });
    }

    // 4. Random ID Button
    const randomBtn = document.getElementById("btnRandomId");
    if (randomBtn) {
        randomBtn.addEventListener("click", function () {
            const queryInput = document.getElementById("queryPropId");
            const randomVal = Math.floor(Math.random() * 5000) + 1;
            queryInput.value = randomVal;
            fetchSimilarProperties(randomVal);
        });
    }
});

// Preset Archetypes Loader
window.loadMatchPreset = function (type) {
    const presets = {
        compact: {
            price: 12000,
            sqft: 550,
            bhk: 1,
            bathrooms: 1,
            furnishing: 0,
        },
        midmarket: {
            price: 25000,
            sqft: 1000,
            bhk: 2,
            bathrooms: 2,
            furnishing: 1,
        },
        executive: {
            price: 32000,
            sqft: 850,
            bhk: 2,
            bathrooms: 2,
            furnishing: 2,
        },
        family: {
            price: 65000,
            sqft: 1750,
            bhk: 3,
            bathrooms: 3,
            furnishing: 1,
        },
        luxury: {
            price: 300000,
            sqft: 5500,
            bhk: 4,
            bathrooms: 5,
            furnishing: 2,
        },
    };

    const p = presets[type];
    if (!p) return;

    document.getElementById("inputPrice").value = p.price;
    document.getElementById("inputSqft").value = p.sqft;
    document.getElementById("inputBhk").value = p.bhk;
    document.getElementById("inputBathrooms").value = p.bathrooms;
    document.getElementById("inputFurnishing").value = p.furnishing;

    // Automatically execute match for immediate feedback
    executePropertyMatch();
};

// Execute Property Match via POST /api/property-match
function executePropertyMatch() {
    const loadingBox = document.getElementById("matchLoadingBox");
    const resultBox = document.getElementById("matchResultBox");
    const submitBtn = document.getElementById("btnMatchSubmit");

    loadingBox.style.display = "block";
    resultBox.style.display = "none";
    submitBtn.disabled = true;

    const payload = {
        price: parseFloat(document.getElementById("inputPrice").value) || 0,
        sqft: parseFloat(document.getElementById("inputSqft").value) || 0,
        bhk: parseFloat(document.getElementById("inputBhk").value) || 1,
        numBathrooms: parseFloat(document.getElementById("inputBathrooms").value) || 1,
        furnishing_tier: parseFloat(document.getElementById("inputFurnishing").value) || 0,
    };

    fetch("/api/property-match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    })
        .then((res) => {
            if (!res.ok) throw new Error("Match failed");
            return res.json();
        })
        .then((data) => {
            loadingBox.style.display = "none";
            submitBtn.disabled = false;
            renderMatchResult(data);
        })
        .catch((err) => {
            loadingBox.style.display = "none";
            submitBtn.disabled = false;
            resultBox.style.display = "block";
            resultBox.innerHTML = `
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid var(--rose); padding: 18px; border-radius: var(--radius-md); color: var(--rose);">
                    <div style="font-weight: 700; margin-bottom: 4px;">Simulation Error</div>
                    <div style="font-size: 0.9rem;">${err.message || 'Unable to compute nearest centroid.'}</div>
                </div>
            `;
        });
}

function renderMatchResult(data) {
    const box = document.getElementById("matchResultBox");
    box.style.display = "block";

    let warningsHtml = "";
    if (data.warnings && data.warnings.length > 0) {
        warningsHtml = `
            <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid var(--amber); padding: 14px 18px; border-radius: 4px; margin-bottom: 24px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: var(--amber); text-transform: uppercase; margin-bottom: 6px;">Empirical Distribution Notice:</div>
                ${data.warnings.map((w) => `<div style="font-size: 0.85rem; color: #CBD5E1; margin-bottom: 3px;">• ${w}</div>`).join("")}
            </div>
        `;
    }

    let compRowsHtml = "";
    if (data.comparison) {
        compRowsHtml = data.comparison
            .map(
                (c) => `
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 10px 14px; font-weight: 600; color: var(--text-main);">${c.attribute}</td>
                <td style="padding: 10px 14px; font-weight: 700; color: #FFF;">${c.your_property}</td>
                <td style="padding: 10px 14px; color: var(--emerald);">${c.cluster_typical}</td>
                <td style="padding: 10px 14px;">
                    <span style="font-size: 0.75rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; background: rgba(30, 41, 59, 0.6); color: ${c.status === 'Aligned' || c.status === 'Identical' ? 'var(--emerald)' : 'var(--amber)'};">
                        ${c.status}
                    </span>
                </td>
            </tr>
        `
            )
            .join("");
    }

    let distancesHtml = "";
    if (data.all_cluster_distances) {
        const maxDist = Math.max(...data.all_cluster_distances.map((x) => x.standardized_distance), 1);
        distancesHtml = data.all_cluster_distances
            .map(
                (cd) => `
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                    <span style="font-weight: ${cd.is_match ? '700' : '500'}; color: ${cd.is_match ? 'var(--emerald)' : 'var(--text-muted)'};">
                        ${cd.segment_name} ${cd.is_match ? '★ (Closest)' : ''}
                    </span>
                    <span class="mono" style="color: ${cd.is_match ? '#FFF' : 'var(--text-dim)'};">
                        Distance: ${cd.standardized_distance}
                    </span>
                </div>
                <div style="height: 6px; background: rgba(15, 23, 42, 0.8); border-radius: 999px; overflow: hidden;">
                    <div style="height: 100%; width: ${(cd.standardized_distance / maxDist) * 100}%; background: ${cd.is_match ? 'var(--emerald)' : '#334155'}; border-radius: 999px;"></div>
                </div>
            </div>
        `
            )
            .join("");
    }

    box.innerHTML = `
        <div style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 28px;">
            ${warningsHtml}

            <!-- Match Banner -->
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; padding-bottom: 20px; border-bottom: 1px solid var(--border);">
                <div>
                    <span style="font-size: 0.75rem; font-weight: 700; color: var(--emerald); text-transform: uppercase; letter-spacing: 0.05em;">Cluster Assignment Result</span>
                    <h2 style="font-size: 1.8rem; font-weight: 800; color: #FFF; margin: 4px 0 8px 0;">${data.headline}</h2>
                    <p style="color: var(--text-muted); font-size: 0.92rem; max-width: 700px; line-height: 1.5;">
                        ${data.explanation}
                    </p>
                </div>
                <div style="text-align: right; background: rgba(15, 23, 42, 0.8); padding: 12px 20px; border-radius: var(--radius-sm); border: 1px solid var(--border-light);">
                    <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Euclidean Distance</div>
                    <div class="mono" style="font-size: 1.6rem; font-weight: 800; color: var(--emerald);">${data.distance_to_centroid}</div>
                    <div style="font-size: 0.7rem; color: var(--text-dim);">Learned ${data.dimensions_evaluated || 5}D Space</div>
                </div>
            </div>

            <!-- Two-column Comparison & Ranking -->
            <div style="display: grid; grid-template-columns: 1.2fr 1fr; gap: 28px;">
                <!-- Column 1: Feature Alignment -->
                <div>
                    <h3 style="font-size: 1.05rem; font-weight: 700; margin-bottom: 12px; color: #FFF;">Feature Alignment vs Cluster Centroid</h3>
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem;">
                            <thead>
                                <tr style="border-bottom: 2px solid var(--border-light); text-align: left;">
                                    <th style="padding: 8px 14px; color: var(--text-dim);">Attribute</th>
                                    <th style="padding: 8px 14px; color: #FFF;">Your Input</th>
                                    <th style="padding: 8px 14px; color: var(--emerald);">Centroid</th>
                                    <th style="padding: 8px 14px; color: var(--text-dim);">Alignment</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${compRowsHtml}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Column 2: Centroid Proximity Ranking -->
                <div>
                    <h3 style="font-size: 1.05rem; font-weight: 700; margin-bottom: 12px; color: #FFF;">Proximity to All Clusters</h3>
                    <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 16px;">
                        Standardized Euclidean distance to every fitted K-Means cluster centroid:
                    </p>
                    ${distancesHtml}
                    <div style="font-size: 0.72rem; color: var(--text-dim); margin-top: 12px; text-align: right;">
                        *Lower distance indicates higher architectural & valuation similarity.
                    </div>
                </div>
            </div>
        </div>
    `;

    if (window.lucide) lucide.createIcons();
}

// Fetch Similar Properties via GET /api/similar/<id>
function fetchSimilarProperties(propId) {
    const container = document.getElementById("similarResultsContainer");
    container.innerHTML = `
        <div style="text-align: center; padding: 32px 0;">
            <div style="display: inline-block; width: 32px; height: 32px; border: 3px solid rgba(16, 185, 129, 0.2); border-top-color: var(--emerald); border-radius: 50%; animation: spin 1s infinite linear;"></div>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 12px;">Querying nearest neighbors in standardized similarity space...</p>
        </div>
    `;

    fetch(`/api/similar/${propId}`)
        .then((res) => {
            if (!res.ok) throw new Error("Property not found");
            return res.json();
        })
        .then((data) => {
            renderSimilarPeers(data);
        })
        .catch((err) => {
            container.innerHTML = `
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid var(--rose); padding: 16px; border-radius: var(--radius-md); color: var(--rose); font-size: 0.9rem;">
                    ${err.message || 'Error loading similar properties.'} Please check the Property ID.
                </div>
            `;
        });
}

function renderSimilarPeers(data) {
    const container = document.getElementById("similarResultsContainer");
    const peers = data.similar_properties || [];

    if (peers.length === 0) {
        container.innerHTML = `<p style="color: var(--text-muted);">No peer properties found.</p>`;
        return;
    }

    const query = data.query_property || {};

    const peerCards = peers
        .map(
            (p) => `
        <div style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 20px; border-left: 4px solid ${p.color || 'var(--emerald)'};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div>
                    <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase; font-weight: 700;">Property ID: ${p.id}</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #FFF; margin-top: 2px;">${p.price_formatted}</div>
                </div>
                <div style="text-align: right;">
                    <span style="background: rgba(16, 185, 129, 0.12); color: var(--emerald); font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 999px;">
                        d = ${p.euclidean_distance}
                    </span>
                    <div style="font-size: 0.7rem; color: var(--text-dim); margin-top: 2px;">Standardized Distance</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.82rem; margin-bottom: 14px; background: rgba(15, 23, 42, 0.5); padding: 10px; border-radius: 6px;">
                <div><span style="color: var(--text-dim);">Space:</span> <strong style="color: #FFF;">${p.sqft.toLocaleString()} sqft</strong></div>
                <div><span style="color: var(--text-dim);">Layout:</span> <strong style="color: #FFF;">${p.bhk} BHK / ${p.numBathrooms} ba</strong></div>
                <div><span style="color: var(--text-dim);">Status:</span> <strong style="color: #FFF;">${p.Status || 'N/A'}</strong></div>
                <div><span style="color: var(--text-dim);">City:</span> <strong style="color: #FFF;">${p.city || 'N/A'}</strong></div>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.75rem; color: ${p.color || 'var(--emerald)'}; font-weight: 700;">${p.cluster_name || `Cluster ${p.cluster}`}</span>
                <span style="font-size: 0.72rem; color: var(--text-dim);">${p.location || ''}</span>
            </div>
        </div>
    `
        )
        .join("");

    container.innerHTML = `
        <div style="margin-bottom: 20px; padding: 14px 18px; background: rgba(15, 23, 42, 0.8); border: 1px solid var(--border-light); border-radius: 6px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; font-weight: 700;">Query Subject:</span>
                <strong style="color: #FFF; margin-left: 6px;">Property ID: ${query.property_id || query.id || 'N/A'}</strong>
                <span style="color: var(--text-muted); margin-left: 10px;">(${query.price_formatted || ''} • ${query.sqft || ''} sqft • ${query.bhk || ''} BHK in ${query.city_clean || query.city || ''})</span>
            </div>
            <div style="font-size: 0.75rem; color: var(--emerald); font-weight: 700;">
                ✓ Query property strictly excluded from peer results
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px;">
            ${peerCards}
        </div>
    `;

    if (window.lucide) lucide.createIcons();
}
