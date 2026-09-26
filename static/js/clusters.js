/**
 * PROPERTY INTELLIGENCE — Clusters Page Client Script
 * Visualizes K Discovery Lab (Elbow & Silhouette), K-Means mechanical stepper,
 * and 2D PCA property landscape.
 */

let currentStep = 1;
const stepDetails = {
    1: {
        title: "Stage 1 — Initialize Centers",
        description: "Initial cluster centroids are positioned across the standardized feature space using the k-means++ probabilistic heuristic, ensuring dispersed initial seeds.",
        visual: `
            <div style="display: flex; justify-content: center; align-items: center; gap: 40px; padding: 20px;">
                <div style="text-align: center;">
                    <div style="width: 50px; height: 50px; border-radius: 50%; border: 3px dashed var(--emerald); display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto; color: var(--emerald);">
                        <i data-lucide="crosshair"></i>
                    </div>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">Center μ₁ (Seed)</span>
                </div>
                <div style="font-size: 1.5rem; color: var(--text-dim);">•••</div>
                <div style="text-align: center;">
                    <div style="width: 50px; height: 50px; border-radius: 50%; border: 3px dashed #38BDF8; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto; color: #38BDF8;">
                        <i data-lucide="crosshair"></i>
                    </div>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">Center μ₂ (Seed)</span>
                </div>
            </div>
        `
    },
    2: {
        title: "Stage 2 — Assign Properties",
        description: "Every property in the dataset computes its standardized Euclidean distance to all candidate cluster centers and is assigned to its closest centroid partition.",
        visual: `
            <div style="display: flex; justify-content: center; align-items: center; gap: 32px; padding: 20px;">
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid var(--emerald); border-radius: 8px; padding: 14px 20px; text-align: center;">
                    <span style="color: var(--emerald); font-weight: 700;">d(x, μ₁) < d(x, μ₂)</span><br>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">Assign to Cluster 1</span>
                </div>
                <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid #38BDF8; border-radius: 8px; padding: 14px 20px; text-align: center;">
                    <span style="color: #38BDF8; font-weight: 700;">d(x, μ₂) < d(x, μ₁)</span><br>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">Assign to Cluster 2</span>
                </div>
            </div>
        `
    },
    3: {
        title: "Stage 3 — Recalculate Centers",
        description: "Each cluster center is recalculated as the mathematical mean vector of all property instances currently assigned to that cluster.",
        visual: `
            <div style="text-align: center; padding: 18px;">
                <div style="font-family: monospace; font-size: 1.15rem; color: var(--emerald); margin-bottom: 8px;">
                    μ_j = (1 / |S_j|) ∑_{x ∈ S_j} x
                </div>
                <span style="font-size: 0.85rem; color: var(--text-muted);">Centroid moves toward the gravitational center of its assigned property cohort.</span>
            </div>
        `
    },
    4: {
        title: "Stage 4 — Stabilize Clusters",
        description: "Stages 2 and 3 repeat iteratively until cluster memberships cease to shift and Within-Cluster Sum of Squares (Inertia) reaches mathematical convergence.",
        visual: `
            <div style="display: flex; justify-content: center; align-items: center; gap: 16px; padding: 20px;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: rgba(16, 185, 129, 0.2); color: var(--emerald); display: flex; align-items: center; justify-content: center;">
                    <i data-lucide="check-circle-2"></i>
                </div>
                <div style="text-align: left;">
                    <span style="font-weight: 700; color: #FFF; font-size: 1.05rem;">Stabilization Complete</span><br>
                    <span style="font-size: 0.82rem; color: var(--text-muted);">Cluster assignments stabilize. Optimal cluster boundaries discovered.</span>
                </div>
            </div>
        `
    }
};

document.addEventListener("DOMContentLoaded", function () {
    renderStep(1);
    loadKMetricsCharts();
    loadPCALandscape();
});

// K-Means Stepper Functions
function renderStep(step) {
    currentStep = step;
    const data = stepDetails[step];
    const container = document.getElementById("stepContent");
    if (!container) return;

    container.innerHTML = `
        <div style="margin-bottom: 8px; font-weight: 700; font-size: 1.15rem; color: #FFF;">${data.title}</div>
        <p style="color: var(--text-muted); font-size: 0.92rem; margin-bottom: 16px;">${data.description}</p>
        ${data.visual}
    `;

    // Update active nav indicators
    document.querySelectorAll(".step-indicator").forEach((el) => {
        const s = parseInt(el.getAttribute("data-step"));
        if (s === step) {
            el.classList.add("active");
        } else {
            el.classList.remove("active");
        }
    });

    document.getElementById("prevStepBtn").disabled = step === 1;
    const nextBtn = document.getElementById("nextStepBtn");
    if (step === 4) {
        nextBtn.innerHTML = '<i data-lucide="rotate-ccw" style="width: 14px; height: 14px;"></i> Replay';
    } else {
        nextBtn.innerHTML = 'Next Step <i data-lucide="chevron-right" style="width: 14px; height: 14px;"></i>';
    }

    window.refreshIcons();
}

window.goToStep = function (step) {
    renderStep(step);
};

window.stepNext = function () {
    if (currentStep < 4) {
        renderStep(currentStep + 1);
    } else {
        renderStep(1);
    }
};

window.stepPrev = function () {
    if (currentStep > 1) {
        renderStep(currentStep - 1);
    }
};

// Load K Discovery Curves (Plotly)
function loadKMetricsCharts() {
    fetch("/api/k-metrics")
        .then((res) => res.json())
        .then((data) => {
            const kVals = data.metrics.map((m) => m.K);
            const inertias = data.metrics.map((m) => m.Inertia);
            const silhouettes = data.metrics.map((m) => m["Silhouette Score"]);

            // 1. Elbow Chart
            const elbowTrace = {
                x: kVals,
                y: inertias,
                mode: "lines+markers",
                type: "scatter",
                name: "WCSS Inertia",
                line: { color: "#14B8A6", width: 3 },
                marker: { color: "#14B8A6", size: 7 },
            };

            const elbowHighlight = {
                x: [data.elbow_k],
                y: [data.metrics.find((m) => m.K === data.elbow_k).Inertia],
                mode: "markers",
                type: "scatter",
                name: "Elbow Inflection",
                marker: { color: "#F59E0B", size: 13, symbol: "diamond" },
            };

            const elbowLayout = {
                title: { text: "Elbow Method (Within-Cluster Sum of Squares)", font: { size: 14, color: "#F8FAFC" } },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(15, 23, 42, 0.6)",
                font: { color: "#CBD5E1", family: "Plus Jakarta Sans" },
                xaxis: { title: "K (Clusters)", gridcolor: "#1E293B", dtick: 1 },
                yaxis: { title: "WCSS Inertia", gridcolor: "#1E293B" },
                margin: { l: 60, r: 20, t: 40, b: 50 },
                legend: { orientation: "h", y: -0.25 },
            };

            Plotly.newPlot("elbowChart", [elbowTrace, elbowHighlight], elbowLayout, { responsive: true, displayModeBar: false });

            // 2. Silhouette Chart
            const silTrace = {
                x: kVals,
                y: silhouettes,
                mode: "lines+markers",
                type: "scatter",
                name: "Silhouette Curve",
                line: { color: "#10B981", width: 3 },
                marker: { color: "#10B981", size: 7 },
            };

            const selectedK = data.recommended_k || 5;
            const selectedMetric = data.metrics.find((m) => m.K === selectedK);
            const selectedScore = selectedMetric ? selectedMetric["Silhouette Score"] : (data.selected_sil_score || 0.3574);

            const selectedHighlight = {
                x: [selectedK],
                y: [selectedScore],
                mode: "markers",
                type: "scatter",
                name: `Selected K=${selectedK} (${selectedScore})`,
                marker: { color: "#10B981", size: 13, symbol: "circle", line: { color: "#FFF", width: 2 } },
            };

            const highestK = data.highest_sil_k || 2;
            const highestMetric = data.metrics.find((m) => m.K === highestK);
            const highestScore = highestMetric ? highestMetric["Silhouette Score"] : (data.highest_sil_score || 0.4521);

            const highestHighlight = {
                x: [highestK],
                y: [highestScore],
                mode: "markers",
                type: "scatter",
                name: `Highest Tested K=${highestK} (${highestScore})`,
                marker: { color: "#38BDF8", size: 13, symbol: "diamond", line: { color: "#FFF", width: 2 } },
            };

            const silLayout = {
                title: { text: "Silhouette Coefficient (Cluster Separation across K=2..10)", font: { size: 14, color: "#F8FAFC" } },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(15, 23, 42, 0.6)",
                font: { color: "#CBD5E1", family: "Plus Jakarta Sans" },
                xaxis: { title: "K (Clusters)", gridcolor: "#1E293B", dtick: 1 },
                yaxis: { title: "Silhouette Score", gridcolor: "#1E293B" },
                margin: { l: 60, r: 20, t: 40, b: 50 },
                legend: { orientation: "h", y: -0.25 },
            };

            Plotly.newPlot("silhouetteChart", [silTrace, selectedHighlight, highestHighlight], silLayout, { responsive: true, displayModeBar: false });
        })
        .catch((err) => console.error("Error loading K metrics charts:", err));
}

// Load 2D PCA Property Landscape (Plotly)
function loadPCALandscape() {
    fetch("/api/pca-landscape")
        .then((res) => res.json())
        .then((data) => {
            const tracesByCluster = {};

            data.points.forEach((pt) => {
                if (!tracesByCluster[pt.segment_name]) {
                    tracesByCluster[pt.segment_name] = {
                        x: [],
                        y: [],
                        text: [],
                        mode: "markers",
                        type: "scatter",
                        name: pt.segment_name,
                        marker: { color: pt.color, size: 6, opacity: 0.7 },
                    };
                }
                tracesByCluster[pt.segment_name].x.push(pt.x);
                tracesByCluster[pt.segment_name].y.push(pt.y);
                tracesByCluster[pt.segment_name].text.push(
                    `Rent: ${pt.price_formatted}<br>Area: ${pt.sqft.toLocaleString()} sqft<br>Layout: ${pt.bhk} BHK<br>Location: ${pt.location}, ${pt.city}<br>Segment: ${pt.segment_name}`
                );
            });

            const allTraces = Object.values(tracesByCluster);

            // Overlay Projected Centroids
            const centroidTrace = {
                x: data.centroids.map((c) => c.x),
                y: data.centroids.map((c) => c.y),
                text: data.centroids.map((c) => `Centroid: ${c.segment_name}`),
                mode: "markers+text",
                type: "scatter",
                name: "Community Centroids",
                textposition: "top center",
                marker: {
                    color: "#FFFFFF",
                    size: 14,
                    symbol: "cross",
                    line: { color: "#000000", width: 2 },
                },
            };
            allTraces.push(centroidTrace);

            const layout = {
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(15, 23, 42, 0.6)",
                font: { color: "#CBD5E1", family: "Plus Jakarta Sans" },
                xaxis: { title: `Principal Component 1 (${data.variance_ratio[0]}% Variance)`, gridcolor: "#1E293B" },
                yaxis: { title: `Principal Component 2 (${data.variance_ratio[1]}% Variance)`, gridcolor: "#1E293B" },
                margin: { l: 60, r: 20, t: 30, b: 60 },
                legend: { orientation: "h", y: -0.2 },
            };

            Plotly.newPlot("pcaLandscapePlot", allTraces, layout, { responsive: true, displayModeBar: true });
        })
        .catch((err) => console.error("Error loading PCA landscape:", err));
}
