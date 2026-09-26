/**
 * PROPERTY INTELLIGENCE — Compare Page Client Script
 * Handles interactive Segment A vs B side-by-side comparison and Plotly comparative charts
 * for Indian rental communities.
 */

let cachedProfiles = [];

document.addEventListener("DOMContentLoaded", function () {
    fetch("/api/cluster-profiles")
        .then((res) => res.json())
        .then((data) => {
            cachedProfiles = data.profiles;
            updateComparisonDiff();
            renderComparativeCharts(data.profiles);
        })
        .catch((err) => console.error("Error loading cluster profiles:", err));

    const selectA = document.getElementById("selectSegA");
    const selectB = document.getElementById("selectSegB");

    if (selectA && selectB) {
        selectA.addEventListener("change", updateComparisonDiff);
        selectB.addEventListener("change", updateComparisonDiff);
    }
});

function updateComparisonDiff() {
    if (!cachedProfiles || cachedProfiles.length === 0) return;

    const idA = parseInt(document.getElementById("selectSegA").value);
    const idB = parseInt(document.getElementById("selectSegB").value);

    const segA = cachedProfiles.find((p) => p.cluster_id === idA) || cachedProfiles[0];
    const segB = cachedProfiles.find((p) => p.cluster_id === idB) || cachedProfiles[cachedProfiles.length - 1];

    const box = document.getElementById("comparisonDiffBox");
    if (!box) return;

    // Mathematical Delta Formatting Helpers
    function formatDelta(diff, prefix = "", suffix = "", decimals = 0) {
        if (Math.abs(diff) < 0.0001) return `0${suffix} (Equal)`;
        const sign = diff > 0 ? "+" : "-";
        const valStr = Math.abs(diff).toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
        return `${sign}${prefix}${valStr}${suffix}`;
    }

    function formatPctDelta(diff, base) {
        if (!base || base <= 0 || Math.abs(diff) < 0.0001) return "0.0%";
        const pct = (diff / base) * 100;
        const sign = pct > 0 ? "+" : "-";
        return `${sign}${Math.abs(pct).toFixed(1)}%`;
    }

    const rentA = segA.price ? Number(segA.price.median) : Number(segA.avg_price || 0);
    const rentB = segB.price ? Number(segB.price.median) : Number(segB.avg_price || 0);
    const rentDiff = rentB - rentA;

    const areaA = segA.area ? Number(segA.area.median) : Number(segA.avg_living_area || 0);
    const areaB = segB.area ? Number(segB.area.median) : Number(segB.avg_living_area || 0);
    const areaDiff = areaB - areaA;

    const bhkA = segA.specs ? Number(segA.specs.typical_bhk) : Number(segA.avg_bedrooms || 0);
    const bhkB = segB.specs ? Number(segB.specs.typical_bhk) : Number(segB.avg_bedrooms || 0);
    const bhkDiff = bhkB - bhkA;

    const bathsA = segA.specs ? Number(segA.specs.typical_bathrooms) : Number(segA.avg_bathrooms || 0);
    const bathsB = segB.specs ? Number(segB.specs.typical_bathrooms) : Number(segB.avg_bathrooms || 0);
    const bathsDiff = bathsB - bathsA;

    const psqftA = segA.price_per_sqft ? Number(segA.price_per_sqft.median) : 0;
    const psqftB = segB.price_per_sqft ? Number(segB.price_per_sqft.median) : 0;
    const psqftDiff = psqftB - psqftA;

    const furnA = segA.furnishing ? Number(segA.furnishing.pct_furnished) : 0;
    const furnB = segB.furnishing ? Number(segB.furnishing.pct_furnished) : 0;
    const furnDiff = furnB - furnA;

    const cityA = segA.geography ? segA.geography.dominant_city : "N/A";
    const cityB = segB.geography ? segB.geography.dominant_city : "N/A";

    box.innerHTML = `
        <div style="display: grid; grid-template-columns: 1.5fr 1fr 1fr 1.2fr; gap: 16px; border-bottom: 2px solid var(--border-light); padding-bottom: 12px; margin-bottom: 12px; font-size: 0.85rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">
            <div>Rental Attribute</div>
            <div style="color: ${segA.color}; font-weight: 800;">${segA.segment_name}</div>
            <div style="color: ${segB.color}; font-weight: 800;">${segB.segment_name}</div>
            <div style="color: var(--emerald);">Comparative Delta (B vs A)</div>
        </div>

        <div style="display: grid; grid-template-columns: 1.5fr 1fr 1fr 1.2fr; gap: 16px; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 0.95rem;">
            <div style="font-weight: 600; color: var(--text-main);">Median Monthly Rent</div>
            <div>₹${Number(rentA).toLocaleString()}/mo</div>
            <div>₹${Number(rentB).toLocaleString()}/mo</div>
            <div style="font-weight: 700; color: ${rentDiff >= 0 ? 'var(--emerald)' : '#F59E0B'};">
                ${formatDelta(rentDiff, "₹", "/mo", 0)} (${formatPctDelta(rentDiff, rentA)})
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1.5fr 1fr 1fr 1.2fr; gap: 16px; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 0.95rem;">
            <div style="font-weight: 600; color: var(--text-main);">Median Living Area</div>
            <div>${Number(areaA).toLocaleString()} sqft</div>
            <div>${Number(areaB).toLocaleString()} sqft</div>
            <div style="font-weight: 700; color: ${areaDiff >= 0 ? 'var(--emerald)' : '#F59E0B'};">
                ${formatDelta(areaDiff, "", " sqft", 0)} (${formatPctDelta(areaDiff, areaA)})
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1.5fr 1fr 1fr 1.2fr; gap: 16px; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 0.95rem;">
            <div style="font-weight: 600; color: var(--text-main);">Typical Layout</div>
            <div>${bhkA} BHK</div>
            <div>${bhkB} BHK</div>
            <div style="font-weight: 700; color: ${bhkDiff >= 0 ? 'var(--emerald)' : '#F59E0B'};">
                ${formatDelta(bhkDiff, "", " BHK", 0)}
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1.5fr 1fr 1fr 1.2fr; gap: 16px; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 0.95rem;">
            <div style="font-weight: 600; color: var(--text-main);">Bathrooms</div>
            <div>${bathsA} Baths</div>
            <div>${bathsB} Baths</div>
            <div style="font-weight: 700; color: ${bathsDiff >= 0 ? 'var(--emerald)' : '#F59E0B'};">
                ${formatDelta(bathsDiff, "", " baths", 1)}
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1.5fr 1fr 1fr 1.2fr; gap: 16px; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 0.95rem;">
            <div style="font-weight: 600; color: var(--text-main);">Median Rent / SqFt</div>
            <div>₹${psqftA.toFixed(1)}/sqft</div>
            <div>₹${psqftB.toFixed(1)}/sqft</div>
            <div style="font-weight: 700; color: ${psqftDiff >= 0 ? 'var(--emerald)' : '#F59E0B'};">
                ${formatDelta(psqftDiff, "₹", "/sqft", 1)}
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1.5fr 1fr 1fr 1.2fr; gap: 16px; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 0.95rem;">
            <div style="font-weight: 600; color: var(--text-main);">Furnished Share</div>
            <div>${furnA}% Fully Furnished</div>
            <div>${furnB}% Fully Furnished</div>
            <div style="font-weight: 700; color: ${furnDiff >= 0 ? 'var(--emerald)' : '#F59E0B'};">
                ${formatDelta(furnDiff, "", "%", 1)}
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1.5fr 1fr 1fr 1.2fr; gap: 16px; padding: 10px 0; font-size: 0.95rem;">
            <div style="font-weight: 600; color: var(--text-main);">Dominant Metro</div>
            <div>${cityA}</div>
            <div>${cityB}</div>
            <div style="font-weight: 600; color: var(--text-dim);">
                ${cityA === cityB ? 'Same Primary Metro' : 'Different Geographic Focus'}
            </div>
        </div>
    `;
}

function renderComparativeCharts(profiles) {
    const names = profiles.map((p) => p.segment_name);
    const colors = profiles.map((p) => p.color);
    const prices = profiles.map((p) => (p.price ? p.price.median : (p.avg_price || 0)));
    const areas = profiles.map((p) => (p.area ? p.area.median : (p.avg_living_area || 0)));

    // 1. Price Bar Chart
    const priceTrace = {
        x: names,
        y: prices,
        type: "bar",
        marker: { color: colors },
        text: prices.map((pr) => `₹${Number(pr).toLocaleString()}`),
        textposition: "auto",
    };
    const priceLayout = {
        title: { text: "Median Monthly Rent (₹) by Segment", font: { size: 14, color: "#F8FAFC" } },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(15, 23, 42, 0.6)",
        font: { color: "#CBD5E1", family: "Plus Jakarta Sans" },
        yaxis: { title: "Monthly Rent (₹)", gridcolor: "#1E293B" },
        margin: { l: 60, r: 20, t: 40, b: 60 },
    };
    Plotly.newPlot("chartPriceComp", [priceTrace], priceLayout, { responsive: true, displayModeBar: false });

    // 2. Living Area Bar Chart
    const areaTrace = {
        x: names,
        y: areas,
        type: "bar",
        marker: { color: colors },
        text: areas.map((ar) => `${Number(ar).toLocaleString()} sqft`),
        textposition: "auto",
    };
    const areaLayout = {
        title: { text: "Median Living Area (sqft) by Segment", font: { size: 14, color: "#F8FAFC" } },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(15, 23, 42, 0.6)",
        font: { color: "#CBD5E1", family: "Plus Jakarta Sans" },
        yaxis: { title: "Living Area (sqft)", gridcolor: "#1E293B" },
        margin: { l: 60, r: 20, t: 40, b: 60 },
    };
    Plotly.newPlot("chartAreaComp", [areaTrace], areaLayout, { responsive: true, displayModeBar: false });
}
